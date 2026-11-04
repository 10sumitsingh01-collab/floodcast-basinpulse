"""BASINSHIFT model (Lane C): LightGBM quantiles + calibrated event classifiers, one set per horizon.

    python basinshift_run.py --validate --folds 0,1 --model src.bs_models:BasinshiftModel --name lgbm
    python basinshift_run.py                        --model src.bs_models:BasinshiftModel --name lgbm

Design
- Target = log change from the latest observed flow: r_h = log1p(flow[t+h]) - log1p(flow_obs[t]).
  Predicting the change (not the level) transfers across basins whose flows differ by orders of magnitude.
- Point + range: LightGBM quantile regression at 0.1 / 0.5 / 0.9 on r_h, then a conformal widening
  (CQR) fitted on calibration basins, so the 10-90% range covers ~80% on basins the boosters never saw.
- Events: LightGBM classifiers for flow[t+h] > q_high / q_extreme of the basin, isotonic-calibrated
  on the calibration basins. p_extreme <= p_high enforced.
- Calibration basins: 20% of the training basins (seeded) are kept out of the boosters inside fit().
- Features (vectorised; every feature at t uses dates <= t only): past log-flow, rain lags / sums /
  antecedent index, temperature, snow and soil water dynamics, evaporation, season, static attributes,
  flow relative to the basin's thresholds.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from sklearn.isotonic import IsotonicRegression

from src.baselines import last_obs
from src.evaluate import DATE, HI, HORIZON, ID, LO, PE, PH, PRED

OBS = "streamflow_obs"
P, T = "total_precipitation_sum", "temperature_2m_mean"
SWE, SM1, SM3 = "snow_depth_water_equivalent_mean", "volumetric_soil_water_layer_1_mean", \
    "volumetric_soil_water_layer_3_mean"
PET = "potential_evaporation_sum_FAO_PENMAN_MONTEITH"
HORIZONS = (1, 3, 7)
QUANTILES = (0.1, 0.5, 0.9)
HISTORY_DAYS = 150   # look-back kept from train when predicting (longest window 60 d; API decays)
FLOW_FEATURES = {"anchor", "lq", "lq_lag1", "lq_lag2", "lq_lag3", "lq_lag7", "lq_mean3", "lq_mean7",
                 "lq_mean30", "lq_d1", "lq_d3", "obs_missing7", "anchor_rel_high", "anchor_rel_extreme",
                 "anchor_rel_pmean"}


class _Basins:
    """Row geometry of a frame sorted by (basin, date) with continuous daily dates."""

    def __init__(self, ids: np.ndarray):
        self.n = len(ids)
        codes, _ = pd.factorize(ids)
        first = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
        lens = np.diff(np.r_[first, self.n])
        self.start = np.repeat(first, lens)               # first row of this row's basin
        self.pos = np.arange(self.n) - self.start          # days since the basin's first row
        self.first, self.lens = first, lens

    def shift(self, x: np.ndarray, k: int) -> np.ndarray:
        out = np.full(self.n, np.nan, dtype=np.float64)
        out[k:] = x[:-k]
        out[self.pos < k] = np.nan
        return out

    def roll_sum(self, x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
        """Trailing sum and count of non-missing values over the last w days (within the basin)."""
        ok = np.isfinite(x)
        c = np.r_[0.0, np.cumsum(np.where(ok, x, 0.0))]
        k = np.r_[0, np.cumsum(ok)]
        i = np.arange(self.n)
        lo = np.maximum(i - w + 1, self.start)
        return c[i + 1] - c[lo], (k[i + 1] - k[lo]).astype(np.float64)

    def roll_mean(self, x: np.ndarray, w: int) -> np.ndarray:
        s, n = self.roll_sum(x, w)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, s / n, np.nan)

    def roll_max(self, x: np.ndarray, w: int) -> np.ndarray:
        stack = np.vstack([x] + [self.shift(x, k) for k in range(1, w)])
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return np.nanmax(stack, axis=0)

    def ewm(self, x: np.ndarray, alpha: float) -> np.ndarray:
        """y_t = (1 - alpha) y_{t-1} + alpha x_t, missing x treated as 0, restarted per basin."""
        out = np.empty(self.n)
        x0 = np.where(np.isfinite(x), x, 0.0)
        for f, L in zip(self.first, self.lens):
            out[f:f + L] = lfilter([alpha], [1, alpha - 1], x0[f:f + L])
        return out


def build_features(inputs: pd.DataFrame, static: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """inputs: gauge_id, date, streamflow_obs, weather (no targets). Returns features aligned to inputs' rows."""
    if any(c.startswith("streamflow_h") or c == "streamflow" for c in inputs.columns):
        raise ValueError("feature builder received target columns")
    order = np.lexsort((inputs[DATE].to_numpy(), inputs[ID].to_numpy()))
    d = inputs.iloc[order].reset_index(drop=True)
    gap = (d[DATE].diff().dt.days.to_numpy()[1:])[d[ID].to_numpy()[1:] == d[ID].to_numpy()[:-1]]
    if (gap != 1).any():
        raise ValueError("dates must be continuous daily within each basin")
    b = _Basins(d[ID].to_numpy())
    col = lambda c: d[c].to_numpy(np.float64)  # noqa: E731
    f: dict[str, np.ndarray] = {}

    anchor = last_obs(d[[ID, DATE, OBS]]).to_numpy(np.float64)   # trailing, <= 30 d forward fill
    f["anchor"] = np.log1p(np.clip(anchor, 0, None))
    lq = np.log1p(np.clip(col(OBS), 0, None))
    f["lq"] = lq
    for k in (1, 2, 3, 7):
        f[f"lq_lag{k}"] = b.shift(lq, k)
    for w in (3, 7, 30):
        f[f"lq_mean{w}"] = b.roll_mean(lq, w)
    f["lq_d1"], f["lq_d3"] = lq - f["lq_lag1"], lq - f["lq_lag3"]
    f["obs_missing7"] = b.roll_sum(np.isnan(lq).astype(float), 7)[0]

    p = col(P)
    f["p0"] = p
    for k in (1, 2, 3):
        f[f"p_lag{k}"] = b.shift(p, k)
    for w in (3, 7, 14, 30, 60):
        f[f"p_sum{w}"] = b.roll_sum(p, w)[0]
    f["p_max7"] = b.roll_max(p, 7)
    f["api"] = b.ewm(p, 0.1)
    t = col(T)
    f["t0"], f["tmax"] = t, col("temperature_2m_max")
    for w in (3, 7, 30):
        f[f"t_mean{w}"] = b.roll_mean(t, w)
    swe = col(SWE)
    f["swe"] = swe
    for k in (1, 3, 7):
        f[f"swe_d{k}"] = swe - b.shift(swe, k)
    sm1 = col(SM1)
    f["sm1"], f["sm3"] = sm1, col(SM3)
    f["sm1_mean7"] = b.roll_mean(sm1, 7)
    f["sm1_d3"] = sm1 - b.shift(sm1, 3)
    pet = col(PET)
    for w in (7, 30):
        f[f"pet_sum{w}"] = b.roll_sum(pet, w)[0]
    f["p_minus_pet30"] = f["p_sum30"] - f["pet_sum30"]
    f["rad"], f["dew"] = col("surface_net_solar_radiation_mean"), col("dewpoint_temperature_2m_mean")
    doy = d[DATE].dt.day_of_year.to_numpy()
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)

    ids = d[ID]
    st = static.set_index(ID)
    for c in st.columns:
        f[c] = ids.map(st[c]).to_numpy(np.float64)
    f["log_area"] = np.log(f["area_km2"])
    th = thresholds.set_index(ID)
    qh = np.log1p(ids.map(th["q_high"]).to_numpy(np.float64))
    qe = np.log1p(ids.map(th["q_extreme"]).to_numpy(np.float64))
    f["log_q_high"] = qh
    f["anchor_rel_high"], f["anchor_rel_extreme"] = f["anchor"] - qh, f["anchor"] - qe
    f["anchor_rel_pmean"] = f["anchor"] - np.log1p(f["p_mean"])

    out = pd.DataFrame({k: v.astype(np.float32) for k, v in f.items()})
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    out = out.iloc[inv].reset_index(drop=True)              # back to the caller's row order
    out.insert(0, DATE, inputs[DATE].to_numpy())
    out.insert(0, ID, inputs[ID].to_numpy())
    return out


def _lgb_params(seed: int, **kw) -> dict:
    p = dict(n_estimators=300, learning_rate=0.06, num_leaves=63, min_child_samples=200,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, max_bin=63,
             n_jobs=6, random_state=seed, verbose=-1)
    p.update(kw)
    return p


class BasinshiftModel:
    def __init__(self, seed: int = 42, max_rows: int = 300_000, cal_frac: float = 0.2,
                 use_static: bool = True, use_flow_history: bool = True, n_estimators: int = 300,
                 point: str = "median", mask_frac: float = 0.0):
        self.seed, self.max_rows, self.cal_frac = seed, max_rows, cal_frac
        self.mask_frac = mask_frac    # H4: share of training input values hidden at random
        self.use_static, self.use_flow_history, self.n_estimators = use_static, use_flow_history, n_estimators
        self.point = point            # "median": 0.5 quantile; "mean": L2 model + Duan smearing
        self.reg, self.clf, self.iso, self.cqr, self.smear = {}, {}, {}, {}, {}
        self.feature_cols: list[str] = []

    def _select(self, cols, static_cols) -> list[str]:
        keep = [c for c in cols if c not in (ID, DATE)]
        if not self.use_static:
            keep = [c for c in keep if c not in static_cols and c not in ("log_area", "anchor_rel_pmean")]
        if not self.use_flow_history:
            keep = [c for c in keep if c not in FLOW_FEATURES]
        return keep

    def _anchor(self, feats: pd.DataFrame) -> np.ndarray:
        """Log-flow the target is measured from: latest observation, or basin mean rain if ungauged."""
        a = feats["anchor"].to_numpy(float) if self.use_flow_history else np.log1p(feats["p_mean"].to_numpy(float))
        return np.where(np.isfinite(a), a, np.nanmedian(a))

    def fit(self, train: pd.DataFrame, static: pd.DataFrame, thresholds: pd.DataFrame) -> "BasinshiftModel":
        t0 = time.time()
        rng = np.random.default_rng(self.seed)
        train = train.reset_index(drop=True)
        inputs = train.drop(columns=[c for c in train.columns if c.startswith("streamflow_h") or c == "streamflow"])
        if self.mask_frac > 0:   # robustness training (H4): hide real values at random, never invent any
            inputs = inputs.copy()
            for c in inputs.columns.drop([ID, DATE]):
                inputs.loc[rng.random(len(inputs)) < self.mask_frac, c] = np.nan
        feats = build_features(inputs, static, thresholds)
        self.feature_cols = self._select(feats.columns, set(static.columns) - {ID})
        Y = np.column_stack([train[f"streamflow_h{h}"].to_numpy(float) for h in HORIZONS])
        thr_h = feats[ID].map(thresholds.set_index(ID)["q_high"]).to_numpy(float)
        thr_e = feats[ID].map(thresholds.set_index(ID)["q_extreme"]).to_numpy(float)
        basins = np.array(sorted(feats[ID].unique()))
        cal_b = set(rng.choice(basins, size=max(1, int(self.cal_frac * len(basins))), replace=False))
        is_cal = feats[ID].isin(cal_b).to_numpy()
        ok = np.isfinite(Y).all(axis=1) & np.isfinite(feats["anchor"].to_numpy())
        fit_idx = np.flatnonzero(ok & ~is_cal)
        cal_idx = np.flatnonzero(ok & is_cal)
        fit_idx = np.sort(rng.choice(fit_idx, min(len(fit_idx), self.max_rows), replace=False))
        cal_idx = np.sort(rng.choice(cal_idx, min(len(cal_idx), self.max_rows // 2), replace=False))
        anchor = self._anchor(feats)
        Xf = feats[self.feature_cols].iloc[fit_idx].to_numpy(np.float32)
        Xc = feats[self.feature_cols].iloc[cal_idx].to_numpy(np.float32)
        del feats, inputs
        print(f"  features ready ({time.time() - t0:.0f} s): fit {Xf.shape}, calibration {Xc.shape}, "
              f"{len(cal_b)} calibration basins", flush=True)

        for j, h in enumerate(HORIZONS):
            yh = Y[:, j]
            r = np.log1p(np.clip(yh, 0, None)) - anchor
            for q in QUANTILES:
                m = lgb.LGBMRegressor(objective="quantile", alpha=q,
                                      **_lgb_params(self.seed, n_estimators=self.n_estimators))
                m.fit(Xf, r[fit_idx])
                self.reg[(h, q)] = m
            lo, hi = self.reg[(h, 0.1)].predict(Xc), self.reg[(h, 0.9)].predict(Xc)
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
            score = np.maximum(lo - r[cal_idx], r[cal_idx] - hi)
            n = len(score)
            self.cqr[h] = float(np.quantile(score, min(1.0, np.ceil((n + 1) * 0.8) / n)))
            if self.point == "mean":
                m = lgb.LGBMRegressor(objective="regression",
                                      **_lgb_params(self.seed, n_estimators=self.n_estimators))
                m.fit(Xf, r[fit_idx])
                self.reg[(h, "mean")] = m
                # Duan smearing: E[exp(r)] = exp(pred) * mean(exp(residual)), estimated on calibration basins
                self.smear[h] = float(np.mean(np.exp(r[cal_idx] - m.predict(Xc))))
            for ev, tv in (("high", thr_h), ("extreme", thr_e)):
                lab = (yh > tv).astype(int)
                c = lgb.LGBMClassifier(objective="binary",
                                       **_lgb_params(self.seed, n_estimators=int(self.n_estimators * 0.8)))
                c.fit(Xf, lab[fit_idx])
                raw = c.predict_proba(Xc)[:, 1]
                self.clf[(h, ev)] = c
                self.iso[(h, ev)] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, lab[cal_idx])
            print(f"  horizon {h} trained ({time.time() - t0:.0f} s), CQR widening {self.cqr[h]:+.3f}", flush=True)
        return self

    def predict(self, model, test: pd.DataFrame, train: pd.DataFrame, static: pd.DataFrame,
                thresholds: pd.DataFrame) -> pd.DataFrame:
        cols = list(test.columns)
        start = test.groupby(ID)[DATE].min()
        hist = train.loc[train[ID].isin(start.index), cols]
        first_test = hist[ID].map(start)
        hist = hist[(hist[DATE] >= first_test - pd.Timedelta(days=HISTORY_DAYS)) & (hist[DATE] < first_test)]
        both = pd.concat([hist, test[cols]], ignore_index=True)
        feats = build_features(both, static, thresholds).iloc[len(hist):].reset_index(drop=True)
        X = feats[self.feature_cols].to_numpy(np.float32)
        anchor = self._anchor(feats)
        parts = []
        for h in HORIZONS:
            qs = np.sort(np.column_stack([self.reg[(h, q)].predict(X) for q in QUANTILES]), axis=1)
            lo, med, hi = qs[:, 0] - self.cqr[h], qs[:, 1], qs[:, 2] + self.cqr[h]
            out = pd.DataFrame({ID: test[ID].astype(str).to_numpy(), DATE: test[DATE].to_numpy()})
            out[HORIZON] = h
            out[LO] = np.clip(np.expm1(anchor + np.minimum(lo, med)), 0, None)
            out[HI] = np.clip(np.expm1(anchor + np.maximum(hi, med)), 0, None)
            if self.point == "mean":
                point = np.exp(anchor + self.reg[(h, "mean")].predict(X)) * self.smear[h] - 1
            else:
                point = np.expm1(anchor + med)
            out[PRED] = np.clip(point, out[LO], out[HI])   # forecast stays inside its own range
            ph = np.clip(self.iso[(h, "high")].predict(self.clf[(h, "high")].predict_proba(X)[:, 1]), 0, 1)
            pe = np.clip(self.iso[(h, "extreme")].predict(self.clf[(h, "extreme")].predict_proba(X)[:, 1]), 0, 1)
            out[PH], out[PE] = ph, np.minimum(pe, ph)
            parts.append(out)
        return pd.concat(parts, ignore_index=True)


class MeanPointModel(BasinshiftModel):
    """Point forecast from an L2 model with Duan smearing (less smoothing of peaks than the median)."""
    def __init__(self, **kw):
        super().__init__(point="mean", **kw)


class RobustModel(BasinshiftModel):
    """H4: trained with 15% of input values hidden at random, to degrade less when data is missing."""
    def __init__(self, **kw):
        kw.setdefault("mask_frac", 0.15)
        super().__init__(**kw)


class RobustMeanModel(BasinshiftModel):
    """H4 robustness training + mean point forecast."""
    def __init__(self, **kw):
        kw.setdefault("mask_frac", 0.15)
        super().__init__(point="mean", **kw)


class NoStaticModel(BasinshiftModel):
    """Ablation H1: identical model without static catchment attributes."""
    def __init__(self, **kw):
        super().__init__(use_static=False, **kw)


class NoFlowHistoryModel(BasinshiftModel):
    """Ungauged-basin variant: no observed-flow inputs at all."""
    def __init__(self, **kw):
        super().__init__(use_flow_history=False, **kw)
