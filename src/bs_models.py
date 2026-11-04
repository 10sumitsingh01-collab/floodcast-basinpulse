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
from sklearn.isotonic import IsotonicRegression

from src.evaluate import DATE, HI, HORIZON, ID, LO, PE, PH, PRED

from src.model_features import (FLOW_FEATURES, OBS, OBS_AGE, build_features)  # noqa: F401

HORIZONS = (1, 3, 7)
QUANTILES = (0.1, 0.5, 0.9)
HISTORY_DAYS = 150   # look-back kept from train when predicting (longest window 60 d; API decays)


def _lgb_params(seed: int, **kw) -> dict:
    p = dict(n_estimators=300, learning_rate=0.06, num_leaves=63, min_child_samples=200,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, max_bin=63,
             n_jobs=6, random_state=seed, verbose=-1)
    p.update(kw)
    return p


class BasinshiftModel:
    def __init__(self, seed: int = 42, max_rows: int = 300_000, cal_frac: float = 0.2,
                 use_static: bool = True, use_flow_history: bool = True, n_estimators: int = 300,
                 point: str = "median", mask_frac: float = 0.0, delay_frac: float = 0.0,
                 chunk_threshold: int = 6_000_000, chunk_basins: int = 40):
        # above chunk_threshold training rows, features are built a few basins at a time (memory); the
        # in-memory path below is used otherwise and is unchanged from the validated runs
        self.chunk_threshold, self.chunk_basins = chunk_threshold, chunk_basins
        self.seed, self.max_rows, self.cal_frac = seed, max_rows, cal_frac
        self.delay_frac = delay_frac  # share of training rows seen with a 1-day-old flow reading (age flagged)
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

    def _collect_chunked(self, train, static, thresholds, rng):
        """Large training sets: build features basin-chunk by basin-chunk and keep only sampled rows."""
        aware = self.delay_frac > 0
        hcols = [f"streamflow_h{h}" for h in HORIZONS]
        basins = np.array(sorted(train[ID].unique()))
        cal_b = set(rng.choice(basins, size=max(1, int(self.cal_frac * len(basins))), replace=False))
        okY = train[hcols].notna().all(axis=1)
        n_cal = int((okY & train[ID].isin(cal_b)).sum())
        n_fit = int(okY.sum()) - n_cal
        pf, pc = min(1.0, self.max_rows / max(n_fit, 1)), min(1.0, (self.max_rows // 2) / max(n_cal, 1))
        thr = thresholds.set_index(ID)
        acc = {k: [] for k in ("Xf", "Yf", "af", "th_f", "te_f", "Xc", "Yc", "ac", "th_c", "te_c")}
        groups = dict(tuple(train.groupby(ID, sort=True)))
        for i in range(0, len(basins), self.chunk_basins):
            chunk = pd.concat([groups[b] for b in basins[i:i + self.chunk_basins]], ignore_index=True)
            inputs = chunk.drop(columns=[c for c in chunk.columns if c.startswith("streamflow_h") or c == "streamflow"])
            if self.mask_frac > 0:
                for c in inputs.columns.drop([ID, DATE]):
                    inputs.loc[rng.random(len(inputs)) < self.mask_frac, c] = np.nan
            variants = [(build_features(inputs, static, thresholds, include_obs_age=aware), 1.0)]
            if aware:
                late = inputs.copy()
                prev = late[[ID, DATE, OBS]].assign(**{DATE: late[DATE] + pd.Timedelta(days=1)})
                late = late.drop(columns=OBS).merge(prev, on=[ID, DATE], how="left")[inputs.columns]
                late[OBS_AGE] = 1.0
                variants.append((build_features(late, static, thresholds, include_obs_age=True), self.delay_frac))
            if not self.feature_cols:
                self.feature_cols = self._select(variants[0][0].columns, set(static.columns) - {ID})
            Y = chunk[hcols].to_numpy(float)
            th_h = chunk[ID].map(thr["q_high"]).to_numpy(float)
            th_e = chunk[ID].map(thr["q_extreme"]).to_numpy(float)
            is_cal = chunk[ID].isin(cal_b).to_numpy()
            for feats, share in variants:
                anc = self._anchor(feats)
                ok = np.isfinite(Y).all(axis=1) & np.isfinite(feats["anchor"].to_numpy())
                draw = rng.random(len(chunk))
                for tag, mask, prob in (("f", ~is_cal, pf), ("c", is_cal, pc)):
                    idx = np.flatnonzero(ok & mask & (draw < prob * share))
                    acc["X" + tag].append(feats[self.feature_cols].iloc[idx].to_numpy(np.float32))
                    acc["Y" + tag].append(Y[idx])
                    acc["a" + tag].append(anc[idx])
                    acc["th_" + tag].append(th_h[idx])
                    acc["te_" + tag].append(th_e[idx])
            del variants, inputs, chunk
        cat = {k: (np.vstack(v) if k[0] in "XY" else np.concatenate(v)) for k, v in acc.items()}
        return (cat["Xf"], cat["Yf"], cat["af"], (cat["th_f"], cat["te_f"]),
                cat["Xc"], cat["Yc"], cat["ac"], (cat["th_c"], cat["te_c"]), cal_b)

    def fit(self, train: pd.DataFrame, static: pd.DataFrame, thresholds: pd.DataFrame) -> "BasinshiftModel":
        t0 = time.time()
        rng = np.random.default_rng(self.seed)
        train = train.reset_index(drop=True)
        if len(train) > getattr(self, "chunk_threshold", 6_000_000):
            Xf, Yf, af, tf, Xc, Yc, ac, tc, cal_b = self._collect_chunked(train, static, thresholds, rng)
            print(f"  features ready, chunked ({time.time() - t0:.0f} s): fit {Xf.shape}, calibration {Xc.shape}, "
                  f"{len(cal_b)} calibration basins", flush=True)
            return self._train(Xf, Yf, af, tf, Xc, Yc, ac, tc, t0)
        inputs = train.drop(columns=[c for c in train.columns if c.startswith("streamflow_h") or c == "streamflow"])
        if self.mask_frac > 0:   # robustness training (H4): hide real values at random, never invent any
            inputs = inputs.copy()
            for c in inputs.columns.drop([ID, DATE]):
                inputs.loc[rng.random(len(inputs)) < self.mask_frac, c] = np.nan
        aware = self.delay_frac > 0
        feats = build_features(inputs, static, thresholds, include_obs_age=aware)
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
        Yf, Yc, af, ac = Y[fit_idx], Y[cal_idx], anchor[fit_idx], anchor[cal_idx]
        tf = (thr_h[fit_idx], thr_e[fit_idx])
        tc = (thr_h[cal_idx], thr_e[cal_idx])
        del feats
        if aware:  # the same origins, seen with the reading one day late and its age flagged
            late = inputs.copy()
            prev = late[[ID, DATE, OBS]].assign(**{DATE: late[DATE] + pd.Timedelta(days=1)})
            late = late.drop(columns=OBS).merge(prev, on=[ID, DATE], how="left")[inputs.columns]
            late[OBS_AGE] = 1.0
            fd = build_features(late, static, thresholds, include_obs_age=True)
            ad = self._anchor(fd)
            for which in ("fit", "cal"):
                base_idx = fit_idx if which == "fit" else cal_idx
                k = int(self.delay_frac * len(base_idx))
                sel = np.sort(rng.choice(base_idx, k, replace=False))
                sel = sel[np.isfinite(fd["anchor"].to_numpy()[sel])]
                Xd = fd[self.feature_cols].iloc[sel].to_numpy(np.float32)
                if which == "fit":
                    Xf, Yf, af = np.vstack([Xf, Xd]), np.vstack([Yf, Y[sel]]), np.r_[af, ad[sel]]
                    tf = (np.r_[tf[0], thr_h[sel]], np.r_[tf[1], thr_e[sel]])
                else:
                    Xc, Yc, ac = np.vstack([Xc, Xd]), np.vstack([Yc, Y[sel]]), np.r_[ac, ad[sel]]
                    tc = (np.r_[tc[0], thr_h[sel]], np.r_[tc[1], thr_e[sel]])
            del fd, late
        del inputs
        print(f"  features ready ({time.time() - t0:.0f} s): fit {Xf.shape}, calibration {Xc.shape}, "
              f"{len(cal_b)} calibration basins", flush=True)
        return self._train(Xf, Yf, af, tf, Xc, Yc, ac, tc, t0)

    def _train(self, Xf, Yf, af, tf, Xc, Yc, ac, tc, t0) -> "BasinshiftModel":
        for j, h in enumerate(HORIZONS):
            r_f = np.log1p(np.clip(Yf[:, j], 0, None)) - af
            r_c = np.log1p(np.clip(Yc[:, j], 0, None)) - ac
            for q in QUANTILES:
                m = lgb.LGBMRegressor(objective="quantile", alpha=q,
                                      **_lgb_params(self.seed, n_estimators=self.n_estimators))
                m.fit(Xf, r_f)
                self.reg[(h, q)] = m
            lo, hi = self.reg[(h, 0.1)].predict(Xc), self.reg[(h, 0.9)].predict(Xc)
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
            score = np.maximum(lo - r_c, r_c - hi)
            n = len(score)
            self.cqr[h] = float(np.quantile(score, min(1.0, np.ceil((n + 1) * 0.8) / n)))
            if self.point == "mean":
                m = lgb.LGBMRegressor(objective="regression",
                                      **_lgb_params(self.seed, n_estimators=self.n_estimators))
                m.fit(Xf, r_f)
                self.reg[(h, "mean")] = m
                # Duan smearing: E[exp(r)] = exp(pred) * mean(exp(residual)), estimated on calibration basins
                self.smear[h] = float(np.mean(np.exp(r_c - m.predict(Xc))))
            for e, ev in enumerate(("high", "extreme")):
                lab_f = (Yf[:, j] > tf[e]).astype(int)
                lab_c = (Yc[:, j] > tc[e]).astype(int)
                c = lgb.LGBMClassifier(objective="binary",
                                       **_lgb_params(self.seed, n_estimators=int(self.n_estimators * 0.8)))
                c.fit(Xf, lab_f)
                raw = c.predict_proba(Xc)[:, 1]
                self.clf[(h, ev)] = c
                self.iso[(h, ev)] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, lab_c)
            print(f"  horizon {h} trained ({time.time() - t0:.0f} s), CQR widening {self.cqr[h]:+.3f}", flush=True)
        return self

    def predict(self, model, test: pd.DataFrame, train: pd.DataFrame, static: pd.DataFrame,
                thresholds: pd.DataFrame) -> pd.DataFrame:
        cols = list(test.columns)
        start = test.groupby(ID)[DATE].min()
        hist = train.loc[train[ID].isin(start.index), [c for c in cols if c in train.columns]]
        first_test = hist[ID].map(start)
        hist = hist[(hist[DATE] >= first_test - pd.Timedelta(days=HISTORY_DAYS)) & (hist[DATE] < first_test)]
        both = pd.concat([hist, test[cols]], ignore_index=True)
        if OBS_AGE in both.columns:          # history rows are on-time readings
            both[OBS_AGE] = both[OBS_AGE].fillna(0.0)
        aware = self.delay_frac > 0
        feats = build_features(both, static, thresholds, include_obs_age=aware).iloc[len(hist):].reset_index(drop=True)
        X = feats[self.feature_cols].to_numpy(np.float32)
        anchor = self._anchor(feats)
        return self.predict_from_features(X, anchor, test[ID].astype(str).to_numpy(), test[DATE].to_numpy())

    def predict_from_features(self, X: np.ndarray, anchor: np.ndarray, ids: np.ndarray,
                              dates: np.ndarray) -> pd.DataFrame:
        """Prediction step on a ready feature matrix (columns = self.feature_cols); used by predict()
        and by the feature-importance analysis."""
        parts = []
        for h in HORIZONS:
            qs = np.sort(np.column_stack([self.reg[(h, q)].predict(X) for q in QUANTILES]), axis=1)
            lo, med, hi = qs[:, 0] - self.cqr[h], qs[:, 1], qs[:, 2] + self.cqr[h]
            out = pd.DataFrame({ID: ids, DATE: dates})
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


class BigRobustMeanModel(BasinshiftModel):
    """RobustMeanModel with twice the training sample and more trees (for use with extra training basins)."""
    def __init__(self, **kw):
        kw.setdefault("mask_frac", 0.15)
        kw.setdefault("max_rows", 600_000)
        kw.setdefault("n_estimators", 500)
        super().__init__(point="mean", **kw)


class DelayAwareModel(BasinshiftModel):
    """Final model + data-age input: 20% of training rows see a 1-day-old flow reading with its age flagged,
    so the model learns what a late reading means instead of trusting it as today's."""
    def __init__(self, **kw):
        kw.setdefault("mask_frac", 0.15)
        kw.setdefault("delay_frac", 0.2)
        super().__init__(point="mean", **kw)


class NoStaticModel(BasinshiftModel):
    """Ablation H1: identical model without static catchment attributes."""
    def __init__(self, **kw):
        super().__init__(use_static=False, **kw)


class NoFlowHistoryModel(BasinshiftModel):
    """Ungauged-basin variant: no observed-flow inputs at all."""
    def __init__(self, **kw):
        super().__init__(use_flow_history=False, **kw)
