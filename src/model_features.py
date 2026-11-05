"""Features used by the BASINSHIFT LightGBM model (src/bs_models.py).

    build_features(inputs, static, thresholds, include_obs_age=False) -> DataFrame
    FEATURE GROUPS: flow history, rain, snow, soil water, temperature / energy / evaporation, season,
                    catchment attributes, flow thresholds  (see scripts/feature_importance.py)

Every feature for the forecast origin at the end of day t uses data from dates <= t only, computed per basin over
continuous daily rows with vectorised trailing windows (cumulative sums, shifts, a per-basin recursive filter for
the antecedent precipitation index). Tests: tests/test_bs_models.py (truncation, row order, target guard).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from src.baselines import last_obs
from src.evaluate import DATE, ID

OBS = "streamflow_obs"
P, T = "total_precipitation_sum", "temperature_2m_mean"
SWE, SM1, SM3 = "snow_depth_water_equivalent_mean", "volumetric_soil_water_layer_1_mean", \
    "volumetric_soil_water_layer_3_mean"
PET = "potential_evaporation_sum_FAO_PENMAN_MONTEITH"
OBS_AGE = "obs_age_days"   # optional input: days between the latest flow reading and the origin
FLOW_FEATURES = {"anchor", "lq", "lq_lag1", "lq_lag2", "lq_lag3", "lq_lag7", "lq_mean3", "lq_mean7",
                 "lq_mean30", "lq_d1", "lq_d3", "obs_missing7", "anchor_rel_high", "anchor_rel_extreme",
                 "anchor_rel_pmean", OBS_AGE,
                 "sig_lq_mean90", "sig_lq_mean365", "sig_lq_std30", "sig_recession30", "sig_runoff_ratio365",
                 "sig_anchor_rel_mean365"}


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


def build_features(inputs: pd.DataFrame, static: pd.DataFrame, thresholds: pd.DataFrame,
                   include_obs_age: bool = False, include_signatures: bool = False) -> pd.DataFrame:
    """inputs: gauge_id, date, streamflow_obs, weather (no targets). Returns features aligned to inputs' rows.

    include_obs_age: add `obs_age_days`, how old the latest flow reading is at the origin (input column of the
    same name if present, else 0). Off by default so earlier models keep exactly the same feature set.
    include_signatures: add trailing flow signatures of the basin (mean level over 90/365 days, 30-day variability,
    recession rate on dry days, 365-day runoff ratio), all from dates <= t. Off by default."""
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
    if include_obs_age:
        f[OBS_AGE] = col(OBS_AGE) if OBS_AGE in d.columns else np.zeros(len(d))
    if include_signatures:   # how this basin behaves, from its own recent record (all trailing)
        f["sig_lq_mean90"] = b.roll_mean(lq, 90)
        f["sig_lq_mean365"] = b.roll_mean(lq, 365)
        m30, m2 = b.roll_mean(lq, 30), b.roll_mean(lq * lq, 30)
        f["sig_lq_std30"] = np.sqrt(np.clip(m2 - m30 * m30, 0, None))
        dry = np.isfinite(p) & (p < 1.0)
        f["sig_recession30"] = b.roll_mean(np.where(dry, f["lq_d1"], np.nan), 30)
        with np.errstate(invalid="ignore", divide="ignore"):
            f["sig_runoff_ratio365"] = b.roll_mean(np.expm1(lq), 365) / b.roll_mean(p, 365)
        f["sig_anchor_rel_mean365"] = f["anchor"] - f["sig_lq_mean365"]

    out = pd.DataFrame({k: v.astype(np.float32) for k, v in f.items()})
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    out = out.iloc[inv].reset_index(drop=True)              # back to the caller's row order
    out.insert(0, DATE, inputs[DATE].to_numpy())
    out.insert(0, ID, inputs[ID].to_numpy())
    return out
