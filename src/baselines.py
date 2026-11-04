"""Reference baseline for the BASINSHIFT pipeline (Lane D): calibrated persistence.

Same interface Lane C's models plug into (see basinshift_run.py --model):
    model = fit(train, static, thresholds)
    preds = predict(model, test, train, static, thresholds)   -> long format (src/submission.py)

Persistence: the forecast for t+h is the last observed flow at the origin t
(trailing forward-fill up to 30 days when the latest observation is missing).
Interval: training quantiles (10%, 90%) of log(flow[t+h] / flow[t]) per horizon.
Probabilities: training frequency of exceeding the basin's threshold, binned by flow[t] / threshold.
Everything is fitted on train.parquet only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.evaluate import DATE, HI, HORIZON, ID, LO, PE, PH, PRED

OBS, Y = "streamflow_obs", "streamflow"
HORIZONS = (1, 3, 7)
EPS = 0.01          # mm/day, keeps log ratios finite at zero flow
FFILL_DAYS = 30
N_BINS = 40


@dataclass
class Persistence:
    log_ratio_q: dict = field(default_factory=dict)   # horizon -> (q10, q90)
    edges: dict = field(default_factory=dict)         # event -> bin edges of obs/threshold
    probs: dict = field(default_factory=dict)         # (event, horizon) -> exceedance freq per bin
    fallback_flow: float = 0.0


def last_obs(history: pd.DataFrame) -> pd.Series:
    """Most recent observed flow at each origin, looking back at most FFILL_DAYS rows (daily data)."""
    h = history.sort_values([ID, DATE])
    s = h.groupby(ID)[OBS].ffill(limit=FFILL_DAYS).astype("float64")
    # still missing: running mean of the basin's earlier observations (trailing only)
    csum = h[OBS].fillna(0).groupby(h[ID]).cumsum()
    cnt = h[OBS].notna().astype(int).groupby(h[ID]).cumsum()
    s = s.fillna((csum / cnt.replace(0, np.nan)))
    return pd.Series(s.to_numpy(), index=pd.MultiIndex.from_frame(h[[ID, DATE]]))


def _bin(z: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, z, side="right") - 1, 0, len(edges) - 2)


def fit(train: pd.DataFrame, static: pd.DataFrame | None, thresholds: pd.DataFrame) -> Persistence:
    m = Persistence()
    tr = train.merge(thresholds[[ID, "q_high", "q_extreme"]], on=ID, how="left")
    obs = tr[OBS].to_numpy(float)
    m.fallback_flow = float(np.nanmedian(obs))
    for h in HORIZONS:
        y = tr[f"{Y}_h{h}"].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(obs)
        r = np.log((y[ok] + EPS) / (obs[ok] + EPS))
        m.log_ratio_q[h] = (float(np.quantile(r, 0.1)), float(np.quantile(r, 0.9)))
    for ev, tcol in (("high", "q_high"), ("extreme", "q_extreme")):
        z = obs / tr[tcol].to_numpy(float)
        zf = z[np.isfinite(z)]
        edges = np.unique(np.quantile(zf, np.linspace(0, 1, N_BINS + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        m.edges[ev] = edges
        for h in HORIZONS:
            y = tr[f"{Y}_h{h}"].to_numpy(float)
            ok = np.isfinite(y) & np.isfinite(z)
            b = _bin(z[ok], edges)
            hit = (y[ok] > tr[tcol].to_numpy(float)[ok]).astype(float)
            cnt = np.bincount(b, minlength=len(edges) - 1)
            pos = np.bincount(b, weights=hit, minlength=len(edges) - 1)
            m.probs[(ev, h)] = (pos + 0.5) / (cnt + 1.0)  # Jeffreys-style smoothing, avoids 0/1
    return m


def predict(m: Persistence, test: pd.DataFrame, train: pd.DataFrame, static: pd.DataFrame | None,
            thresholds: pd.DataFrame) -> pd.DataFrame:
    hist = pd.concat([train[[ID, DATE, OBS]], test[[ID, DATE, OBS]]], ignore_index=True)
    lo = last_obs(hist)
    base = test[[ID, DATE]].copy()
    q0 = lo.reindex(pd.MultiIndex.from_frame(base)).to_numpy(float)
    q0 = np.where(np.isfinite(q0), q0, m.fallback_flow)
    thr = base.merge(thresholds[[ID, "q_high", "q_extreme"]], on=ID, how="left")
    parts = []
    for h in HORIZONS:
        a, b = m.log_ratio_q[h]
        out = base.copy()
        out[HORIZON] = h
        out[PRED] = q0
        out[LO] = np.maximum((q0 + EPS) * np.exp(a) - EPS, 0)
        out[HI] = np.maximum((q0 + EPS) * np.exp(b) - EPS, 0)
        ph = m.probs[("high", h)][_bin(q0 / thr["q_high"].to_numpy(float), m.edges["high"])]
        pe = m.probs[("extreme", h)][_bin(q0 / thr["q_extreme"].to_numpy(float), m.edges["extreme"])]
        out[PH] = ph
        out[PE] = np.minimum(pe, ph)
        parts.append(out)
    p = pd.concat(parts, ignore_index=True)
    p[ID] = p[ID].astype(str)
    return p


# ------------------------------------------------------------------------------------------------
# Seasonal climatology baseline (BASINSHIFT 9.1 "Seasonal/climatological")
# ------------------------------------------------------------------------------------------------
def climatology_ref(history: pd.DataFrame, keys: pd.DataFrame, h: int) -> np.ndarray:
    """Mean observed flow in the calendar week of t+h, from the basin's observations on dates <= t.

    history: gauge_id, date, streamflow_obs (all inputs available, any period); keys: gauge_id, date (= t).
    Uses only observations dated <= t, so it is leakage-safe and works for basins with no training rows.
    Falls back to the basin's running mean (all weeks), then NaN.
    """
    hist = history[[ID, DATE, OBS]].drop_duplicates([ID, DATE]).dropna(subset=[OBS]).copy()
    hist["week"] = np.minimum((hist[DATE].dt.day_of_year - 1) // 7, 51)
    hist = hist.sort_values([ID, "week", DATE])
    g = hist.groupby([ID, "week"])[OBS]
    hist["wk_mean"] = g.cumsum() / g.cumcount().add(1)
    gb = hist.sort_values([ID, DATE]).groupby(ID)[OBS]
    q = keys[[ID, DATE]].copy()
    q["_row"] = np.arange(len(q))
    q["week"] = np.minimum(((q[DATE] + pd.Timedelta(days=h)).dt.day_of_year - 1) // 7, 51)
    right = hist[[ID, "week", DATE, "wk_mean"]].sort_values(DATE)
    m = pd.merge_asof(q.sort_values(DATE), right, on=DATE, by=[ID, "week"], direction="backward")
    # fallback: running mean of all earlier observations of the basin
    allm = hist.sort_values([ID, DATE])[[ID, DATE]].copy()
    allm["all_mean"] = gb.cumsum().to_numpy() / (gb.cumcount() + 1).to_numpy()
    m = pd.merge_asof(m.sort_values(DATE), allm.sort_values(DATE), on=DATE, by=ID, direction="backward")
    out = m.sort_values("_row")
    return out["wk_mean"].fillna(out["all_mean"]).to_numpy(float)


@dataclass
class Climatology:
    """Forecast = the basin's mean flow for that calendar week, from its own past observations.
    Range: training quantiles of log(flow / climatology); probabilities: binned by climatology / threshold."""
    log_ratio_q: dict = field(default_factory=dict)
    edges: dict = field(default_factory=dict)
    probs: dict = field(default_factory=dict)
    fallback_flow: float = 0.0

    def fit(self, train, static, thresholds):
        tr = train.reset_index(drop=True)
        thr = tr[[ID]].merge(thresholds[[ID, "q_high", "q_extreme"]], on=ID, how="left")
        self.fallback_flow = float(np.nanmedian(tr[OBS]))
        for h in HORIZONS:
            ref = climatology_ref(tr, tr, h)
            ref = np.where(np.isfinite(ref), ref, self.fallback_flow)
            y = tr[f"{Y}_h{h}"].to_numpy(float)
            ok = np.isfinite(y)
            r = np.log((y[ok] + EPS) / (ref[ok] + EPS))
            self.log_ratio_q[h] = (float(np.quantile(r, 0.1)), float(np.quantile(r, 0.9)))
            for ev, tcol in (("high", "q_high"), ("extreme", "q_extreme")):
                z = ref / thr[tcol].to_numpy(float)
                if (ev, "edges") not in self.edges:
                    zf = z[np.isfinite(z)]
                    e = np.unique(np.quantile(zf, np.linspace(0, 1, N_BINS + 1)))
                    e[0], e[-1] = -np.inf, np.inf
                    self.edges[(ev, "edges")] = e
                e = self.edges[(ev, "edges")]
                okz = ok & np.isfinite(z)
                b = _bin(z[okz], e)
                hit = (y[okz] > thr[tcol].to_numpy(float)[okz]).astype(float)
                cnt = np.bincount(b, minlength=len(e) - 1)
                pos = np.bincount(b, weights=hit, minlength=len(e) - 1)
                self.probs[(ev, h)] = (pos + 0.5) / (cnt + 1.0)
        return self

    def predict(self, model, test, train, static, thresholds):
        hist = pd.concat([train[[ID, DATE, OBS]], test[[ID, DATE, OBS]]], ignore_index=True)
        base = test[[ID, DATE]].reset_index(drop=True)
        thr = base[[ID]].merge(thresholds[[ID, "q_high", "q_extreme"]], on=ID, how="left")
        parts = []
        for h in HORIZONS:
            ref = climatology_ref(hist, base, h)
            ref = np.where(np.isfinite(ref), ref, self.fallback_flow)
            a, b = self.log_ratio_q[h]
            out = base.copy()
            out[HORIZON] = h
            out[PRED] = ref
            out[LO] = np.maximum((ref + EPS) * np.exp(a) - EPS, 0)
            out[HI] = np.maximum((ref + EPS) * np.exp(b) - EPS, 0)
            ph = self.probs[("high", h)][_bin(ref / thr["q_high"].to_numpy(float), self.edges[("high", "edges")])]
            pe = self.probs[("extreme", h)][_bin(ref / thr["q_extreme"].to_numpy(float),
                                                 self.edges[("extreme", "edges")])]
            out[PH] = ph
            out[PE] = np.minimum(pe, ph)
            parts.append(out)
        p = pd.concat(parts, ignore_index=True)
        p[ID] = p[ID].astype(str)
        return p
