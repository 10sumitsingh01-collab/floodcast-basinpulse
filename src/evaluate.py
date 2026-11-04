"""BASINSHIFT evaluation (Lane D), section 10 of the challenge documentation.

Inputs
  preds   long format, one row per (gauge_id, date, horizon):
          predicted_discharge, lower, upper, p_high, p_extreme
  labels  data/caravan/test_labels.parquet: gauge_id, date, test_set, streamflow_h1/_h3/_h7
  thr     data/caravan/thresholds.csv: gauge_id, q_high, q_extreme  (event = flow > threshold)

Output  tidy table, one row per (test_set, horizon, metric) with value, 95% bootstrap CI
        (resampling BASINS, 10.4), n_basins, n_rows, n_events.

Only this module and basinshift_run.py read test_labels.parquet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ID, DATE, HORIZON = "gauge_id", "date", "horizon"
PRED, LO, HI, PH, PE = "predicted_discharge", "lower", "upper", "p_high", "p_extreme"
PRED_COLS = [PRED, LO, HI, PH, PE]
TARGET_PREFIX = "streamflow_h"
MIN_OBS_PER_BASIN = 30          # NSE/KGE need a minimum record to be meaningful
INTERVAL_QUANTILES = (0.1, 0.9)  # nominal 80% interval
BASE_RATE = {"high": 0.05, "extreme": 0.01}  # by construction of the 95th/99th percentile thresholds
N_BINS_AP = 1000                # binned AUPRC for fast bootstrap
ALERT_BUDGETS = (10, 20)        # alert-days per basin-year for "precision at fixed alert budget" (10.1)


# ---------------------------------------------------------------------------------------------
# joining predictions to labels
# ---------------------------------------------------------------------------------------------
def labels_long(labels: pd.DataFrame) -> pd.DataFrame:
    """test_labels (wide, one column per horizon) -> one row per (gauge_id, date, horizon)."""
    hcols = [c for c in labels.columns if c.startswith(TARGET_PREFIX)]
    out = labels.melt(id_vars=[ID, DATE, "test_set"], value_vars=hcols, var_name=HORIZON, value_name="y_true")
    out[HORIZON] = out[HORIZON].str.replace(TARGET_PREFIX, "", regex=False).astype(int)
    return out


def join(preds: pd.DataFrame, labels: pd.DataFrame, thr: pd.DataFrame) -> pd.DataFrame:
    """Rows with an observed target only (missing observations are never scored or filled)."""
    lab = labels_long(labels).dropna(subset=["y_true"])
    df = lab.merge(preds[[ID, DATE, HORIZON] + PRED_COLS], on=[ID, DATE, HORIZON], how="left")
    missing = df[PRED].isna().sum()
    if missing:
        raise ValueError(f"{missing:,} labelled rows have no prediction; run the submission validator first")
    df = df.merge(thr[[ID, "q_high", "q_extreme"]], on=ID, how="left")
    if df["q_high"].isna().any():
        raise ValueError("some scored basins have no threshold in thresholds.csv")
    df["ev_high"] = df["y_true"] > df["q_high"]
    df["ev_extreme"] = df["y_true"] > df["q_extreme"]
    return df


# ---------------------------------------------------------------------------------------------
# per-basin continuous metrics
# ---------------------------------------------------------------------------------------------
def basin_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """MAE, NSE, KGE per basin (one test_set x horizon slice)."""
    def one(g: pd.DataFrame) -> pd.Series:
        y, p = g["y_true"].to_numpy(float), g[PRED].to_numpy(float)
        n = len(y)
        mae = float(np.mean(np.abs(y - p)))
        if n < MIN_OBS_PER_BASIN or np.var(y) == 0:
            return pd.Series({"n": n, "mae": mae, "nse": np.nan, "kge": np.nan})
        nse = 1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
        r = np.corrcoef(y, p)[0, 1] if np.std(p) > 0 else 0.0
        alpha = np.std(p) / np.std(y)
        beta = np.mean(p) / np.mean(y) if np.mean(y) != 0 else np.nan
        kge = 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        return pd.Series({"n": n, "mae": mae, "nse": float(nse), "kge": float(kge)})
    return df.groupby(ID).apply(one, include_groups=False)


# ---------------------------------------------------------------------------------------------
# pooled metrics computed from per-basin sufficient statistics (so the bootstrap is cheap)
# ---------------------------------------------------------------------------------------------
def _pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def basin_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Additive per-basin sums + per-basin histograms of event scores for binned AUPRC."""
    y, p = df["y_true"].to_numpy(float), df[PRED].to_numpy(float)
    lo, hi = df[LO].to_numpy(float), df[HI].to_numpy(float)
    q_lo, q_hi = INTERVAL_QUANTILES
    s = pd.DataFrame({
        ID: df[ID].to_numpy(),
        "n": 1.0,
        "abs_err": np.abs(y - p),
        "covered": ((y >= lo) & (y <= hi)).astype(float),
        "width": hi - lo,
        "pinball": (_pinball(y, lo, q_lo) + _pinball(y, p, 0.5) + _pinball(y, hi, q_hi)) / 3.0,
    })
    hists = {}
    for ev, pcol in (("high", PH), ("extreme", PE)):
        e = df[f"ev_{ev}"].to_numpy(bool)
        pr = np.clip(df[pcol].to_numpy(float), 0, 1)
        s[f"brier_{ev}"] = (pr - e) ** 2
        s[f"events_{ev}"] = e.astype(float)
        alert = pr >= 0.5
        s[f"tp_{ev}"] = (alert & e).astype(float)
        s[f"fp_{ev}"] = (alert & ~e).astype(float)
        # fixed alert budget (10.1): each basin may raise B alert-days per year, on its highest probabilities
        rank = pd.Series(pr).groupby(df[ID].to_numpy()).rank(ascending=False, method="first").to_numpy()
        n_basin = pd.Series(1, index=df.index).groupby(df[ID].to_numpy()).transform("size").to_numpy()
        for B in ALERT_BUDGETS:
            k = np.maximum(1, np.round(B * n_basin / 365.25))
            s[f"tpb{B}_{ev}"] = ((rank <= k) & e).astype(float)
            s[f"nb{B}_{ev}"] = (rank <= k).astype(float)
        # histogram of scores per basin: bins x {neg, pos}
        b = np.minimum((pr * N_BINS_AP).astype(int), N_BINS_AP - 1)
        codes, uniq = pd.factorize(df[ID].to_numpy())
        h = np.zeros((len(uniq), N_BINS_AP, 2), dtype=np.int64)
        np.add.at(h, (codes, b, e.astype(int)), 1)
        hists[ev] = h
    sums = s.groupby(ID, sort=False).sum()
    order = pd.factorize(df[ID].to_numpy())[1]
    return sums.loc[order], hists


def _binned_ap(h: np.ndarray) -> float:
    """Average precision from a (bins, 2) histogram of scores (neg, pos), thresholds high -> low."""
    neg, pos = h[::-1, 0].cumsum(), h[::-1, 1].cumsum()
    P = pos[-1]
    if P == 0:
        return np.nan
    prec = pos / np.maximum(pos + neg, 1)
    rec = pos / P
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def _pooled(sums: pd.DataFrame, hists: dict[str, np.ndarray] | None) -> dict[str, float]:
    t = sums.sum()
    n = t["n"]
    basin_years = n / 365.25
    out = {
        "mae_pooled": t["abs_err"] / n,
        "interval_coverage": t["covered"] / n,
        "interval_width": t["width"] / n,
        "pinball_loss": t["pinball"] / n,
    }
    for ev in ("high", "extreme"):
        bs = t[f"brier_{ev}"] / n
        ref = BASE_RATE[ev] * (1 - BASE_RATE[ev])
        tp, fp, ne = t[f"tp_{ev}"], t[f"fp_{ev}"], t[f"events_{ev}"]
        out[f"brier_{ev}"] = bs
        out[f"brier_skill_{ev}"] = 1 - bs / ref
        out[f"recall_{ev}"] = tp / ne if ne else np.nan
        out[f"precision_{ev}"] = tp / (tp + fp) if (tp + fp) else np.nan
        out[f"false_alarms_per_basin_year_{ev}"] = fp / basin_years if basin_years else np.nan
        for B in ALERT_BUDGETS:
            tpb, nb = t[f"tpb{B}_{ev}"], t[f"nb{B}_{ev}"]
            out[f"recall_at_budget{B}_{ev}"] = tpb / ne if ne else np.nan
            out[f"precision_at_budget{B}_{ev}"] = tpb / nb if nb else np.nan
        if hists is not None:
            out[f"auprc_{ev}"] = _binned_ap(hists[ev].sum(axis=0))
    return out


# ---------------------------------------------------------------------------------------------
# reliability table (10.1 "reliability diagrams")
# ---------------------------------------------------------------------------------------------
def reliability(df: pd.DataFrame, event: str = "high", n_bins: int = 10) -> pd.DataFrame:
    pcol = PH if event == "high" else PE
    p = np.clip(df[pcol].to_numpy(float), 0, 1)
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    t = pd.DataFrame({"bin": b, "p": p, "event": df[f"ev_{event}"].to_numpy(float)})
    r = t.groupby("bin").agg(mean_predicted=("p", "mean"), observed_frequency=("event", "mean"),
                             n=("p", "size")).reset_index()
    r["bin_range"] = [f"{i / n_bins:.1f}-{(i + 1) / n_bins:.1f}" for i in r["bin"]]
    return r[["bin_range", "mean_predicted", "observed_frequency", "n"]]


# ---------------------------------------------------------------------------------------------
# main entry: score everything with basin-bootstrap CIs
# ---------------------------------------------------------------------------------------------
def score(preds: pd.DataFrame, labels: pd.DataFrame, thr: pd.DataFrame,
          n_boot: int = 200, seed: int = 42) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    df = join(preds, labels, thr)
    rng = np.random.default_rng(seed)
    rows, rel = [], {}
    for (ts, h), g in df.groupby(["test_set", HORIZON], sort=True):
        sums, hists = basin_stats(g)
        bm = basin_metrics(g).loc[sums.index]  # same basin order, so one resample drives every metric
        nb = len(sums)
        point = {
            "nse_median": float(bm["nse"].median()),
            "kge_median": float(bm["kge"].median()),
            "mae_basin_mean": float(bm["mae"].mean()),
            **_pooled(sums, None),
        }
        for ev, pcol in (("high", PH), ("extreme", PE)):  # exact AUPRC for the point estimate
            e = g[f"ev_{ev}"].to_numpy(bool)
            point[f"auprc_{ev}"] = float(average_precision_score(e, g[pcol])) if e.any() else np.nan
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, nb, nb)
            bb = bm.iloc[idx]
            d = {"nse_median": bb["nse"].median(), "kge_median": bb["kge"].median(),
                 "mae_basin_mean": bb["mae"].mean()}
            d.update(_pooled(sums.iloc[idx], {k: v[idx] for k, v in hists.items()}))
            boots.append(d)
        boots = pd.DataFrame(boots)
        n_ev = {ev: int(g[f"ev_{ev}"].sum()) for ev in ("high", "extreme")}
        for m, v in point.items():
            lo, hi = (np.nanpercentile(boots[m], [2.5, 97.5]) if m in boots and boots[m].notna().any()
                      else (np.nan, np.nan))
            ev = "extreme" if m.endswith("extreme") else ("high" if m.endswith("high") else None)
            rows.append({"test_set": ts, "horizon": h, "metric": m, "value": v, "ci_low": lo, "ci_high": hi,
                         "n_basins": nb, "n_rows": len(g), "n_events": n_ev[ev] if ev else np.nan})
        rel[f"{ts}_h{h}"] = reliability(g, "high")
    return pd.DataFrame(rows), rel
