"""ShiftGuard part 3: "why this alert?" - the drivers behind each forecast.

LightGBM splits every prediction into exact per-feature contributions (TreeSHAP, `pred_contrib=True`). We sum them
into hydrological groups and report, for each forecast, the groups that pushed the high-flow probability up the
most (in log-odds) and the share of the flow-change forecast they explain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluate import DATE, ID
from src.model_features import FLOW_FEATURES

GROUP_LABELS = {
    "flow": "recent river flow", "rain": "recent rain", "snow": "snowmelt / snowpack", "soil": "soil wetness",
    "energy": "temperature & evaporation", "season": "time of year", "basin": "catchment traits",
    "scale": "basin flood scale", "age": "age of the latest reading",
}


def group_of(feature: str, static_cols: set[str]) -> str:
    if feature == "obs_age_days":
        return "age"
    if feature in FLOW_FEATURES:
        return "flow"
    if feature.startswith(("p0", "p_lag", "p_sum", "p_max", "api")):
        return "rain"
    if feature.startswith("swe"):
        return "snow"
    if feature.startswith("sm"):
        return "soil"
    if feature.startswith(("t0", "tmax", "t_mean", "rad", "dew", "pet_sum", "p_minus_pet")):
        return "energy"
    if feature.startswith("doy"):
        return "season"
    if feature == "log_q_high":
        return "scale"
    return "basin" if feature in static_cols or feature == "log_area" else "basin"


def explain(model, test: pd.DataFrame, train: pd.DataFrame, static: pd.DataFrame, thresholds: pd.DataFrame,
            horizon: int = 1, top: int = 3) -> pd.DataFrame:
    """Per forecast origin: top drivers of P(high flow) at `horizon`, with their contribution shares."""
    cols = list(test.columns)
    start = test.groupby(ID)[DATE].min()
    hist = train.loc[train[ID].isin(start.index), [c for c in cols if c in train.columns]]
    first = hist[ID].map(start)
    look = getattr(model, "history_days", 150)
    hist = hist[(hist[DATE] >= first - pd.Timedelta(days=look)) & (hist[DATE] < first)]
    both = pd.concat([hist, test[cols]], ignore_index=True)
    X, _ = model.feature_matrix(both, len(hist), static, thresholds)
    groups = np.array([group_of(f, set(static.columns)) for f in model.feature_cols])
    names = sorted(set(groups))
    contrib = model.clf[(horizon, "high")].booster_.predict(X, pred_contrib=True)[:, :-1]   # drop bias column
    G = np.column_stack([contrib[:, groups == g].sum(axis=1) for g in names])               # log-odds per group
    pos = np.clip(G, 0, None)
    share = pos / np.maximum(pos.sum(axis=1, keepdims=True), 1e-12)
    order = np.argsort(-G, axis=1)[:, :top]
    out = test[[ID, DATE]].reset_index(drop=True).copy()
    out["horizon"] = horizon
    for k in range(top):
        g = np.array(names)[order[:, k]]
        out[f"driver_{k + 1}"] = [GROUP_LABELS[x] for x in g]
        out[f"share_{k + 1}"] = np.round(share[np.arange(len(out)), order[:, k]], 3)
        out[f"logodds_{k + 1}"] = np.round(G[np.arange(len(out)), order[:, k]], 3)
    return out
