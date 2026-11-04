"""BASINSHIFT submission: section 8.3 output schema and section 8.4 validation rules (Lane D).

Models hand over predictions in LONG format, one row per (gauge_id, date, horizon):
    gauge_id (str), date (forecast origin, day t), horizon (1/3/7),
    predicted_discharge, lower, upper, p_high, p_extreme          (mm/day, probabilities 0-1)

write_jsonl() turns that into one JSON record per (catchment, forecast origin), as in 8.3:
    {"catchment_id": "...", "forecast_origin": "2009-01-15T00:00:00Z", "forecast_horizons_days": [1, 3, 7],
     "predicted_discharge": [...], "high_flow_probability": [...], "extreme_flow_probability": [...],
     "prediction_intervals": {"lower": [...], "upper": [...], "nominal_coverage": 0.8}, "units": "mm/day"}

validate() applies every 8.4 rule and returns the list of problems (empty = valid).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import DATE, HI, HORIZON, ID, LO, PE, PH, PRED, PRED_COLS

HORIZONS = (1, 3, 7)
NOMINAL_COVERAGE = 0.8


def validate(preds: pd.DataFrame, expected_keys: pd.DataFrame, horizons=HORIZONS) -> list[str]:
    """BASINSHIFT 8.4. expected_keys = test.parquet[[gauge_id, date]] (one row per forecast origin)."""
    errs: list[str] = []
    need = [ID, DATE, HORIZON] + PRED_COLS
    miss_cols = [c for c in need if c not in preds.columns]
    if miss_cols:
        return [f"missing columns: {miss_cols}"]
    if not pd.api.types.is_string_dtype(preds[ID]):
        errs.append("gauge_id must be a string column")
    known = set(expected_keys[ID])
    unknown = sorted(set(preds[ID]) - known)
    if unknown:
        errs.append(f"{len(unknown)} unknown catchment ids, e.g. {unknown[:3]}")
    bad_h = sorted(set(preds[HORIZON]) - set(horizons))
    if bad_h:
        errs.append(f"unexpected horizons {bad_h}")
    vals = preds[PRED_COLS].to_numpy(float)
    if not np.isfinite(vals).all():
        errs.append(f"{int((~np.isfinite(vals)).any(axis=1).sum()):,} rows contain NaN or infinity")
    for c in (PH, PE):
        bad = ((preds[c] < 0) | (preds[c] > 1)).sum()
        if bad:
            errs.append(f"{bad:,} rows have {c} outside [0, 1]")
    for c in (PRED, LO, HI):
        bad = (preds[c] < 0).sum()
        if bad:
            errs.append(f"{bad:,} rows have negative {c}")
    bad = (preds[LO] > preds[HI]).sum()
    if bad:
        errs.append(f"{bad:,} rows have lower > upper")
    bad = (preds[PE] > preds[PH] + 1e-9).sum()
    if bad:
        errs.append(f"{bad:,} rows have p_extreme > p_high (an extreme event is also a high-flow event)")
    dup = preds.duplicated([ID, DATE, HORIZON]).sum()
    if dup:
        errs.append(f"{dup:,} duplicate (gauge_id, date, horizon) rows")
    want = expected_keys[[ID, DATE]].drop_duplicates().assign(_k=1).merge(
        pd.DataFrame({HORIZON: list(horizons), "_k": 1}), on="_k").drop(columns="_k")
    got = preds[[ID, DATE, HORIZON]].drop_duplicates()
    m = want.merge(got, on=[ID, DATE, HORIZON], how="left", indicator=True)
    n_missing = int((m["_merge"] == "left_only").sum())
    if n_missing:
        errs.append(f"{n_missing:,} required (gauge_id, date, horizon) predictions are missing")
    extra = len(got) - (len(want) - n_missing)
    if extra > 0:
        errs.append(f"{extra:,} predictions for (gauge_id, date) keys that are not in the test set")
    return errs


def assert_valid(preds: pd.DataFrame, expected_keys: pd.DataFrame, horizons=HORIZONS) -> None:
    errs = validate(preds, expected_keys, horizons)
    if errs:
        raise ValueError("Submission invalid (BASINSHIFT 8.4):\n  - " + "\n  - ".join(errs))


def write_jsonl(preds: pd.DataFrame, path: str | Path, horizons=HORIZONS, decimals: int = 4) -> Path:
    """One JSON record per (catchment, forecast origin), gzip-compressed if path ends in .gz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = preds.sort_values([ID, DATE, HORIZON])
    k = len(horizons)
    if len(p) % k:
        raise ValueError("every forecast origin needs exactly one row per horizon; validate first")
    ids = p[ID].to_numpy()[::k]
    dates = pd.to_datetime(p[DATE]).dt.strftime("%Y-%m-%dT00:00:00Z").to_numpy()[::k]
    arr = {c: np.round(p[c].to_numpy(float), decimals).reshape(-1, k) for c in PRED_COLS}
    hz = list(horizons)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        for i in range(len(ids)):
            rec = {
                "catchment_id": str(ids[i]),
                "forecast_origin": dates[i],
                "forecast_horizons_days": hz,
                "predicted_discharge": arr[PRED][i].tolist(),
                "high_flow_probability": arr[PH][i].tolist(),
                "extreme_flow_probability": arr[PE][i].tolist(),
                "prediction_intervals": {"lower": arr[LO][i].tolist(), "upper": arr[HI][i].tolist(),
                                         "nominal_coverage": NOMINAL_COVERAGE},
                "units": "mm/day",
            }
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    return path


def read_jsonl(path: str | Path) -> pd.DataFrame:
    """Back to long format, so a written file can be re-validated exactly as submitted."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    cols = {c: [] for c in [ID, DATE, HORIZON] + PRED_COLS}
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n = len(r["forecast_horizons_days"])
            cols[ID] += [r["catchment_id"]] * n
            cols[DATE] += [r["forecast_origin"]] * n
            cols[HORIZON] += r["forecast_horizons_days"]
            cols[PRED] += r["predicted_discharge"]
            cols[PH] += r["high_flow_probability"]
            cols[PE] += r["extreme_flow_probability"]
            cols[LO] += r["prediction_intervals"]["lower"]
            cols[HI] += r["prediction_intervals"]["upper"]
    df = pd.DataFrame(cols)
    df[ID] = df[ID].astype(str)
    df[DATE] = pd.to_datetime(df[DATE], utc=True).dt.tz_localize(None)
    return df
