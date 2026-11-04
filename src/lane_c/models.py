"""BASINSHIFT Multi-Horizon Hydrological Modeling Module (Lane C).

Step 1: Calendar-date horizon target preparation and log1p/expm1 transformations
        for temporary Data v2 (discharge_cfs in cfs) and Caravan (streamflow in mm/day).
Step 2: Causal persistence baseline across forecast horizons h in (1, 3, 7).
Step 3: Day-of-year (DOY) climatology baseline with training-only global DOY fallback
        for unseen stations and safe leap-year / missing-DOY handling.
Step 4: Nine LightGBM quantile regression models (Q10, Q50, Q90 x horizons 1, 3, 7 days)
        trained in log1p space (float32, seed=42, n_jobs<=4) with non-crossing enforcement.
Step 5: Six LightGBM flood exceedance probability classifiers (q_high=Q95, q_extreme=Q99
        x horizons 1, 3, 7 days) with training-only thresholds, class-imbalance weighting,
        single-class fold safety, and unseen-basin threshold policy.
Step 6: Leakage-safe isotonic probability calibration using a temporally held-out inner
        training split (inner_train + purge gap + inner_cal), preserving [0, 1] bounds
        and P(Y >= Q99) <= P(Y >= Q95) monotonicity.
Step 7: Temporal and unseen-station (LOSO) validation across horizons (1, 3, 7 days),
        evaluating forecast skill (MAE, RMSE, pinball loss, skill vs persistence/climatology),
        interval coverage ([Q10, Q90]), Q95/Q99 event detection (precision, recall, F1),
        and probability calibration (Brier score, reliability MAE) with strict fold
        boundary and station-leakage enforcement.
Step 8: Static-feature ablation comparing `dynamic_only` vs `full_static_plus_dynamic`
        across identical temporal and unseen-station validation folds and horizons (1, 3, 7),
        preserving Lane B's feature implementation and ensuring static attributes for unseen
        stations never access target data.
Step 9: Reproducible, append-only experiment logging (`log_basinshift_experiment` and
        `run_and_log_experiment`) recording timestamps, model configurations, feature sets,
        validation strategies, random seeds, per-horizon metrics, static ablation deltas,
        and explicit failure states without ever overwriting prior experiment records or
        logging test/holdout or fabricated metrics.

Hard rules enforced:
  - Never divide Caravan streamflow by drainage area (already in mm/day).
  - Use log1p(target) and expm1(prediction) in float32.
  - Exact calendar-date shifts (t + h days per station), never row-index shifts.
  - Strict temporal cutoff masking (max_target_date) to prevent horizon leakage.
  - Persistence uses strictly causal origin flow at <= t (with train-fold fallback).
  - Climatology is computed strictly from training-fold observations (zero test/val leakage).
  - Quantile models enforce 0.0 <= Q10 <= Q50 <= Q90 across all horizons and rows.
  - Flood classifiers strictly use q_high (Q95) and q_extreme (Q99) derived from training data only.
  - Probability calibration never uses validation/test targets and never calibrates a model on
    its own training rows or refits the base model on all training data after calibration.
  - Validation never uses the legacy FLOODCAST composite score and never allows validation
    stations or future targets to leak into training, thresholds, climatology, or calibration.
  - All randomness uses random_state=42; n_jobs never exceeds 4.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

from src.lane_c.config import (
    CARAVAN_TARGET_COL,
    CARAVAN_TARGET_UNITS,
    DATA_V2_TARGET_UNITS,
    DATE_COL,
    DEFAULT_WEIGHT_BOOST,
    DEFAULT_WEIGHT_Q,
    EXPERIMENT_LOG_COLUMNS,
    EXPERIMENTS_PATH,
    FLOOD_THRESHOLDS,
    HORIZONS,
    MAX_N_JOBS,
    QUANTILES,
    RANDOM_STATE,
    STATIC_DERIVED_FEATURE_COLS,
    STATIC_FEATURE_COLS,
    STATIC_RAW_FEATURE_COLS,
    STATION_COL,
    TARGET_COL,
)
try:  # Lane C's own validation helpers use the earlier FLOODCAST features/folds (now in legacy/)
    from src.features import FEATURE_COLS, build_features
    from src.validate import forward_folds, loso_folds
except ImportError:  # the BASINSHIFT pipeline feeds this model through src/lane_c_adapter.py instead
    FEATURE_COLS, build_features, forward_folds, loso_folds = [], None, None, None


def _sanitize_n_jobs(n_jobs: Optional[int] = None) -> int:
    """Enforce resource limit: 1 <= n_jobs <= MAX_N_JOBS (4)."""
    if n_jobs is None or int(n_jobs) <= 0:
        return int(MAX_N_JOBS)
    return min(int(n_jobs), int(MAX_N_JOBS))


def resolve_target_metadata(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> Dict[str, str]:
    """Identify target column and physical units without modifying or scaling values.

    Distinguishes between:
      - Caravan: 'streamflow' in 'mm/day' (already area-normalized)
      - Temporary Data v2: 'discharge_cfs' in 'cfs' (ft^3/s)
    """
    if target_col is not None:
        if target_col not in df.columns:
            raise KeyError(f"Specified target_col '{target_col}' not found in DataFrame columns.")
        col = target_col
    elif CARAVAN_TARGET_COL in df.columns:
        col = CARAVAN_TARGET_COL
    elif TARGET_COL in df.columns:
        col = TARGET_COL
    else:
        raise KeyError(
            f"Neither '{CARAVAN_TARGET_COL}' nor '{TARGET_COL}' found in DataFrame columns: "
            f"{list(df.columns)}"
        )

    if col == CARAVAN_TARGET_COL:
        units = CARAVAN_TARGET_UNITS
    elif col == TARGET_COL:
        units = DATA_V2_TARGET_UNITS
    else:
        units = "unknown"

    return {
        "target_col": col,
        "target_units": units,
    }


def transform_target(y: np.ndarray | pd.Series) -> np.ndarray:
    """Apply log1p transformation in float32: z = log1p(max(0, y)), preserving NaNs."""
    y_arr = np.asarray(y, dtype=np.float32)
    out = np.full_like(y_arr, np.nan, dtype=np.float32)
    valid = np.isfinite(y_arr) & (y_arr >= np.float32(0.0))
    out[valid] = np.log1p(y_arr[valid]).astype(np.float32)
    return out


def inverse_transform_target(z: np.ndarray | pd.Series) -> np.ndarray:
    """Apply expm1 inverse transformation in float32: y = max(0, expm1(z)), preserving NaNs."""
    z_arr = np.asarray(z, dtype=np.float32)
    out = np.full_like(z_arr, np.nan, dtype=np.float32)
    valid = np.isfinite(z_arr)
    # Clip extreme upper exponents before expm1 to avoid float32 overflow
    z_clipped = np.clip(z_arr[valid], np.float32(0.0), np.float32(80.0))
    out[valid] = np.maximum(np.float32(0.0), np.expm1(z_clipped)).astype(np.float32)
    return out


def prepare_horizon_targets(
    df: pd.DataFrame,
    horizons: Tuple[int, ...] = HORIZONS,
    target_col: Optional[str] = None,
    station_col: str = STATION_COL,
    date_col: str = DATE_COL,
    max_target_date: Optional[pd.Timestamp | str] = None,
) -> pd.DataFrame:
    """Construct multi-horizon targets using exact calendar-date shifts (t + h days).

    For each forecast origin (station_id, date = t) and each horizon h in `horizons`:
      - Computes exact target calendar date: t_target = t + pd.Timedelta(days=h)
      - Looks up ground-truth target at (station_id, t_target)
      - Masks any target where t_target > max_target_date to NaN to prevent leakage
      - Computes z_h{h} = log1p(y_h{h}) in float32

    Inputs:
      df: DataFrame containing station_col, date_col, and target column.
      horizons: tuple of positive integer day horizons, default (1, 3, 7).
      target_col: optional explicit target column name ('discharge_cfs' or 'streamflow').
      station_col: station identifier column name.
      date_col: forecast origin date column name.
      max_target_date: optional upper cutoff timestamp for valid target dates.
    Outputs:
      DataFrame with identical index and row order to `df`, containing:
        - station_col, date_col
        - y_origin, z_origin (causal observed target at origin date t, float32)
        - For each h in horizons:
            target_date_h{h} (datetime64[ns]),
            y_h{h} (float32 raw target at t + h days),
            z_h{h} (float32 log1p target at t + h days),
            valid_h{h} (bool mask where y_h{h} is finite and valid)
    """
    if station_col not in df.columns:
        raise KeyError(f"Missing required station column '{station_col}'")
    if date_col not in df.columns:
        raise KeyError(f"Missing required date column '{date_col}'")

    meta = resolve_target_metadata(df, target_col=target_col)
    resolved_target_col = meta["target_col"]
    resolved_units = meta["target_units"]

    orig_index = df.index
    stations = df[station_col].astype(str).str.strip().values
    origin_dates = pd.to_datetime(df[date_col]).dt.normalize()

    raw_y = pd.to_numeric(df[resolved_target_col], errors="coerce").to_numpy(dtype=np.float32)
    # Negative discharge/streamflow is physically invalid -> treat as NaN
    raw_y = np.where(raw_y >= np.float32(0.0), raw_y, np.nan).astype(np.float32)

    cutoff_ts: Optional[pd.Timestamp] = None
    if max_target_date is not None:
        cutoff_ts = pd.to_datetime(max_target_date).normalize()

    # Build a deduplicated (station_id, date) -> target lookup table
    lookup_df = pd.DataFrame(
        {
            station_col: stations,
            date_col: origin_dates.values,
            "_target_val": raw_y,
        }
    )
    if lookup_df.duplicated(subset=[station_col, date_col]).any():
        raise ValueError(f"Duplicate ({station_col}, {date_col}) keys detected in prepare_horizon_targets input.")

    if cutoff_ts is not None:
        lookup_df.loc[lookup_df[date_col] > cutoff_ts, "_target_val"] = np.nan

    lookup_series = lookup_df.set_index([station_col, date_col])["_target_val"]

    # Origin target at date t
    y_origin = raw_y.copy()
    if cutoff_ts is not None:
        y_origin = np.where(origin_dates.values <= cutoff_ts, y_origin, np.nan).astype(np.float32)
    z_origin = transform_target(y_origin)

    out_data: Dict[str, Any] = {
        station_col: stations,
        date_col: origin_dates.values,
        "y_origin": y_origin,
        "z_origin": z_origin,
    }

    for h in horizons:
        if int(h) <= 0:
            raise ValueError(f"Horizon h must be a positive integer, got {h}")
        h_int = int(h)
        target_dates = (origin_dates + pd.Timedelta(days=h_int)).values

        target_keys = pd.MultiIndex.from_arrays([stations, target_dates], names=[station_col, date_col])
        y_h = lookup_series.reindex(target_keys).to_numpy(dtype=np.float32)

        if cutoff_ts is not None:
            y_h = np.where(target_dates <= cutoff_ts, y_h, np.nan).astype(np.float32)

        z_h = transform_target(y_h)
        valid_h = np.isfinite(y_h) & np.isfinite(z_h)

        out_data[f"target_date_h{h_int}"] = target_dates
        out_data[f"y_h{h_int}"] = y_h
        out_data[f"z_h{h_int}"] = z_h
        out_data[f"valid_h{h_int}"] = valid_h

    out_df = pd.DataFrame(out_data, index=orig_index)
    out_df.attrs["target_col"] = resolved_target_col
    out_df.attrs["target_units"] = resolved_units
    out_df.attrs["horizons"] = tuple(int(h) for h in horizons)
    return out_df


def fit_persistence(
    train_df: pd.DataFrame,
    target_col: Optional[str] = None,
    station_col: str = STATION_COL,
    date_col: str = DATE_COL,
) -> Dict[str, Any]:
    """Fit causal persistence baseline fallback statistics strictly on training rows.

    Computes per-station training medians, last known training-fold observations,
    and global training median in float32 to serve as leakage-free fallbacks when
    forecast-origin flow y_t is missing or when predicting on unseen basins without y_t.
    """
    if station_col not in train_df.columns:
        raise KeyError(f"Missing required station column '{station_col}'")

    meta = resolve_target_metadata(train_df, target_col=target_col)
    t_col = meta["target_col"]
    t_units = meta["target_units"]

    stations = train_df[station_col].astype(str).str.strip().values
    raw_y = pd.to_numeric(train_df[t_col], errors="coerce").to_numpy(dtype=np.float32)
    raw_y = np.where(np.isfinite(raw_y) & (raw_y >= np.float32(0.0)), raw_y, np.nan).astype(np.float32)

    valid_all = raw_y[np.isfinite(raw_y)]
    global_median = np.float32(np.median(valid_all)) if len(valid_all) > 0 else np.float32(0.0)

    station_medians: Dict[str, np.float32] = {}
    station_last_known: Dict[str, np.float32] = {}

    if date_col in train_df.columns:
        dates = pd.to_datetime(train_df[date_col]).dt.normalize().values
    else:
        dates = np.arange(len(train_df))

    for sid in np.unique(stations):
        mask = (stations == sid) & np.isfinite(raw_y)
        if np.any(mask):
            s_y = raw_y[mask]
            s_d = dates[mask]
            station_medians[str(sid)] = np.float32(np.median(s_y))
            last_idx = int(np.argmax(s_d))
            station_last_known[str(sid)] = np.float32(s_y[last_idx])

    return {
        "model_type": "persistence",
        "target_col": t_col,
        "target_units": t_units,
        "station_col": station_col,
        "date_col": date_col,
        "station_medians": station_medians,
        "station_last_known": station_last_known,
        "global_median": global_median,
    }


def predict_persistence(
    model: Dict[str, Any],
    df: pd.DataFrame,
    horizons: Tuple[int, ...] = HORIZONS,
    target_col: Optional[str] = None,
    station_col: Optional[str] = None,
    date_col: Optional[str] = None,
) -> pd.DataFrame:
    """Predict future flow across horizons using causal origin flow y_t: y_hat(t+h) = y(t).

    Fallback hierarchy when y(t) at forecast origin t is NaN or absent:
      1. Causal forward-fill from earlier observations (<= t) of the same station in `df`.
      2. Training-fold `station_medians[station_id]` for stations seen during training.
      3. Training-fold `global_median` for unseen stations (LOSO).

    Outputs:
      DataFrame with identical index and row order to `df`, containing for each h in `horizons`:
        - persist_h{h}: float32 point forecast in raw target units ('cfs' or 'mm/day')
        - persist_z_h{h}: float32 log1p-transformed forecast
    """
    s_col = station_col or model.get("station_col", STATION_COL)
    d_col = date_col or model.get("date_col", DATE_COL)
    t_col = target_col or model.get("target_col", TARGET_COL)
    t_units = model.get("target_units", DATA_V2_TARGET_UNITS)

    if s_col not in df.columns:
        raise KeyError(f"Missing required station column '{s_col}' in predict_persistence.")

    orig_index = df.index
    stations = df[s_col].astype(str).str.strip()

    # Extract causal origin flow y_t if available in df
    if t_col in df.columns:
        raw_origin = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=np.float32)
    elif "y_origin" in df.columns:
        raw_origin = pd.to_numeric(df["y_origin"], errors="coerce").to_numpy(dtype=np.float32)
    else:
        raw_origin = np.full(len(df), np.nan, dtype=np.float32)

    raw_origin = np.where(
        np.isfinite(raw_origin) & (raw_origin >= np.float32(0.0)),
        raw_origin,
        np.nan,
    ).astype(np.float32)

    # 1. Causal forward-fill within each station sorted chronologically (never looking past t)
    if np.isnan(raw_origin).any() and np.isfinite(raw_origin).any():
        work_df = pd.DataFrame(
            {
                "_sid": stations.values,
                "_y": raw_origin,
                "_pos": np.arange(len(df)),
            },
            index=orig_index,
        )
        if d_col in df.columns:
            work_df["_dt"] = pd.to_datetime(df[d_col]).dt.normalize().values
            work_df = work_df.sort_values(["_sid", "_dt", "_pos"], kind="mergesort")
        else:
            work_df = work_df.sort_values(["_sid", "_pos"], kind="mergesort")

        work_df["_y"] = work_df.groupby("_sid", sort=False)["_y"].ffill()
        work_df = work_df.sort_values("_pos", kind="mergesort")
        raw_origin = work_df["_y"].to_numpy(dtype=np.float32)

    # 2. Fallback to train-fold station median, then 3. global train median
    station_medians: Dict[str, np.float32] = model.get("station_medians", {})
    global_median = np.float32(model.get("global_median", 0.0))

    fallback_vals = stations.map(station_medians).to_numpy(dtype=np.float32)
    fallback_vals = np.where(np.isfinite(fallback_vals), fallback_vals, global_median).astype(np.float32)

    y_persist = np.where(np.isfinite(raw_origin), raw_origin, fallback_vals).astype(np.float32)
    y_persist = np.maximum(np.float32(0.0), np.nan_to_num(y_persist, nan=global_median)).astype(np.float32)
    z_persist = transform_target(y_persist)

    out_cols: Dict[str, Any] = {}
    for h in horizons:
        if int(h) <= 0:
            raise ValueError(f"Horizon h must be a positive integer, got {h}")
        h_int = int(h)
        out_cols[f"persist_h{h_int}"] = y_persist.copy()
        out_cols[f"persist_z_h{h_int}"] = z_persist.copy()

    out_df = pd.DataFrame(out_cols, index=orig_index)
    out_df.attrs["target_col"] = t_col
    out_df.attrs["target_units"] = t_units
    out_df.attrs["horizons"] = tuple(int(h) for h in horizons)
    return out_df


def _quantile_col_name(q: float) -> str:
    """Format quantile float (e.g. 0.10, 0.50, 0.90) as 'q10', 'q50', 'q90'."""
    return f"q{int(round(float(q) * 100.0)):02d}"


def _fill_doy_curve_circular(
    doy_df: pd.DataFrame,
    value_cols: List[str],
    fallback_summary: Dict[str, np.float32],
) -> pd.DataFrame:
    """Ensure a complete 1..366 DOY table with circular interpolation for missing DOYs and leap day 366.

    If DOY 366 (or any intermediate DOY in 1..365) is missing in the training fold:
      - Reindexes to full 1..366 calendar days.
      - Uses circular extension (wrapping DOY 1..365 around both ends) with linear interpolation
        across any gaps in 1..365, and populates DOY 366 from DOY 365 / circular neighbors.
      - Falls back to `fallback_summary` if all DOYs are NaN.
      - Enforces non-negativity and non-crossing quantile order across `value_cols` in float32.
    """
    full_index = pd.Index(np.arange(1, 367, dtype=np.int32), name="doy")
    reindexed = doy_df.reindex(full_index).copy()

    # Check if we have any valid observations in 1..366
    has_any_valid = reindexed[value_cols].notna().any().any()
    if not has_any_valid:
        for col in value_cols:
            reindexed[col] = np.float32(fallback_summary.get(col, 0.0))
        return reindexed.astype(np.float32)

    # First, if doy 366 is NaN and doy 365 (or 1) is available, seed doy 366 from circular neighbors
    # or handle all 1..366 via circular padding
    base_365 = reindexed.loc[1:365, value_cols].copy()
    if base_365.isna().any().any():
        # Tile 3 copies to perform circular linear interpolation across year boundaries
        tiled = pd.concat([base_365, base_365, base_365], ignore_index=True)
        tiled = tiled.interpolate(method="linear", limit_direction="both")
        filled_365 = tiled.iloc[365:730].to_numpy(dtype=np.float32)
        for i, col in enumerate(value_cols):
            reindexed.loc[1:365, col] = filled_365[:, i]

    # Populate leap day 366 if missing using average of DOY 365 and DOY 1 (or DOY 365)
    if reindexed.loc[366, value_cols].isna().any():
        for col in value_cols:
            v365 = reindexed.loc[365, col]
            v001 = reindexed.loc[1, col]
            if np.isfinite(v365) and np.isfinite(v001):
                reindexed.loc[366, col] = np.float32(0.5 * (float(v365) + float(v001)))
            elif np.isfinite(v365):
                reindexed.loc[366, col] = np.float32(v365)
            else:
                reindexed.loc[366, col] = np.float32(fallback_summary.get(col, 0.0))

    # Final NaN guard and float32 cast
    for col in value_cols:
        fb = np.float32(fallback_summary.get(col, 0.0))
        arr = pd.to_numeric(reindexed[col], errors="coerce").to_numpy(dtype=np.float32)
        arr = np.maximum(np.float32(0.0), np.nan_to_num(arr, nan=fb, posinf=fb, neginf=np.float32(0.0)))
        reindexed[col] = arr.astype(np.float32)

    # Enforce monotonic non-crossing order across quantile columns (excluding 'mean')
    q_cols = [c for c in value_cols if c.startswith("q")]
    if len(q_cols) > 1:
        q_mat = np.sort(reindexed[q_cols].to_numpy(dtype=np.float32), axis=1)
        for i, col in enumerate(q_cols):
            reindexed[col] = q_mat[:, i].astype(np.float32)

    return reindexed.astype(np.float32)


def fit_climatology(
    train_df: pd.DataFrame,
    quantiles: Tuple[float, ...] = QUANTILES,
    target_col: Optional[str] = None,
    station_col: str = STATION_COL,
    date_col: str = DATE_COL,
    max_target_date: Optional[pd.Timestamp | str] = None,
) -> Dict[str, Any]:
    """Fit day-of-year (DOY) climatology strictly from training-fold observations.

    Computes:
      1. Per-station (station_id, doy) empirical quantiles (q10, q50, q90) and mean
         for doy in 1..366, filling missing DOYs and leap day 366 circularly.
      2. Training-only global (doy) empirical quantiles and mean across all training
         observations to serve as the unseen-station (LOSO) fallback.
      3. Per-station and global overall summary fallbacks for missing dates.
    """
    if station_col not in train_df.columns:
        raise KeyError(f"Missing required station column '{station_col}'")
    if date_col not in train_df.columns:
        raise KeyError(f"Missing required date column '{date_col}'")

    meta = resolve_target_metadata(train_df, target_col=target_col)
    t_col = meta["target_col"]
    t_units = meta["target_units"]

    sorted_qs = tuple(sorted(float(q) for q in quantiles))
    q_names = [_quantile_col_name(q) for q in sorted_qs]
    value_cols = [*q_names, "mean"]

    stations = train_df[station_col].astype(str).str.strip().values
    dates = pd.to_datetime(train_df[date_col], errors="coerce").dt.normalize()
    raw_y = pd.to_numeric(train_df[t_col], errors="coerce").to_numpy(dtype=np.float32)

    valid_mask = dates.notna().to_numpy() & np.isfinite(raw_y) & (raw_y >= np.float32(0.0))
    if max_target_date is not None:
        cutoff_ts = pd.to_datetime(max_target_date).normalize()
        valid_mask = valid_mask & (dates.to_numpy() <= cutoff_ts)

    valid_stations = stations[valid_mask]
    valid_dates = pd.DatetimeIndex(dates[valid_mask])
    valid_y = raw_y[valid_mask]
    valid_doy = valid_dates.day_of_year.to_numpy(dtype=np.int32)

    # Global training summary fallback
    global_summary: Dict[str, np.float32] = {}
    if len(valid_y) > 0:
        for q, q_name in zip(sorted_qs, q_names):
            global_summary[q_name] = np.float32(np.quantile(valid_y, q))
        global_summary["mean"] = np.float32(np.mean(valid_y))
    else:
        for col in value_cols:
            global_summary[col] = np.float32(0.0)

    work = pd.DataFrame(
        {
            station_col: valid_stations,
            "doy": valid_doy,
            "y": valid_y,
        }
    )

    # 1. Global DOY table (1..366) across all training observations for unseen stations
    if not work.empty:
        grp_global = work.groupby("doy", sort=True)["y"]
        global_dict: Dict[str, pd.Series] = {}
        for q, q_name in zip(sorted_qs, q_names):
            global_dict[q_name] = grp_global.quantile(q).astype(np.float32)
        global_dict["mean"] = grp_global.mean().astype(np.float32)
        raw_global_doy = pd.DataFrame(global_dict)
    else:
        raw_global_doy = pd.DataFrame(columns=value_cols, dtype=np.float32)

    global_doy_table = _fill_doy_curve_circular(raw_global_doy, value_cols, global_summary)

    # 2. Per-station summary and per-station (station_id, doy) table (1..366 per station)
    station_summary: Dict[str, Dict[str, np.float32]] = {}
    station_frames: List[pd.DataFrame] = []

    if not work.empty:
        grp_st_doy = work.groupby([station_col, "doy"], sort=True)["y"]
        st_doy_dict: Dict[str, pd.Series] = {}
        for q, q_name in zip(sorted_qs, q_names):
            st_doy_dict[q_name] = grp_st_doy.quantile(q).astype(np.float32)
        st_doy_dict["mean"] = grp_st_doy.mean().astype(np.float32)
        all_st_doy = pd.DataFrame(st_doy_dict)

        for sid in np.unique(valid_stations):
            sid_str = str(sid)
            s_vals = valid_y[valid_stations == sid]
            s_sum: Dict[str, np.float32] = {}
            for q, q_name in zip(sorted_qs, q_names):
                s_sum[q_name] = np.float32(np.quantile(s_vals, q))
            s_sum["mean"] = np.float32(np.mean(s_vals))
            station_summary[sid_str] = s_sum

            s_doy_raw = all_st_doy.loc[sid_str] if sid_str in all_st_doy.index else pd.DataFrame(columns=value_cols)
            s_doy_filled = _fill_doy_curve_circular(s_doy_raw, value_cols, s_sum)
            s_doy_filled[station_col] = sid_str
            s_doy_filled = s_doy_filled.reset_index().set_index([station_col, "doy"])
            station_frames.append(s_doy_filled)

    if station_frames:
        station_doy_table = pd.concat(station_frames, axis=0).astype(np.float32)
    else:
        empty_idx = pd.MultiIndex.from_arrays([[], []], names=[station_col, "doy"])
        station_doy_table = pd.DataFrame(columns=value_cols, index=empty_idx, dtype=np.float32)

    return {
        "model_type": "climatology",
        "target_col": t_col,
        "target_units": t_units,
        "station_col": station_col,
        "date_col": date_col,
        "quantiles": sorted_qs,
        "quantile_cols": q_names,
        "value_cols": value_cols,
        "station_doy_table": station_doy_table,
        "global_doy_table": global_doy_table,
        "station_summary": station_summary,
        "global_summary": global_summary,
    }


def predict_climatology(
    clim_model: Dict[str, Any],
    df: pd.DataFrame,
    horizons: Tuple[int, ...] = HORIZONS,
    station_col: Optional[str] = None,
    date_col: Optional[str] = None,
) -> pd.DataFrame:
    """Predict multi-horizon flow quantiles and median using training-fold DOY climatology.

    For each forecast origin row (station_id, date = t) and horizon h in `horizons`:
      - Computes target calendar date t_target = t + pd.Timedelta(days=h) and its DOY in 1..366.
      - Known stations use `station_doy_table[(station_id, target_doy)]`.
      - Unseen stations fall back to `global_doy_table[target_doy]` (training-only global DOY).
      - Missing/NaT dates fall back to `station_summary[station_id]` or `global_summary`.
      - Never reads or touches any target column in `df`.

    Outputs:
      DataFrame with identical index and row order to `df`, containing for each h in `horizons`:
        - clim_h{h}: float32 median (q50) climatology forecast in raw target units
        - clim_z_h{h}: float32 log1p(clim_h{h})
        - clim_q10_h{h}, clim_q50_h{h}, clim_q90_h{h} (and any configured quantile in float32)
        - clim_mean_h{h}: float32 mean climatology forecast
    """
    s_col = station_col or clim_model.get("station_col", STATION_COL)
    d_col = date_col or clim_model.get("date_col", DATE_COL)
    t_col = clim_model.get("target_col", TARGET_COL)
    t_units = clim_model.get("target_units", DATA_V2_TARGET_UNITS)
    q_names: List[str] = list(clim_model.get("quantile_cols", ["q10", "q50", "q90"]))
    value_cols: List[str] = list(clim_model.get("value_cols", [*q_names, "mean"]))

    if s_col not in df.columns:
        raise KeyError(f"Missing required station column '{s_col}' in predict_climatology.")
    if d_col not in df.columns:
        raise KeyError(f"Missing required date column '{d_col}' in predict_climatology.")

    orig_index = df.index
    stations = df[s_col].astype(str).str.strip().values
    origin_dates = pd.to_datetime(df[d_col], errors="coerce").dt.normalize()

    station_doy_table: pd.DataFrame = clim_model["station_doy_table"]
    global_doy_table: pd.DataFrame = clim_model["global_doy_table"]
    station_summary: Dict[str, Dict[str, np.float32]] = clim_model.get("station_summary", {})
    global_summary: Dict[str, np.float32] = clim_model.get("global_summary", {})

    # Precompute per-row fallback for missing/NaT dates (station summary -> global summary)
    n_rows = len(df)
    fallback_matrix = np.zeros((n_rows, len(value_cols)), dtype=np.float32)
    for j, col in enumerate(value_cols):
        g_val = np.float32(global_summary.get(col, 0.0))
        col_map = {sid: s_dict.get(col, g_val) for sid, s_dict in station_summary.items()}
        mapped = pd.Series(stations).map(col_map).to_numpy(dtype=np.float32)
        fallback_matrix[:, j] = np.where(np.isfinite(mapped), mapped, g_val).astype(np.float32)

    primary_q_col = "q50" if "q50" in q_names else q_names[len(q_names) // 2]
    out_cols: Dict[str, Any] = {}

    for h in horizons:
        if int(h) <= 0:
            raise ValueError(f"Horizon h must be a positive integer, got {h}")
        h_int = int(h)

        target_dates = origin_dates + pd.Timedelta(days=h_int)
        valid_date_mask = target_dates.notna().to_numpy()
        # Use dummy DOY=1 for NaT rows during index lookup, then overwrite with fallback_matrix
        doy_arr = np.where(
            valid_date_mask,
            target_dates.dt.day_of_year.fillna(1).to_numpy(dtype=np.int32),
            np.int32(1),
        )

        # 1. Lookup known (station_id, doy)
        st_keys = pd.MultiIndex.from_arrays([stations, doy_arr], names=[s_col, "doy"])
        st_vals = station_doy_table.reindex(st_keys)[value_cols].to_numpy(dtype=np.float32)

        # 2. Lookup global (doy) for unseen stations
        gl_vals = global_doy_table.reindex(doy_arr)[value_cols].to_numpy(dtype=np.float32)

        combined = np.where(np.isfinite(st_vals), st_vals, gl_vals).astype(np.float32)
        # 3. Apply missing-date fallback where target_date is NaT or any residual NaN
        combined = np.where(valid_date_mask[:, None] & np.isfinite(combined), combined, fallback_matrix).astype(np.float32)
        combined = np.maximum(np.float32(0.0), np.nan_to_num(combined, nan=0.0)).astype(np.float32)

        # Enforce quantile non-crossing across quantile columns
        n_q = len(q_names)
        if n_q > 1:
            combined[:, :n_q] = np.sort(combined[:, :n_q], axis=1)

        col_idx_map = {col: idx for idx, col in enumerate(value_cols)}
        y_med = combined[:, col_idx_map[primary_q_col]].copy()
        out_cols[f"clim_h{h_int}"] = y_med
        out_cols[f"clim_z_h{h_int}"] = transform_target(y_med)

        for q_name in q_names:
            out_cols[f"clim_{q_name}_h{h_int}"] = combined[:, col_idx_map[q_name]].copy()
        if "mean" in col_idx_map:
            out_cols[f"clim_mean_h{h_int}"] = combined[:, col_idx_map["mean"]].copy()

    out_df = pd.DataFrame(out_cols, index=orig_index)
    out_df.attrs["target_col"] = t_col
    out_df.attrs["target_units"] = t_units
    out_df.attrs["horizons"] = tuple(int(h) for h in horizons)
    return out_df


def _prepare_feature_matrix(
    X: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    forbidden_cols: Optional[Tuple[str, ...]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Sanitize and cast feature matrix X to float32 while guarding against target/ID leakage."""
    forbidden = set(forbidden_cols or (STATION_COL, DATE_COL, TARGET_COL, CARAVAN_TARGET_COL))
    if feature_cols is None:
        cols = [c for c in X.columns if c not in forbidden]
    else:
        leaked = [c for c in feature_cols if c in forbidden]
        if leaked:
            raise ValueError(f"Forbidden target/identifier columns found in feature_cols: {leaked}")
        missing = [c for c in feature_cols if c not in X.columns]
        if missing:
            raise KeyError(f"Feature matrix X is missing required feature_cols: {missing}")
        cols = list(feature_cols)

    if not cols:
        raise ValueError("Feature matrix X contains zero valid feature columns.")

    X_sub = X[cols].copy()
    for c in cols:
        arr = pd.to_numeric(X_sub[c], errors="coerce").to_numpy(dtype=np.float32)
        arr = np.nan_to_num(arr, nan=np.float32(0.0), posinf=np.float32(0.0), neginf=np.float32(0.0))
        X_sub[c] = arr.astype(np.float32)

    return X_sub, cols


def fit_quantile_models(
    X_train: pd.DataFrame,
    train_df: pd.DataFrame,
    horizons: Tuple[int, ...] = HORIZONS,
    quantiles: Tuple[float, ...] = QUANTILES,
    params: Optional[Dict[str, Any]] = None,
    target_col: Optional[str] = None,
    station_col: str = STATION_COL,
    date_col: str = DATE_COL,
    max_target_date: Optional[pd.Timestamp | str] = None,
) -> Dict[str, Any]:
    """Fit 9 LightGBM quantile regression models (3 horizons x 3 quantiles) in log1p space.

    Requirements enforced:
      - Trains strictly on training-fold rows where target_date (t + h) <= max_target_date
        (defaults to max date in `train_df` so horizon targets never cross fold boundaries).
      - Fits on float32 log1p(y_{t+h}) targets (`z_h{h}`).
      - Uses LightGBM `objective='quantile'` with `random_state=42` and `n_jobs <= 4`.
      - Never uses validation or test targets, and supports unseen stations seamlessly via `X`.
    """
    if len(X_train) != len(train_df):
        raise ValueError(
            f"Row count mismatch between X_train ({len(X_train)}) and train_df ({len(train_df)})."
        )

    params = dict(params or {})
    meta = resolve_target_metadata(train_df, target_col=target_col)
    t_col = meta["target_col"]
    t_units = meta["target_units"]

    forbidden = (station_col, date_col, t_col, TARGET_COL, CARAVAN_TARGET_COL)
    if any(c in X_train.columns for c in (t_col, TARGET_COL, CARAVAN_TARGET_COL)):
        raise ValueError("Target column detected inside X_train! Target leakage into features is forbidden.")

    X_clean, feature_cols = _prepare_feature_matrix(X_train, forbidden_cols=forbidden)

    # Default max_target_date to max origin date in train_df to prevent horizon lookahead
    if max_target_date is None and date_col in train_df.columns:
        effective_cutoff = pd.to_datetime(train_df[date_col]).dt.normalize().max()
    else:
        effective_cutoff = max_target_date

    targets_df = prepare_horizon_targets(
        train_df,
        horizons=horizons,
        target_col=t_col,
        station_col=station_col,
        date_col=date_col,
        max_target_date=effective_cutoff,
    )

    sorted_qs = tuple(sorted(float(q) for q in quantiles))
    q_names = [_quantile_col_name(q) for q in sorted_qs]
    horizons_tuple = tuple(int(h) for h in horizons)

    n_jobs = _sanitize_n_jobs(params.get("n_jobs", MAX_N_JOBS))
    n_estimators = int(params.get("n_estimators", 200))
    learning_rate = float(params.get("learning_rate", 0.05))
    num_leaves = int(params.get("num_leaves", 31))
    min_child_samples = int(params.get("min_child_samples", 20))
    subsample = float(params.get("subsample", 0.8))
    colsample_bytree = float(params.get("colsample_bytree", 0.8))

    # Compute training-fold fallback quantiles in log1p and raw space
    valid_origin_z = targets_df["z_origin"].to_numpy(dtype=np.float32)
    valid_origin_z = valid_origin_z[np.isfinite(valid_origin_z)]
    fallback_z: Dict[str, np.float32] = {}
    fallback_y: Dict[str, np.float32] = {}
    for q, q_name in zip(sorted_qs, q_names):
        if len(valid_origin_z) > 0:
            zq = np.float32(np.quantile(valid_origin_z, q))
        else:
            zq = np.float32(0.0)
        fallback_z[q_name] = zq
        fallback_y[q_name] = np.float32(inverse_transform_target(np.array([zq], dtype=np.float32))[0])

    models: Dict[Tuple[int, str], Optional[lgb.LGBMRegressor]] = {}
    models_by_name: Dict[str, Optional[lgb.LGBMRegressor]] = {}
    train_counts: Dict[int, int] = {}

    for h in horizons_tuple:
        valid_mask = targets_df[f"valid_h{h}"].to_numpy(dtype=bool)
        z_target = targets_df.loc[valid_mask, f"z_h{h}"].to_numpy(dtype=np.float32)
        X_h = X_clean.loc[valid_mask]
        train_counts[h] = int(len(X_h))

        # Adapt min_child_samples for small training slices so trees can still split
        eff_min_child = max(2, min(min_child_samples, max(2, len(X_h) // 5)))

        for q, q_name in zip(sorted_qs, q_names):
            key = (h, q_name)
            str_key = f"h{h}_{q_name}"
            if len(X_h) < 4:
                models[key] = None
                models_by_name[str_key] = None
                continue

            reg = lgb.LGBMRegressor(
                objective="quantile",
                alpha=float(q),
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                min_child_samples=eff_min_child,
                subsample=subsample,
                subsample_freq=1,
                colsample_bytree=colsample_bytree,
                random_state=RANDOM_STATE,
                n_jobs=n_jobs,
                verbosity=-1,
                force_col_wise=True,
            )
            reg.fit(X_h, z_target)
            models[key] = reg
            models_by_name[str_key] = reg

    return {
        "model_type": "quantile_lgbm",
        "models": models,
        "models_by_name": models_by_name,
        "feature_cols": feature_cols,
        "horizons": horizons_tuple,
        "quantiles": sorted_qs,
        "quantile_cols": q_names,
        "target_col": t_col,
        "target_units": t_units,
        "fallback_z": fallback_z,
        "fallback_y": fallback_y,
        "train_counts": train_counts,
        "n_jobs": n_jobs,
        "random_state": RANDOM_STATE,
    }


def predict_quantiles(
    q_bundle: Dict[str, Any],
    X: pd.DataFrame,
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Generate multi-horizon Q10, Q50, Q90 predictions in float32 with non-crossing enforcement.

    Inputs:
      q_bundle: fitted quantile model bundle from `fit_quantile_models`.
      X: feature DataFrame containing `q_bundle['feature_cols']`.
      df: optional reference DataFrame to validate length and index alignment.
    Outputs:
      DataFrame indexed identically to `df` (or `X`), containing for each h in `horizons`:
        - q10_h{h}, q50_h{h}, q90_h{h}: float32 non-negative predictions in raw target units
          satisfying 0.0 <= q10_h{h} <= q50_h{h} <= q90_h{h}.
        - z_q10_h{h}, z_q50_h{h}, z_q90_h{h}: float32 log1p predictions satisfying
          0.0 <= z_q10_h{h} <= z_q50_h{h} <= z_q90_h{h}.
    """
    if df is not None and len(X) != len(df):
        raise ValueError(f"Row count mismatch between X ({len(X)}) and df ({len(df)}).")

    orig_index = df.index if df is not None else X.index
    feature_cols: List[str] = list(q_bundle["feature_cols"])
    horizons: Tuple[int, ...] = tuple(q_bundle.get("horizons", HORIZONS))
    q_names: List[str] = list(q_bundle.get("quantile_cols", ["q10", "q50", "q90"]))
    models: Dict[Tuple[int, str], Optional[lgb.LGBMRegressor]] = q_bundle["models"]
    fallback_z: Dict[str, np.float32] = q_bundle.get("fallback_z", {})
    t_col = q_bundle.get("target_col", TARGET_COL)
    t_units = q_bundle.get("target_units", DATA_V2_TARGET_UNITS)

    X_mat, _ = _prepare_feature_matrix(X, feature_cols=feature_cols)
    n_rows = len(X_mat)
    n_q = len(q_names)

    out_cols: Dict[str, Any] = {}

    for h in horizons:
        h_int = int(h)
        z_mat = np.zeros((n_rows, n_q), dtype=np.float32)

        for j, q_name in enumerate(q_names):
            reg = models.get((h_int, q_name))
            fb_z = np.float32(fallback_z.get(q_name, 0.0))
            if reg is None or n_rows == 0:
                z_pred = np.full(n_rows, fb_z, dtype=np.float32)
            else:
                raw_pred = np.asarray(reg.predict(X_mat), dtype=np.float32).ravel()
                z_pred = np.nan_to_num(raw_pred, nan=fb_z, posinf=fb_z, neginf=np.float32(0.0)).astype(np.float32)
                z_pred = np.maximum(np.float32(0.0), z_pred).astype(np.float32)
            z_mat[:, j] = z_pred

        # Enforce non-crossing monotonicity in log1p space across quantiles (Q10 <= Q50 <= Q90)
        if n_q > 1 and n_rows > 0:
            z_mat = np.sort(z_mat, axis=1).astype(np.float32)

        # Apply inverse transform (expm1) and enforce non-crossing in raw space
        y_mat = inverse_transform_target(z_mat)
        y_mat = np.maximum(np.float32(0.0), np.nan_to_num(y_mat, nan=0.0)).astype(np.float32)
        if n_q > 1 and n_rows > 0:
            y_mat = np.sort(y_mat, axis=1).astype(np.float32)

        for j, q_name in enumerate(q_names):
            out_cols[f"{q_name}_h{h_int}"] = y_mat[:, j].copy()
            out_cols[f"z_{q_name}_h{h_int}"] = z_mat[:, j].copy()

    out_df = pd.DataFrame(out_cols, index=orig_index)
    out_df.attrs["target_col"] = t_col
    out_df.attrs["target_units"] = t_units
    out_df.attrs["horizons"] = horizons
    out_df.attrs["quantiles"] = q_bundle.get("quantiles", QUANTILES)
    return out_df


def _validate_flood_thresholds(thresholds: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Validate and enforce BASINSHIFT flood threshold definition: q_high=0.95, q_extreme=0.99.

    Explicitly rejects Q90 (0.90) as the high-flow threshold.
    """
    thresh = dict(thresholds or FLOOD_THRESHOLDS)
    if set(thresh.keys()) != {"q_high", "q_extreme"}:
        raise ValueError(
            f"Flood thresholds must have exact keys {{'q_high', 'q_extreme'}}, got {set(thresh.keys())}"
        )
    q_high = float(thresh["q_high"])
    q_extreme = float(thresh["q_extreme"])

    if np.isclose(q_high, 0.90) or q_high < 0.94:
        raise ValueError(
            f"Invalid q_high={q_high}: BASINSHIFT requires q_high=0.95 (Q95); never use Q90 (0.90)."
        )
    if not np.isclose(q_high, 0.95, atol=1e-4):
        raise ValueError(f"Expected q_high=0.95 (Q95), got {q_high}.")
    if not np.isclose(q_extreme, 0.99, atol=1e-4):
        raise ValueError(f"Expected q_extreme=0.99 (Q99), got {q_extreme}.")
    if q_high >= q_extreme:
        raise ValueError(f"q_high ({q_high}) must be strictly less than q_extreme ({q_extreme}).")

    return {"q_high": q_high, "q_extreme": q_extreme}


def compute_flood_thresholds(
    train_df: pd.DataFrame,
    thresholds: Optional[Dict[str, float]] = None,
    target_col: Optional[str] = None,
    station_col: str = STATION_COL,
    date_col: str = DATE_COL,
    max_target_date: Optional[pd.Timestamp | str] = None,
) -> Dict[str, Any]:
    """Compute per-station Q95 (q_high) and Q99 (q_extreme) flood thresholds strictly on training data.

    Unseen-Basin Policy:
      - Never accesses validation or test targets.
      - Stores per-station thresholds for all stations in `train_df`.
      - Also fits a training-only log-area linear regression (`log1p(threshold) ~ log1p(drainage_area_sqmi)`)
        and pooled `global_thresholds` strictly on `train_df` so unseen basins can be assigned
        a leakage-free reference threshold (or flagged via `allow_unseen_fallback=False`).
    """
    if station_col not in train_df.columns:
        raise KeyError(f"Missing required station column '{station_col}'")

    validated_thresh = _validate_flood_thresholds(thresholds)
    q_high_level = validated_thresh["q_high"]
    q_extreme_level = validated_thresh["q_extreme"]

    meta = resolve_target_metadata(train_df, target_col=target_col)
    t_col = meta["target_col"]
    t_units = meta["target_units"]

    stations = train_df[station_col].astype(str).str.strip().values
    raw_y = pd.to_numeric(train_df[t_col], errors="coerce").to_numpy(dtype=np.float32)
    valid_mask = np.isfinite(raw_y) & (raw_y >= np.float32(0.0))

    if max_target_date is not None and date_col in train_df.columns:
        cutoff_ts = pd.to_datetime(max_target_date).normalize()
        dates = pd.to_datetime(train_df[date_col], errors="coerce").dt.normalize()
        valid_mask = valid_mask & dates.notna().to_numpy() & (dates.to_numpy() <= cutoff_ts)

    valid_stations = stations[valid_mask]
    valid_y = raw_y[valid_mask]

    if len(valid_y) > 0:
        g_high = np.float32(np.quantile(valid_y, q_high_level))
        g_extreme = np.float32(max(float(g_high), float(np.quantile(valid_y, q_extreme_level))))
    else:
        g_high = np.float32(0.0)
        g_extreme = np.float32(0.0)

    global_thresholds = {
        "q_high": g_high,
        "q_extreme": g_extreme,
    }

    station_thresholds: Dict[str, Dict[str, np.float32]] = {}
    station_log_areas: List[float] = []
    station_log_q_high: List[float] = []
    station_log_q_ext: List[float] = []

    has_area = "drainage_area_sqmi" in train_df.columns

    for sid in np.unique(valid_stations):
        sid_str = str(sid)
        s_mask = valid_stations == sid
        s_y = valid_y[s_mask]
        if len(s_y) == 0:
            continue
        sh = np.float32(np.quantile(s_y, q_high_level))
        se = np.float32(max(float(sh), float(np.quantile(s_y, q_extreme_level))))
        station_thresholds[sid_str] = {
            "q_high": sh,
            "q_extreme": se,
        }
        if has_area:
            st_rows = train_df.loc[train_df[station_col].astype(str).str.strip() == sid_str, "drainage_area_sqmi"]
            area_val = float(pd.to_numeric(st_rows, errors="coerce").dropna().iloc[0]) if len(st_rows.dropna()) > 0 else np.nan
            if np.isfinite(area_val) and area_val > 0.0:
                station_log_areas.append(float(np.log1p(area_val)))
                station_log_q_high.append(float(np.log1p(sh)))
                station_log_q_ext.append(float(np.log1p(se)))

    # Fit training-only log-area scaling prior for unseen basins (when target is in cfs and >= 2 stations)
    area_scaling_prior: Optional[Dict[str, Tuple[float, float]]] = None
    if t_units == DATA_V2_TARGET_UNITS and len(station_log_areas) >= 2:
        x_arr = np.asarray(station_log_areas, dtype=np.float64)
        if np.std(x_arr) > 1e-6:
            slope_h, intercept_h = np.polyfit(x_arr, np.asarray(station_log_q_high, dtype=np.float64), 1)
            slope_e, intercept_e = np.polyfit(x_arr, np.asarray(station_log_q_ext, dtype=np.float64), 1)
            area_scaling_prior = {
                "q_high": (float(intercept_h), float(slope_h)),
                "q_extreme": (float(intercept_e), float(slope_e)),
            }

    return {
        "thresholds": validated_thresh,
        "target_col": t_col,
        "target_units": t_units,
        "station_col": station_col,
        "station_thresholds": station_thresholds,
        "global_thresholds": global_thresholds,
        "area_scaling_prior": area_scaling_prior,
    }


def resolve_station_thresholds(
    thresh_info: Dict[str, Any],
    df: pd.DataFrame,
    station_col: Optional[str] = None,
    allow_unseen_fallback: bool = True,
) -> pd.DataFrame:
    """Resolve per-row (q_high, q_extreme) thresholds from training-only `thresh_info`.

    Never inspects any target column in `df`.
    If `allow_unseen_fallback=False` and `df` contains a station not in `thresh_info['station_thresholds']`,
    raises ValueError rather than silently fabricating or computing thresholds from held-out targets.
    """
    s_col = station_col or thresh_info.get("station_col", STATION_COL)
    if s_col not in df.columns:
        raise KeyError(f"Missing required station column '{s_col}' in resolve_station_thresholds.")

    stations = df[s_col].astype(str).str.strip().values
    station_thresholds: Dict[str, Dict[str, np.float32]] = thresh_info.get("station_thresholds", {})
    global_thresholds: Dict[str, np.float32] = thresh_info.get("global_thresholds", {})
    area_prior = thresh_info.get("area_scaling_prior")

    unseen_stations = sorted({str(s) for s in stations if str(s) not in station_thresholds})
    if unseen_stations and not allow_unseen_fallback:
        raise ValueError(
            f"Unseen stations {unseen_stations} have no training-fold thresholds and "
            f"allow_unseen_fallback=False. Refusing to compute thresholds from held-out targets."
        )

    n_rows = len(df)
    q_high_arr = np.zeros(n_rows, dtype=np.float32)
    q_ext_arr = np.zeros(n_rows, dtype=np.float32)
    is_seen_arr = np.zeros(n_rows, dtype=bool)

    has_area = "drainage_area_sqmi" in df.columns
    area_vals = (
        pd.to_numeric(df["drainage_area_sqmi"], errors="coerce").to_numpy(dtype=np.float32)
        if has_area
        else np.full(n_rows, np.nan, dtype=np.float32)
    )

    g_high = np.float32(global_thresholds.get("q_high", 0.0))
    g_ext = np.float32(max(float(g_high), float(global_thresholds.get("q_extreme", 0.0))))

    for i, sid in enumerate(stations):
        sid_str = str(sid)
        if sid_str in station_thresholds:
            st_dict = station_thresholds[sid_str]
            q_high_arr[i] = np.float32(st_dict["q_high"])
            q_ext_arr[i] = np.float32(max(float(st_dict["q_high"]), float(st_dict["q_extreme"])))
            is_seen_arr[i] = True
        else:
            is_seen_arr[i] = False
            if area_prior is not None and np.isfinite(area_vals[i]) and area_vals[i] > 0.0:
                log_a = float(np.log1p(area_vals[i]))
                int_h, slp_h = area_prior["q_high"]
                int_e, slp_e = area_prior["q_extreme"]
                est_h = float(np.expm1(np.clip(int_h + slp_h * log_a, 0.0, 40.0)))
                est_e = float(np.expm1(np.clip(int_e + slp_e * log_a, 0.0, 40.0)))
                q_high_arr[i] = np.float32(max(0.0, est_h))
                q_ext_arr[i] = np.float32(max(float(q_high_arr[i]), est_e))
            else:
                q_high_arr[i] = g_high
                q_ext_arr[i] = g_ext

    return pd.DataFrame(
        {
            "q_high": q_high_arr,
            "q_extreme": q_ext_arr,
            "is_seen_station": is_seen_arr,
        },
        index=df.index,
    )


def prepare_exceedance_labels(
    df: pd.DataFrame,
    thresh_info: Dict[str, Any],
    horizons: Tuple[int, ...] = HORIZONS,
    target_col: Optional[str] = None,
    station_col: str = STATION_COL,
    date_col: str = DATE_COL,
    max_target_date: Optional[pd.Timestamp | str] = None,
    allow_unseen_fallback: bool = True,
) -> pd.DataFrame:
    """Create binary exceedance labels for q_high (Q95) and q_extreme (Q99) across horizons.

    Uses strictly the training-derived thresholds in `thresh_info`.
    Outputs for each h in `horizons`:
      - y_q_high_h{h}: float32 (1.0 if y_{t+h} >= q_high else 0.0; NaN if target invalid)
      - y_q_extreme_h{h}: float32 (1.0 if y_{t+h} >= q_extreme else 0.0; NaN if target invalid)
      - valid_h{h}: bool mask
    """
    targets_df = prepare_horizon_targets(
        df,
        horizons=horizons,
        target_col=target_col,
        station_col=station_col,
        date_col=date_col,
        max_target_date=max_target_date,
    )
    row_thresh = resolve_station_thresholds(
        thresh_info,
        df,
        station_col=station_col,
        allow_unseen_fallback=allow_unseen_fallback,
    )

    q_high_vec = row_thresh["q_high"].to_numpy(dtype=np.float32)
    q_ext_vec = row_thresh["q_extreme"].to_numpy(dtype=np.float32)

    out_cols: Dict[str, Any] = {
        station_col: targets_df[station_col].values,
        date_col: targets_df[date_col].values,
        "q_high": q_high_vec,
        "q_extreme": q_ext_vec,
        "is_seen_station": row_thresh["is_seen_station"].to_numpy(dtype=bool),
    }

    for h in horizons:
        h_int = int(h)
        y_h = targets_df[f"y_h{h_int}"].to_numpy(dtype=np.float32)
        valid_h = targets_df[f"valid_h{h_int}"].to_numpy(dtype=bool)

        lbl_high = np.full(len(df), np.nan, dtype=np.float32)
        lbl_ext = np.full(len(df), np.nan, dtype=np.float32)

        lbl_high[valid_h] = (y_h[valid_h] >= q_high_vec[valid_h]).astype(np.float32)
        lbl_ext[valid_h] = (y_h[valid_h] >= q_ext_vec[valid_h]).astype(np.float32)
        # Guarantee nested event consistency: extreme event implies high-flow event
        lbl_ext[valid_h] = np.minimum(lbl_high[valid_h], lbl_ext[valid_h]).astype(np.float32)

        out_cols[f"y_q_high_h{h_int}"] = lbl_high
        out_cols[f"y_q_extreme_h{h_int}"] = lbl_ext
        out_cols[f"valid_h{h_int}"] = valid_h

    out_df = pd.DataFrame(out_cols, index=df.index)
    out_df.attrs["thresholds"] = thresh_info.get("thresholds", dict(FLOOD_THRESHOLDS))
    out_df.attrs["horizons"] = tuple(int(h) for h in horizons)
    return out_df


def fit_exceedance_models(
    X_train: pd.DataFrame,
    train_df: pd.DataFrame,
    horizons: Tuple[int, ...] = HORIZONS,
    thresholds: Optional[Dict[str, float]] = None,
    params: Optional[Dict[str, Any]] = None,
    target_col: Optional[str] = None,
    station_col: str = STATION_COL,
    date_col: str = DATE_COL,
    max_target_date: Optional[pd.Timestamp | str] = None,
) -> Dict[str, Any]:
    """Fit 6 LightGBM binary classifiers (horizons 1, 3, 7 x thresholds q_high=Q95, q_extreme=Q99).

    Requirements enforced:
      - Thresholds (`q_high` = 0.95, `q_extreme` = 0.99) are derived strictly from `train_df`
        (with `date <= max_target_date`). Q90 is rejected.
      - Uses LightGBM `objective='binary'`, `float32` features, `random_state=42`, `n_jobs <= 4`.
      - Handles class imbalance via bounded `scale_pos_weight = clip(n_neg / n_pos, 1.0, 100.0)`.
      - Safely handles single-class training slices (all 0s or all 1s) via constant probability fallback.
    """
    if len(X_train) != len(train_df):
        raise ValueError(
            f"Row count mismatch between X_train ({len(X_train)}) and train_df ({len(train_df)})."
        )

    validated_thresh = _validate_flood_thresholds(thresholds)
    params = dict(params or {})
    meta = resolve_target_metadata(train_df, target_col=target_col)
    t_col = meta["target_col"]
    t_units = meta["target_units"]

    forbidden = (station_col, date_col, t_col, TARGET_COL, CARAVAN_TARGET_COL)
    if any(c in X_train.columns for c in (t_col, TARGET_COL, CARAVAN_TARGET_COL)):
        raise ValueError("Target column detected inside X_train! Target leakage into features is forbidden.")

    X_clean, feature_cols = _prepare_feature_matrix(X_train, forbidden_cols=forbidden)

    # Default max_target_date to max origin date in train_df so horizon targets never cross fold end
    if max_target_date is None and date_col in train_df.columns:
        effective_cutoff = pd.to_datetime(train_df[date_col]).dt.normalize().max()
    else:
        effective_cutoff = max_target_date

    thresh_info = compute_flood_thresholds(
        train_df,
        thresholds=validated_thresh,
        target_col=t_col,
        station_col=station_col,
        date_col=date_col,
        max_target_date=effective_cutoff,
    )

    labels_df = prepare_exceedance_labels(
        train_df,
        thresh_info=thresh_info,
        horizons=horizons,
        target_col=t_col,
        station_col=station_col,
        date_col=date_col,
        max_target_date=effective_cutoff,
        allow_unseen_fallback=True,
    )

    horizons_tuple = tuple(int(h) for h in horizons)
    level_names = ("q_high", "q_extreme")

    n_jobs = _sanitize_n_jobs(params.get("n_jobs", MAX_N_JOBS))
    n_estimators = int(params.get("n_estimators", 150))
    learning_rate = float(params.get("learning_rate", 0.05))
    num_leaves = int(params.get("num_leaves", 31))
    min_child_samples = int(params.get("min_child_samples", 20))
    subsample = float(params.get("subsample", 0.8))
    colsample_bytree = float(params.get("colsample_bytree", 0.8))

    models: Dict[Tuple[int, str], Optional[lgb.LGBMClassifier]] = {}
    models_by_name: Dict[str, Optional[lgb.LGBMClassifier]] = {}
    constant_probs: Dict[Tuple[int, str], np.float32] = {}
    scale_pos_weights: Dict[Tuple[int, str], float] = {}
    pos_counts: Dict[Tuple[int, str], int] = {}
    train_counts: Dict[int, int] = {}

    for h in horizons_tuple:
        valid_mask = labels_df[f"valid_h{h}"].to_numpy(dtype=bool)
        X_h = X_clean.loc[valid_mask]
        train_counts[h] = int(len(X_h))
        eff_min_child = max(2, min(min_child_samples, max(2, len(X_h) // 5)))

        for level_name in level_names:
            key = (h, level_name)
            str_key = f"h{h}_{level_name}"
            y_bin = labels_df.loc[valid_mask, f"y_{level_name}_h{h}"].to_numpy(dtype=np.int32)

            n_total = len(y_bin)
            n_pos = int(np.sum(y_bin == 1))
            n_neg = int(n_total - n_pos)
            pos_counts[key] = n_pos

            base_rate = np.float32(n_pos / n_total) if n_total > 0 else np.float32(0.0)
            constant_probs[key] = np.clip(base_rate, np.float32(0.0), np.float32(1.0))

            # Single-class or insufficient rows guard
            if n_total < 4 or n_pos == 0 or n_neg == 0:
                models[key] = None
                models_by_name[str_key] = None
                scale_pos_weights[key] = 1.0
                continue

            spw = float(params.get("scale_pos_weight", np.clip(n_neg / max(1, n_pos), 1.0, 100.0)))
            scale_pos_weights[key] = spw

            clf = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                min_child_samples=eff_min_child,
                subsample=subsample,
                subsample_freq=1,
                colsample_bytree=colsample_bytree,
                scale_pos_weight=spw,
                random_state=RANDOM_STATE,
                n_jobs=n_jobs,
                verbosity=-1,
                force_col_wise=True,
            )
            clf.fit(X_h, y_bin)
            models[key] = clf
            models_by_name[str_key] = clf

    return {
        "model_type": "exceedance_lgbm",
        "models": models,
        "models_by_name": models_by_name,
        "calibrators": {
            (h, lvl): None for h in horizons_tuple for lvl in level_names
        },
        "calibration_diagnostics": {
            (h, lvl): {"status": "uncalibrated"} for h in horizons_tuple for lvl in level_names
        },
        "feature_cols": feature_cols,
        "horizons": horizons_tuple,
        "thresholds": validated_thresh,
        "level_names": level_names,
        "thresh_info": thresh_info,
        "constant_probs": constant_probs,
        "scale_pos_weights": scale_pos_weights,
        "pos_counts": pos_counts,
        "train_counts": train_counts,
        "target_col": t_col,
        "target_units": t_units,
        "n_jobs": n_jobs,
        "random_state": RANDOM_STATE,
    }


def calibrate_exceedance_models(
    X_train: pd.DataFrame,
    train_df: pd.DataFrame,
    horizons: Tuple[int, ...] = HORIZONS,
    thresholds: Optional[Dict[str, float]] = None,
    cal_fraction: float = 0.20,
    gap_days: int = 14,
    min_cal_positives: int = 2,
    params: Optional[Dict[str, Any]] = None,
    target_col: Optional[str] = None,
    station_col: str = STATION_COL,
    date_col: str = DATE_COL,
    max_target_date: Optional[pd.Timestamp | str] = None,
) -> Dict[str, Any]:
    """Fit and calibrate 6 flood-exceedance classifiers without evaluation or in-sample leakage.

    Exact Calibration Strategy:
      1. Temporal Inner Split within `train_df`:
         - Sorts unique training dates (<= effective_cutoff) chronologically.
         - Allocates the trailing `cal_fraction` (default 20%) of dates to `inner_cal`
           (dates >= cal_start_date).
         - Enforces a strict purge buffer `eff_gap_days = max(gap_days, max(horizons))`
           between `inner_train` and `inner_cal`, so `inner_train_cutoff = cal_start_date - eff_gap_days`.
         - `inner_train` uses only rows with origin date <= `inner_train_cutoff` and masks any
           horizon target date t + h > `inner_train_cutoff`.
      2. Base Model & Threshold Fitting on `inner_train` Only:
         - Fits the 6 LightGBM classifiers (`q_high` = Q95, `q_extreme` = Q99 across h in 1, 3, 7)
           and flood thresholds strictly on `inner_train`.
      3. Out-of-Sample Calibration on `inner_cal`:
         - Generates raw probability predictions on `inner_cal` using the `inner_train`-fitted
           classifiers (never predicting on `inner_train` rows).
         - Evaluates binary exceedance targets on `inner_cal` (with target date t + h <= effective_cutoff)
           against `inner_train`'s thresholds.
         - Fits a separate `IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')`
           calibrator for each `(h, level_name)` on `(raw_p_cal, y_cal_bin)`.
      4. Paired Retention (No Blind Refit):
         - Retains the exact `inner_train` base classifiers paired with their `inner_cal` isotonic
           calibrators so the base probability distribution fed into isotonic regression at
           inference time is identical to the distribution on which the calibrator was fitted.
      5. Rare-Event & Single-Class Safety:
         - If `train_df` has insufficient date span for an inner split, or if `inner_cal` for
           a given `(h, level_name)` has only 1 unique class, fewer than `min_cal_positives`
           positive events, or zero prediction variance, isotonic calibration is skipped for that
           `(h, level_name)` (`calibrator = None`) and raw bounded probabilities are preserved.
    """
    if len(X_train) != len(train_df):
        raise ValueError(
            f"Row count mismatch between X_train ({len(X_train)}) and train_df ({len(train_df)})."
        )
    if not (0.05 <= float(cal_fraction) <= 0.50):
        raise ValueError(f"cal_fraction must be in [0.05, 0.50], got {cal_fraction}.")

    validated_thresh = _validate_flood_thresholds(thresholds)
    horizons_tuple = tuple(int(h) for h in horizons)
    level_names = ("q_high", "q_extreme")
    eff_gap_days = max(int(gap_days), max(horizons_tuple))

    meta = resolve_target_metadata(train_df, target_col=target_col)
    t_col = meta["target_col"]

    forbidden = (station_col, date_col, t_col, TARGET_COL, CARAVAN_TARGET_COL)
    if any(c in X_train.columns for c in (t_col, TARGET_COL, CARAVAN_TARGET_COL)):
        raise ValueError("Target column detected inside X_train! Target leakage into features is forbidden.")

    X_clean, feature_cols = _prepare_feature_matrix(X_train, forbidden_cols=forbidden)

    origin_dates = pd.to_datetime(train_df[date_col], errors="coerce").dt.normalize()
    if max_target_date is None:
        effective_cutoff = origin_dates.max()
    else:
        effective_cutoff = pd.to_datetime(max_target_date).normalize()

    valid_dates = np.sort(origin_dates[origin_dates <= effective_cutoff].dropna().unique())
    min_required_days = eff_gap_days + max(horizons_tuple) + 10

    # Fallback if total date span in train_df is too short to partition into inner_train + gap + inner_cal
    if len(valid_dates) < min_required_days:
        base_bundle = fit_exceedance_models(
            X_clean,
            train_df,
            horizons=horizons_tuple,
            thresholds=validated_thresh,
            params=params,
            target_col=t_col,
            station_col=station_col,
            date_col=date_col,
            max_target_date=effective_cutoff,
        )
        for h in horizons_tuple:
            for lvl in level_names:
                base_bundle["calibration_diagnostics"][(h, lvl)] = {
                    "status": "fallback_insufficient_dates",
                    "n_cal_samples": 0,
                    "n_cal_positives": 0,
                    "brier_raw": np.nan,
                    "brier_calibrated": np.nan,
                }
        base_bundle["model_type"] = "calibrated_exceedance_lgbm"
        return base_bundle

    split_idx = int(np.floor(len(valid_dates) * (1.0 - float(cal_fraction))))
    split_idx = max(1, min(split_idx, len(valid_dates) - max(horizons_tuple) - 2))
    cal_start_date = pd.Timestamp(valid_dates[split_idx]).normalize()
    inner_train_cutoff = cal_start_date - pd.Timedelta(days=eff_gap_days)

    inner_train_mask = (origin_dates <= inner_train_cutoff).to_numpy(dtype=bool)
    inner_cal_mask = ((origin_dates >= cal_start_date) & (origin_dates <= effective_cutoff)).to_numpy(dtype=bool)

    if int(np.sum(inner_train_mask)) < 10 or int(np.sum(inner_cal_mask)) < 5:
        base_bundle = fit_exceedance_models(
            X_clean,
            train_df,
            horizons=horizons_tuple,
            thresholds=validated_thresh,
            params=params,
            target_col=t_col,
            station_col=station_col,
            date_col=date_col,
            max_target_date=effective_cutoff,
        )
        for h in horizons_tuple:
            for lvl in level_names:
                base_bundle["calibration_diagnostics"][(h, lvl)] = {
                    "status": "fallback_insufficient_dates",
                    "n_cal_samples": 0,
                    "n_cal_positives": 0,
                    "brier_raw": np.nan,
                    "brier_calibrated": np.nan,
                }
        base_bundle["model_type"] = "calibrated_exceedance_lgbm"
        return base_bundle

    inner_train_df = train_df.loc[inner_train_mask]
    X_inner_train = X_clean.loc[inner_train_mask]

    # 1. Fit base classifiers and flood thresholds strictly on inner_train
    exc_bundle = fit_exceedance_models(
        X_inner_train,
        inner_train_df,
        horizons=horizons_tuple,
        thresholds=validated_thresh,
        params=params,
        target_col=t_col,
        station_col=station_col,
        date_col=date_col,
        max_target_date=inner_train_cutoff,
    )

    # 2. Prepare calibration labels across train_df (up to effective_cutoff) using inner_train's thresh_info,
    #    then slice strictly to inner_cal rows so target lookups at t + h within inner_cal remain intact.
    full_train_labels = prepare_exceedance_labels(
        train_df,
        thresh_info=exc_bundle["thresh_info"],
        horizons=horizons_tuple,
        target_col=t_col,
        station_col=station_col,
        date_col=date_col,
        max_target_date=effective_cutoff,
        allow_unseen_fallback=True,
    )
    cal_labels_df = full_train_labels.loc[inner_cal_mask]
    X_inner_cal = X_clean.loc[inner_cal_mask]
    inner_cal_df = train_df.loc[inner_cal_mask]

    # 3. Predict raw out-of-sample probabilities on inner_cal using inner_train base models
    raw_cal_preds_df = predict_exceedance_probs(exc_bundle, X_inner_cal, inner_cal_df, calibrated=False)

    calibrators: Dict[Tuple[int, str], Optional[IsotonicRegression]] = {}
    calibrators_by_name: Dict[str, Optional[IsotonicRegression]] = {}
    diagnostics: Dict[Tuple[int, str], Dict[str, Any]] = {}

    for h in horizons_tuple:
        valid_cal_h = cal_labels_df[f"valid_h{h}"].to_numpy(dtype=bool)
        for lvl in level_names:
            key = (h, lvl)
            str_key = f"h{h}_{lvl}"

            if exc_bundle["models"].get(key) is None or int(np.sum(valid_cal_h)) < 4:
                calibrators[key] = None
                calibrators_by_name[str_key] = None
                diagnostics[key] = {
                    "status": "fallback_uncalibrated_base_or_small_cal",
                    "n_cal_samples": int(np.sum(valid_cal_h)),
                    "n_cal_positives": 0,
                    "brier_raw": np.nan,
                    "brier_calibrated": np.nan,
                }
                continue

            raw_p = raw_cal_preds_df.loc[valid_cal_h, f"raw_prob_{lvl}_h{h}"].to_numpy(dtype=np.float32)
            y_cal = cal_labels_df.loc[valid_cal_h, f"y_{lvl}_h{h}"].to_numpy(dtype=np.float32)

            n_samples = int(len(y_cal))
            n_pos = int(np.sum(y_cal == 1.0))
            n_neg = int(n_samples - n_pos)
            brier_raw = float(np.mean((raw_p - y_cal) ** 2)) if n_samples > 0 else np.nan

            # Single-class or rare-event guard on inner_cal
            if n_pos == 0 or n_neg == 0:
                calibrators[key] = None
                calibrators_by_name[str_key] = None
                diagnostics[key] = {
                    "status": "fallback_single_class_cal",
                    "n_cal_samples": n_samples,
                    "n_cal_positives": n_pos,
                    "brier_raw": brier_raw,
                    "brier_calibrated": brier_raw,
                }
                continue

            if n_pos < int(min_cal_positives):
                calibrators[key] = None
                calibrators_by_name[str_key] = None
                diagnostics[key] = {
                    "status": "fallback_rare_positives_cal",
                    "n_cal_samples": n_samples,
                    "n_cal_positives": n_pos,
                    "brier_raw": brier_raw,
                    "brier_calibrated": brier_raw,
                }
                continue

            if float(np.max(raw_p) - np.min(raw_p)) < 1e-7:
                calibrators[key] = None
                calibrators_by_name[str_key] = None
                diagnostics[key] = {
                    "status": "fallback_constant_preds",
                    "n_cal_samples": n_samples,
                    "n_cal_positives": n_pos,
                    "brier_raw": brier_raw,
                    "brier_calibrated": brier_raw,
                }
                continue

            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(raw_p.astype(np.float64), y_cal.astype(np.float64))
            cal_p = np.asarray(iso.predict(raw_p.astype(np.float64)), dtype=np.float32)
            cal_p = np.clip(cal_p, np.float32(0.0), np.float32(1.0))
            brier_cal = float(np.mean((cal_p - y_cal) ** 2))

            calibrators[key] = iso
            calibrators_by_name[str_key] = iso
            diagnostics[key] = {
                "status": "calibrated",
                "n_cal_samples": n_samples,
                "n_cal_positives": n_pos,
                "brier_raw": brier_raw,
                "brier_calibrated": brier_cal,
            }

    exc_bundle["model_type"] = "calibrated_exceedance_lgbm"
    exc_bundle["calibrators"] = calibrators
    exc_bundle["calibrators_by_name"] = calibrators_by_name
    exc_bundle["calibration_diagnostics"] = diagnostics
    exc_bundle["calibration_split_info"] = {
        "inner_train_cutoff": inner_train_cutoff,
        "cal_start_date": cal_start_date,
        "effective_cutoff": effective_cutoff,
        "gap_days": eff_gap_days,
        "cal_fraction": float(cal_fraction),
        "n_inner_train_rows": int(np.sum(inner_train_mask)),
        "n_inner_cal_rows": int(np.sum(inner_cal_mask)),
    }
    return exc_bundle


def predict_exceedance_probs(
    exc_bundle: Dict[str, Any],
    X: pd.DataFrame,
    df: Optional[pd.DataFrame] = None,
    calibrated: bool = True,
) -> pd.DataFrame:
    """Predict high-flow (Q95) and extreme-flow (Q99) exceedance probabilities in float32.

    Enforces:
      - Applies isotonic calibration per (horizon, threshold) when `calibrated=True`
        and a fitted calibrator is present in `exc_bundle['calibrators']`.
      - All probabilities (both raw and calibrated) are finite float32 values in [0.0, 1.0].
      - Monotonicity across severity thresholds:
          prob_q_extreme_h{h} <= prob_q_high_h{h}
          raw_prob_q_extreme_h{h} <= raw_prob_q_high_h{h}
      - Never reads or depends on target values in `df`.
    """
    if df is not None and len(X) != len(df):
        raise ValueError(f"Row count mismatch between X ({len(X)}) and df ({len(df)}).")

    orig_index = df.index if df is not None else X.index
    feature_cols: List[str] = list(exc_bundle["feature_cols"])
    horizons: Tuple[int, ...] = tuple(exc_bundle.get("horizons", HORIZONS))
    models: Dict[Tuple[int, str], Optional[lgb.LGBMClassifier]] = exc_bundle["models"]
    calibrators: Dict[Tuple[int, str], Optional[IsotonicRegression]] = exc_bundle.get("calibrators", {})
    constant_probs: Dict[Tuple[int, str], np.float32] = exc_bundle.get("constant_probs", {})

    X_mat, _ = _prepare_feature_matrix(X, feature_cols=feature_cols)
    n_rows = len(X_mat)

    out_cols: Dict[str, Any] = {}

    for h in horizons:
        h_int = int(h)
        raw_preds: Dict[str, np.ndarray] = {}
        cal_preds: Dict[str, np.ndarray] = {}

        for level_name in ("q_high", "q_extreme"):
            key = (h_int, level_name)
            clf = models.get(key)
            iso = calibrators.get(key) if calibrated else None
            fb_prob = np.float32(constant_probs.get(key, 0.0))

            if clf is None or n_rows == 0:
                raw_p = np.full(n_rows, fb_prob, dtype=np.float32)
            else:
                proba = clf.predict_proba(X_mat)
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    p_col = np.asarray(proba[:, 1], dtype=np.float32).ravel()
                else:
                    p_col = np.full(n_rows, fb_prob, dtype=np.float32)
                raw_p = np.nan_to_num(p_col, nan=fb_prob, posinf=np.float32(1.0), neginf=np.float32(0.0))
                raw_p = np.clip(raw_p, np.float32(0.0), np.float32(1.0)).astype(np.float32)

            raw_preds[level_name] = raw_p

            if iso is not None and n_rows > 0:
                iso_out = np.asarray(iso.predict(raw_p.astype(np.float64)), dtype=np.float32).ravel()
                cal_p = np.nan_to_num(iso_out, nan=fb_prob, posinf=np.float32(1.0), neginf=np.float32(0.0))
                cal_p = np.clip(cal_p, np.float32(0.0), np.float32(1.0)).astype(np.float32)
            else:
                cal_p = raw_p.copy()

            cal_preds[level_name] = cal_p

        # Enforce nested event monotonicity: P(Y >= Q99) <= P(Y >= Q95)
        raw_high = raw_preds["q_high"]
        raw_ext = np.minimum(raw_high, raw_preds["q_extreme"]).astype(np.float32)

        p_high = cal_preds["q_high"]
        p_ext = np.minimum(p_high, cal_preds["q_extreme"]).astype(np.float32)

        out_cols[f"prob_q_high_h{h_int}"] = p_high
        out_cols[f"prob_q_extreme_h{h_int}"] = p_ext
        out_cols[f"raw_prob_q_high_h{h_int}"] = raw_high
        out_cols[f"raw_prob_q_extreme_h{h_int}"] = raw_ext

    out_df = pd.DataFrame(out_cols, index=orig_index)
    out_df.attrs["thresholds"] = exc_bundle.get("thresholds", dict(FLOOD_THRESHOLDS))
    out_df.attrs["horizons"] = horizons
    out_df.attrs["target_col"] = exc_bundle.get("target_col", TARGET_COL)
    out_df.attrs["target_units"] = exc_bundle.get("target_units", DATA_V2_TARGET_UNITS)
    return out_df


def make_weights(
    train_df: pd.DataFrame,
    q: float = DEFAULT_WEIGHT_Q,
    boost: float = DEFAULT_WEIGHT_BOOST,
) -> np.ndarray:
    """Compute per-station sample weights on training fold rows (float32)."""
    if len(train_df) == 0:
        return np.array([], dtype=np.float32)

    meta = resolve_target_metadata(train_df) if (
        CARAVAN_TARGET_COL in train_df.columns or TARGET_COL in train_df.columns
    ) else None
    if meta is None:
        return np.ones(len(train_df), dtype=np.float32)

    t_col = meta["target_col"]
    weights = np.ones(len(train_df), dtype=np.float32)
    y = pd.to_numeric(train_df[t_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    stations = train_df[STATION_COL].values

    for sid in np.unique(stations):
        mask = stations == sid
        station_y = y[mask]
        if len(station_y) > 0:
            threshold = np.float32(np.nanquantile(station_y, q))
            extreme_mask = mask & (y >= threshold)
            weights[extreme_mask] = np.float32(boost)

    return weights


def _compute_pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float,
) -> float:
    """Compute quantile (pinball) loss for quantile level alpha in (0, 1)."""
    if len(y_true) == 0:
        return float("nan")
    residual = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    loss = np.where(residual >= 0.0, float(alpha) * residual, (float(alpha) - 1.0) * residual)
    return float(np.mean(loss))


def _compute_calibration_mae(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute sample-weighted reliability (expected calibration error / MAE) across probability bins."""
    if len(y_true) == 0:
        return float("nan")
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.clip(np.asarray(y_prob, dtype=np.float64).ravel(), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bin_ids = np.digitize(y_p, bins[1:-1], right=True)

    total = len(y_t)
    ece = 0.0
    for b in range(int(n_bins)):
        mask = bin_ids == b
        cnt = int(np.sum(mask))
        if cnt > 0:
            ece += (cnt / total) * abs(float(np.mean(y_p[mask])) - float(np.mean(y_t[mask])))
    return float(ece)


def _verify_fold_integrity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    mode: str,
    horizons: Tuple[int, ...] = HORIZONS,
    effective_train_cutoff: Optional[pd.Timestamp] = None,
    station_col: str = STATION_COL,
    date_col: str = DATE_COL,
) -> Dict[str, Any]:
    """Verify strict temporal or spatial fold boundaries before model fitting.

    Raises ValueError if:
      - `train_df` or `val_df` is empty.
      - For `mode == 'temporal'`: any training origin date or horizon target date
        (t_train + max(horizons)) reaches or crosses `min(val_dates)`.
      - For `mode == 'unseen_station'`: any station in `val_df` appears in `train_df`.
    """
    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("Both train_df and val_df must be non-empty for fold validation.")

    train_dates = pd.to_datetime(train_df[date_col], errors="coerce").dt.normalize()
    val_dates = pd.to_datetime(val_df[date_col], errors="coerce").dt.normalize()
    max_h = max(int(h) for h in horizons)

    max_train_origin = train_dates.max()
    min_val_origin = val_dates.min()
    max_val_origin = val_dates.max()
    cutoff = pd.to_datetime(effective_train_cutoff).normalize() if effective_train_cutoff is not None else max_train_origin

    train_stations = set(train_df[station_col].astype(str).unique())
    val_stations = set(val_df[station_col].astype(str).unique())
    overlap_stations = sorted(train_stations.intersection(val_stations))

    if mode == "temporal":
        if max_train_origin > cutoff:
            raise ValueError(
                f"Temporal fold leakage: max train origin date ({max_train_origin}) > cutoff ({cutoff})."
            )
        if cutoff >= min_val_origin:
            raise ValueError(
                f"Temporal fold leakage: train target cutoff ({cutoff}) >= min val origin date ({min_val_origin})."
            )
        if max_train_origin + pd.Timedelta(days=max_h) >= min_val_origin:
            raise ValueError(
                f"Temporal horizon leakage: max train target date ({max_train_origin + pd.Timedelta(days=max_h)}) "
                f"reaches or exceeds min val origin date ({min_val_origin})."
            )
    elif mode == "unseen_station":
        if overlap_stations:
            raise ValueError(
                f"Station leakage detected in unseen-station fold! Overlapping stations: {overlap_stations}"
            )
    else:
        raise ValueError(f"Unknown validation mode '{mode}'. Expected 'temporal' or 'unseen_station'.")

    return {
        "mode": mode,
        "max_train_origin_date": max_train_origin,
        "effective_train_cutoff": cutoff,
        "min_val_origin_date": min_val_origin,
        "max_val_origin_date": max_val_origin,
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "n_train_stations": int(len(train_stations)),
        "n_val_stations": int(len(val_stations)),
        "val_stations": sorted(val_stations),
        "overlap_stations": overlap_stations,
    }


def _verify_no_unseen_station_in_bundle(
    model_bundle: Dict[str, Any],
    val_stations: List[str],
) -> None:
    """Verify that no held-out validation station leaked into any fitted sub-bundle."""
    val_set = set(str(s) for s in val_stations)

    if "persistence" in model_bundle and model_bundle["persistence"] is not None:
        pers = model_bundle["persistence"]
        pers_stations = set(str(s) for s in pers.get("station_medians", {}).keys()) | set(
            str(s) for s in pers.get("station_last_known", {}).keys()
        )
        leaked = val_set.intersection(pers_stations)
        if leaked:
            raise ValueError(f"Station leakage in persistence bundle: {sorted(leaked)}")

    if "climatology" in model_bundle and model_bundle["climatology"] is not None:
        clim = model_bundle["climatology"]
        clim_stations = set(str(s) for s in clim.get("station_summary", {}).keys())
        if "station_doy_table" in clim and isinstance(clim["station_doy_table"].index, pd.MultiIndex):
            clim_stations |= set(
                str(s) for s in clim["station_doy_table"].index.get_level_values(0).unique()
            )
        leaked = val_set.intersection(clim_stations)
        if leaked:
            raise ValueError(f"Station leakage in climatology bundle: {sorted(leaked)}")

    if "exceedance" in model_bundle and model_bundle["exceedance"] is not None:
        exc = model_bundle["exceedance"]
        thresh_obj = exc.get("thresh_info", {}).get("station_thresholds", {})
        if isinstance(thresh_obj, dict):
            thresh_stations = set(str(s) for s in thresh_obj.keys())
        else:
            thresh_stations = set(str(s) for s in thresh_obj.index)
        leaked = val_set.intersection(thresh_stations)
        if leaked:
            raise ValueError(f"Station leakage in flood thresholds: {sorted(leaked)}")


def fit_models(
    X_train: pd.DataFrame,
    y_train: Optional[pd.Series | np.ndarray | pd.DataFrame] = None,
    weights: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    *,
    train_df: Optional[pd.DataFrame] = None,
    horizons: Tuple[int, ...] = HORIZONS,
    quantiles: Tuple[float, ...] = QUANTILES,
    thresholds: Optional[Dict[str, float]] = None,
    calibrate: bool = True,
    cal_fraction: float = 0.20,
    cal_gap_days: int = 14,
    include_baselines: bool = True,
    target_col: Optional[str] = None,
    station_col: str = STATION_COL,
    date_col: str = DATE_COL,
    max_target_date: Optional[pd.Timestamp | str] = None,
) -> Dict[str, Any]:
    """Fit the unified BASINSHIFT multi-horizon model suite on training fold data only.

    Components fitted strictly on `train_df` (with horizon targets masked at `max_target_date`):
      1. Causal persistence baseline (`fit_persistence`)
      2. Day-of-year (DOY) climatology baseline (`fit_climatology`)
      3. Nine LightGBM quantile regressors (`fit_quantile_models`: Q10, Q50, Q90 x h=1, 3, 7)
      4. Six LightGBM flood exceedance classifiers (`calibrate_exceedance_models` or
         `fit_exceedance_models`: q_high=Q95, q_extreme=Q99 x h=1, 3, 7)
    """
    if train_df is None:
        if isinstance(y_train, pd.DataFrame):
            train_df = y_train
        elif isinstance(weights, pd.DataFrame):
            train_df = weights
            weights = None
        else:
            raise ValueError(
                "BASINSHIFT fit_models requires `train_df` (containing station_id, date, and target) "
                "either via `train_df=...` or as the second/third argument."
            )

    if len(X_train) != len(train_df):
        raise ValueError(
            f"Row count mismatch between X_train ({len(X_train)}) and train_df ({len(train_df)})."
        )

    meta = resolve_target_metadata(train_df, target_col=target_col)
    t_col = meta["target_col"]
    t_units = meta["target_units"]

    origin_dates = pd.to_datetime(train_df[date_col], errors="coerce").dt.normalize()
    if max_target_date is None:
        effective_cutoff = origin_dates.max()
    else:
        effective_cutoff = pd.to_datetime(max_target_date).normalize()

    # Mask train_df rows whose origin date exceeds effective_cutoff
    valid_origin_mask = (origin_dates <= effective_cutoff).to_numpy(dtype=bool)
    if not np.all(valid_origin_mask):
        train_df = train_df.loc[valid_origin_mask]
        X_train = X_train.loc[valid_origin_mask]
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float32)[valid_origin_mask]

    horizons_tuple = tuple(int(h) for h in horizons)
    quantiles_tuple = tuple(float(q) for q in quantiles)
    validated_thresh = _validate_flood_thresholds(thresholds)

    pers_bundle = None
    clim_bundle = None
    if include_baselines:
        pers_bundle = fit_persistence(
            train_df,
            target_col=t_col,
            station_col=station_col,
            date_col=date_col,
        )
        clim_bundle = fit_climatology(
            train_df,
            quantiles=quantiles_tuple,
            target_col=t_col,
            station_col=station_col,
            date_col=date_col,
            max_target_date=effective_cutoff,
        )

    q_bundle = fit_quantile_models(
        X_train,
        train_df,
        horizons=horizons_tuple,
        quantiles=quantiles_tuple,
        params=params,
        target_col=t_col,
        station_col=station_col,
        date_col=date_col,
        max_target_date=effective_cutoff,
    )

    if calibrate:
        exc_bundle = calibrate_exceedance_models(
            X_train,
            train_df,
            horizons=horizons_tuple,
            thresholds=validated_thresh,
            cal_fraction=cal_fraction,
            gap_days=cal_gap_days,
            params=params,
            target_col=t_col,
            station_col=station_col,
            date_col=date_col,
            max_target_date=effective_cutoff,
        )
    else:
        exc_bundle = fit_exceedance_models(
            X_train,
            train_df,
            horizons=horizons_tuple,
            thresholds=validated_thresh,
            params=params,
            target_col=t_col,
            station_col=station_col,
            date_col=date_col,
            max_target_date=effective_cutoff,
        )

    return {
        "model_type": "basinshift_multi_horizon",
        "persistence": pers_bundle,
        "climatology": clim_bundle,
        "quantile": q_bundle,
        "exceedance": exc_bundle,
        "feature_cols": q_bundle["feature_cols"],
        "horizons": horizons_tuple,
        "quantiles": quantiles_tuple,
        "thresholds": validated_thresh,
        "target_col": t_col,
        "target_units": t_units,
        "max_target_date": effective_cutoff,
        "n_jobs": q_bundle["n_jobs"],
        "random_state": RANDOM_STATE,
    }


def predict_multi_horizon(
    model_bundle: Dict[str, Any],
    X: pd.DataFrame,
    df: Optional[pd.DataFrame] = None,
    include_baselines: bool = True,
    calibrated: bool = True,
) -> pd.DataFrame:
    """Predict all multi-horizon quantiles, exceedance probabilities, and baselines in float32."""
    if df is not None and len(X) != len(df):
        raise ValueError(f"Row count mismatch between X ({len(X)}) and df ({len(df)}).")

    horizons_tuple = tuple(int(h) for h in model_bundle.get("horizons", HORIZONS))
    q_bundle = model_bundle["quantile"] if "quantile" in model_bundle else model_bundle
    q_preds = predict_quantiles(q_bundle, X, df=df)

    parts = [q_preds]

    if "exceedance" in model_bundle and model_bundle["exceedance"] is not None:
        exc_preds = predict_exceedance_probs(
            model_bundle["exceedance"],
            X,
            df=df,
            calibrated=calibrated,
        )
        parts.append(exc_preds)

    if include_baselines and df is not None:
        if model_bundle.get("persistence") is not None:
            pers_df = predict_persistence(
                model_bundle["persistence"],
                df,
                horizons=horizons_tuple,
            )
            for h in horizons_tuple:
                if f"persist_h{h}" in pers_df.columns:
                    pers_df[f"persistence_h{h}"] = pers_df[f"persist_h{h}"]
            parts.append(pers_df)

        if model_bundle.get("climatology") is not None:
            clim_df = predict_climatology(
                model_bundle["climatology"],
                df,
                horizons=horizons_tuple,
            )
            parts.append(clim_df)

    out_df = pd.concat(parts, axis=1)
    out_df.attrs["horizons"] = model_bundle.get("horizons", HORIZONS)
    out_df.attrs["quantiles"] = model_bundle.get("quantiles", QUANTILES)
    out_df.attrs["thresholds"] = model_bundle.get("thresholds", dict(FLOOD_THRESHOLDS))
    out_df.attrs["target_col"] = model_bundle.get("target_col", TARGET_COL)
    out_df.attrs["target_units"] = model_bundle.get("target_units", DATA_V2_TARGET_UNITS)
    return out_df


def predict(
    model_bundle: Dict[str, Any],
    X: pd.DataFrame,
    df: Optional[pd.DataFrame] = None,
    horizon: int = 1,
    quantile: float = 0.50,
) -> np.ndarray:
    """Return 1D non-negative float32 point predictions (default: Q50 at h=1)."""
    q_bundle = model_bundle["quantile"] if "quantile" in model_bundle else model_bundle
    q_df = predict_quantiles(q_bundle, X, df=df)
    col = f"{_quantile_col_name(quantile)}_h{int(horizon)}"
    if col not in q_df.columns:
        raise ValueError(f"Requested prediction column '{col}' not found in {list(q_df.columns)}.")
    preds = q_df[col].to_numpy(dtype=np.float32)
    return np.maximum(np.float32(0.0), np.nan_to_num(preds, nan=0.0)).astype(np.float32)


def compute_horizon_metrics(
    val_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    thresh_info: Dict[str, Any],
    horizons: Tuple[int, ...] = HORIZONS,
    quantiles: Tuple[float, ...] = QUANTILES,
    prob_threshold: float = 0.50,
    target_col: Optional[str] = None,
    station_col: str = STATION_COL,
    date_col: str = DATE_COL,
    max_val_target_date: Optional[pd.Timestamp | str] = None,
    allow_unseen_fallback: bool = True,
) -> pd.DataFrame:
    """Compute BASINSHIFT multi-horizon evaluation metrics without legacy composite scores.

    Evaluates for each horizon h in `horizons`:
      - Continuous forecast skill:
          `mae`, `rmse`, `persistence_mae`, `persistence_rmse`,
          `climatology_mae`, `climatology_rmse`,
          `skill_mae_vs_persistence`, `skill_mae_vs_climatology`,
          `pinball_q10`, `pinball_q50`, `pinball_q90`, `mean_pinball_loss`
      - Prediction interval quality ([Q10, Q90]):
          `interval_coverage_80`, `interval_width_80`
      - High-flow (Q95 = `q_high`) & Extreme-flow (Q99 = `q_extreme`) event detection:
          `n_pos_q_high`, `pos_rate_q_high`, `precision_q_high`, `recall_q_high`, `f1_q_high`,
          `n_pos_q_extreme`, `pos_rate_q_extreme`, `precision_q_extreme`, `recall_q_extreme`, `f1_q_extreme`
      - Probability calibration quality:
          `brier_q_high`, `brier_raw_q_high`, `cal_mae_q_high`,
          `brier_q_extreme`, `brier_raw_q_extreme`, `cal_mae_q_extreme`
    """
    if len(val_df) != len(preds_df):
        raise ValueError(
            f"Row count mismatch between val_df ({len(val_df)}) and preds_df ({len(preds_df)})."
        )

    meta = resolve_target_metadata(val_df, target_col=target_col)
    t_col = meta["target_col"]
    t_units = meta["target_units"]

    val_dates = pd.to_datetime(val_df[date_col], errors="coerce").dt.normalize()
    effective_val_cutoff = (
        pd.to_datetime(max_val_target_date).normalize()
        if max_val_target_date is not None
        else val_dates.max()
    )

    horizons_tuple = tuple(int(h) for h in horizons)
    targets_df = prepare_horizon_targets(
        val_df,
        horizons=horizons_tuple,
        target_col=t_col,
        station_col=station_col,
        date_col=date_col,
        max_target_date=effective_val_cutoff,
    )
    labels_df = prepare_exceedance_labels(
        val_df,
        thresh_info=thresh_info,
        horizons=horizons_tuple,
        target_col=t_col,
        station_col=station_col,
        date_col=date_col,
        max_target_date=effective_val_cutoff,
        allow_unseen_fallback=allow_unseen_fallback,
    )

    records: List[Dict[str, Any]] = []
    for h in horizons_tuple:
        valid_mask = (
            targets_df[f"valid_h{h}"].to_numpy(dtype=bool)
            & labels_df[f"valid_h{h}"].to_numpy(dtype=bool)
        )
        n_eval = int(np.sum(valid_mask))
        row: Dict[str, Any] = {
            "horizon": int(h),
            "n_eval": n_eval,
            "target_col": t_col,
            "target_units": t_units,
        }

        if n_eval == 0:
            for k in (
                "mae", "rmse", "persistence_mae", "persistence_rmse",
                "climatology_mae", "climatology_rmse",
                "skill_mae_vs_persistence", "skill_mae_vs_climatology",
                "pinball_q10", "pinball_q50", "pinball_q90", "mean_pinball_loss",
                "interval_coverage_80", "interval_width_80",
            ):
                row[k] = float("nan")
            for lvl in ("q_high", "q_extreme"):
                row[f"n_pos_{lvl}"] = 0
                for m in ("pos_rate", "precision", "recall", "f1", "brier", "brier_raw", "cal_mae"):
                    row[f"{m}_{lvl}"] = float("nan")
            records.append(row)
            continue

        y_true = targets_df.loc[valid_mask, f"y_h{h}"].to_numpy(dtype=np.float64)
        q50_pred = preds_df.loc[valid_mask, f"q50_h{h}"].to_numpy(dtype=np.float64)
        q10_pred = preds_df.loc[valid_mask, f"q10_h{h}"].to_numpy(dtype=np.float64)
        q90_pred = preds_df.loc[valid_mask, f"q90_h{h}"].to_numpy(dtype=np.float64)

        mae = float(mean_absolute_error(y_true, q50_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, q50_pred)))
        row["mae"] = mae
        row["rmse"] = rmse

        # Persistence baseline comparison
        pers_col = f"persistence_h{h}"
        if pers_col in preds_df.columns:
            p_pred = preds_df.loc[valid_mask, pers_col].to_numpy(dtype=np.float64)
            p_mae = float(mean_absolute_error(y_true, p_pred))
            p_rmse = float(np.sqrt(mean_squared_error(y_true, p_pred)))
            row["persistence_mae"] = p_mae
            row["persistence_rmse"] = p_rmse
            row["skill_mae_vs_persistence"] = float(1.0 - mae / p_mae) if p_mae > 1e-12 else float("nan")
        else:
            row["persistence_mae"] = float("nan")
            row["persistence_rmse"] = float("nan")
            row["skill_mae_vs_persistence"] = float("nan")

        # Climatology baseline comparison
        clim_col = f"clim_q50_h{h}"
        if clim_col in preds_df.columns:
            c_pred = preds_df.loc[valid_mask, clim_col].to_numpy(dtype=np.float64)
            c_mae = float(mean_absolute_error(y_true, c_pred))
            c_rmse = float(np.sqrt(mean_squared_error(y_true, c_pred)))
            row["climatology_mae"] = c_mae
            row["climatology_rmse"] = c_rmse
            row["skill_mae_vs_climatology"] = float(1.0 - mae / c_mae) if c_mae > 1e-12 else float("nan")
        else:
            row["climatology_mae"] = float("nan")
            row["climatology_rmse"] = float("nan")
            row["skill_mae_vs_climatology"] = float("nan")

        # Quantile pinball losses & 80% interval metrics
        pb10 = _compute_pinball_loss(y_true, q10_pred, 0.10)
        pb50 = _compute_pinball_loss(y_true, q50_pred, 0.50)
        pb90 = _compute_pinball_loss(y_true, q90_pred, 0.90)
        row["pinball_q10"] = pb10
        row["pinball_q50"] = pb50
        row["pinball_q90"] = pb90
        row["mean_pinball_loss"] = float(np.mean([pb10, pb50, pb90]))
        row["interval_coverage_80"] = float(np.mean((y_true >= q10_pred) & (y_true <= q90_pred)))
        row["interval_width_80"] = float(np.mean(q90_pred - q10_pred))

        # Q95 (q_high) and Q99 (q_extreme) classification and calibration metrics
        for lvl in ("q_high", "q_extreme"):
            y_bin = labels_df.loc[valid_mask, f"y_{lvl}_h{h}"].to_numpy(dtype=np.int32)
            n_pos = int(np.sum(y_bin == 1))
            row[f"n_pos_{lvl}"] = n_pos
            row[f"pos_rate_{lvl}"] = float(n_pos / n_eval)

            prob_col = f"prob_{lvl}_h{h}"
            raw_prob_col = f"raw_prob_{lvl}_h{h}"
            if prob_col in preds_df.columns:
                p_cal = preds_df.loc[valid_mask, prob_col].to_numpy(dtype=np.float64)
                p_raw = (
                    preds_df.loc[valid_mask, raw_prob_col].to_numpy(dtype=np.float64)
                    if raw_prob_col in preds_df.columns
                    else p_cal
                )
                y_hat = (p_cal >= float(prob_threshold)).astype(np.int32)
                row[f"precision_{lvl}"] = float(precision_score(y_bin, y_hat, zero_division=0.0))
                row[f"recall_{lvl}"] = float(recall_score(y_bin, y_hat, zero_division=0.0))
                row[f"f1_{lvl}"] = float(f1_score(y_bin, y_hat, zero_division=0.0))
                row[f"brier_{lvl}"] = float(np.mean((p_cal - y_bin.astype(np.float64)) ** 2))
                row[f"brier_raw_{lvl}"] = float(np.mean((p_raw - y_bin.astype(np.float64)) ** 2))
                row[f"cal_mae_{lvl}"] = _compute_calibration_mae(y_bin, p_cal)
            else:
                for m in ("precision", "recall", "f1", "brier", "brier_raw", "cal_mae"):
                    row[f"{m}_{lvl}"] = float("nan")

        records.append(row)

    return pd.DataFrame(records)


def _summarize_fold_metrics(fold_metrics_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Aggregate per-fold, per-horizon metrics into horizon summaries and overall summary."""
    if len(fold_metrics_df) == 0:
        return pd.DataFrame(), {}

    numeric_cols = [
        c
        for c in fold_metrics_df.select_dtypes(include=[np.number]).columns
        if c not in ("fold_id", "val_year", "n_train_rows", "n_val_rows", "n_train_stations", "n_val_stations")
    ]
    summary_by_horizon = (
        fold_metrics_df.groupby("horizon", as_index=False)[numeric_cols].mean(numeric_only=True)
    )

    overall: Dict[str, float] = {}
    for col in numeric_cols:
        if col == "horizon":
            continue
        overall[f"mean_{col}"] = float(fold_metrics_df[col].mean(skipna=True))
    return summary_by_horizon, overall


def validate_temporal(
    df: pd.DataFrame,
    features_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    val_years: Optional[List[int]] = None,
    gap_days: int = 30,
    horizons: Tuple[int, ...] = HORIZONS,
    quantiles: Tuple[float, ...] = QUANTILES,
    thresholds: Optional[Dict[str, float]] = None,
    params: Optional[Dict[str, Any]] = None,
    calibrate: bool = True,
    prob_threshold: float = 0.50,
    target_col: Optional[str] = None,
    station_col: str = STATION_COL,
    date_col: str = DATE_COL,
) -> Dict[str, Any]:
    """Run forward-chaining temporal validation across horizons (1, 3, 7 days).

    Enforces:
      - Purge gap `eff_gap_days = max(gap_days, max(horizons))` between training and validation.
      - Every training origin date AND horizon target date t_train + h is strictly <=
        `min(val_dates) - eff_gap_days`, preventing any target leakage across fold boundaries.
      - All baselines, quantile models, flood thresholds (Q95/Q99), classifiers, and
        isotonic calibrators are fitted strictly within each training fold.
    """
    if features_df is None:
        features_df = build_features(df)

    use_feat_cols = list(feature_cols) if feature_cols is not None else [
        c for c in FEATURE_COLS if c in features_df.columns
    ]
    years = list(val_years) if val_years is not None else [2019, 2020, 2021]
    horizons_tuple = tuple(int(h) for h in horizons)
    eff_gap_days = max(int(gap_days), max(horizons_tuple))

    all_dates = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    fold_dfs: List[pd.DataFrame] = []

    for fold_idx, (trn_idx, val_idx) in enumerate(forward_folds(df, val_years=years, gap_days=eff_gap_days)):
        val_sub = df.iloc[val_idx]
        val_min_date = all_dates.iloc[val_idx].min()
        val_max_date = all_dates.iloc[val_idx].max()
        val_year = int(val_min_date.year)

        # Enforce strict training cutoff so t_train + h never reaches val_min_date
        train_cutoff = val_min_date - pd.Timedelta(days=eff_gap_days)
        trn_dates = all_dates.iloc[trn_idx]
        keep_trn = (trn_dates <= train_cutoff).to_numpy(dtype=bool)
        trn_idx_clean = trn_idx[keep_trn]

        trn_sub = df.iloc[trn_idx_clean]
        X_trn = features_df.iloc[trn_idx_clean][use_feat_cols]
        X_val = features_df.iloc[val_idx][use_feat_cols]

        integrity = _verify_fold_integrity(
            trn_sub,
            val_sub,
            mode="temporal",
            horizons=horizons_tuple,
            effective_train_cutoff=train_cutoff,
            station_col=station_col,
            date_col=date_col,
        )

        bundle = fit_models(
            X_trn,
            train_df=trn_sub,
            params=params,
            horizons=horizons_tuple,
            quantiles=quantiles,
            thresholds=thresholds,
            calibrate=calibrate,
            target_col=target_col,
            station_col=station_col,
            date_col=date_col,
            max_target_date=train_cutoff,
        )

        preds_df = predict_multi_horizon(
            bundle,
            X_val,
            df=val_sub,
            include_baselines=True,
            calibrated=calibrate,
        )

        metrics_df = compute_horizon_metrics(
            val_sub,
            preds_df,
            thresh_info=bundle["exceedance"]["thresh_info"],
            horizons=horizons_tuple,
            quantiles=quantiles,
            prob_threshold=prob_threshold,
            target_col=target_col,
            station_col=station_col,
            date_col=date_col,
            max_val_target_date=val_max_date,
            allow_unseen_fallback=True,
        )

        metrics_df.insert(0, "fold_id", fold_idx)
        metrics_df.insert(1, "val_year", val_year)
        metrics_df["max_train_origin_date"] = integrity["max_train_origin_date"]
        metrics_df["effective_train_cutoff"] = integrity["effective_train_cutoff"]
        metrics_df["min_val_origin_date"] = integrity["min_val_origin_date"]
        metrics_df["max_val_origin_date"] = integrity["max_val_origin_date"]
        metrics_df["n_train_rows"] = integrity["n_train_rows"]
        metrics_df["n_val_rows"] = integrity["n_val_rows"]
        fold_dfs.append(metrics_df)

    fold_metrics = pd.concat(fold_dfs, ignore_index=True) if fold_dfs else pd.DataFrame()
    summary_by_horizon, overall_summary = _summarize_fold_metrics(fold_metrics)

    return {
        "mode": "temporal",
        "fold_metrics": fold_metrics,
        "summary_by_horizon": summary_by_horizon,
        "overall_summary": overall_summary,
    }


def validate_unseen_stations(
    df: pd.DataFrame,
    features_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    val_stations: Optional[List[str]] = None,
    max_stations: Optional[int] = None,
    horizons: Tuple[int, ...] = HORIZONS,
    quantiles: Tuple[float, ...] = QUANTILES,
    thresholds: Optional[Dict[str, float]] = None,
    params: Optional[Dict[str, Any]] = None,
    calibrate: bool = True,
    prob_threshold: float = 0.50,
    allow_unseen_threshold_fallback: bool = True,
    target_col: Optional[str] = None,
    station_col: str = STATION_COL,
    date_col: str = DATE_COL,
) -> Dict[str, Any]:
    """Run whole-station / Leave-One-Station-Out (LOSO) validation for unseen basins.

    Enforces:
      - Zero station overlap between `train_df` and `val_df`.
      - Held-out validation stations never contribute to model training, flood thresholds
        (Q95/Q99), climatology, persistence, or isotonic probability calibration.
      - Unseen-basin flood thresholds are resolved strictly via the training-only unseen-basin
        policy (`area_scaling_prior` for cfs or `global_thresholds` for mm/day) when
        `allow_unseen_threshold_fallback=True`.
    """
    if features_df is None:
        features_df = build_features(df)

    use_feat_cols = list(feature_cols) if feature_cols is not None else [
        c for c in FEATURE_COLS if c in features_df.columns
    ]
    horizons_tuple = tuple(int(h) for h in horizons)
    target_val_stations = set(str(s) for s in val_stations) if val_stations is not None else None

    fold_dfs: List[pd.DataFrame] = []
    fold_count = 0

    for trn_idx, val_idx in loso_folds(df):
        val_sub = df.iloc[val_idx]
        held_out_stations = sorted(val_sub[station_col].astype(str).unique())
        held_out_id = held_out_stations[0]

        if target_val_stations is not None and held_out_id not in target_val_stations:
            continue
        if max_stations is not None and fold_count >= int(max_stations):
            break

        trn_sub = df.iloc[trn_idx]
        X_trn = features_df.iloc[trn_idx][use_feat_cols]
        X_val = features_df.iloc[val_idx][use_feat_cols]

        integrity = _verify_fold_integrity(
            trn_sub,
            val_sub,
            mode="unseen_station",
            horizons=horizons_tuple,
            station_col=station_col,
            date_col=date_col,
        )

        bundle = fit_models(
            X_trn,
            train_df=trn_sub,
            params=params,
            horizons=horizons_tuple,
            quantiles=quantiles,
            thresholds=thresholds,
            calibrate=calibrate,
            target_col=target_col,
            station_col=station_col,
            date_col=date_col,
        )

        # Double-check that held-out station(s) did not leak into any sub-bundle
        _verify_no_unseen_station_in_bundle(bundle, held_out_stations)

        preds_df = predict_multi_horizon(
            bundle,
            X_val,
            df=val_sub,
            include_baselines=True,
            calibrated=calibrate,
        )

        metrics_df = compute_horizon_metrics(
            val_sub,
            preds_df,
            thresh_info=bundle["exceedance"]["thresh_info"],
            horizons=horizons_tuple,
            quantiles=quantiles,
            prob_threshold=prob_threshold,
            target_col=target_col,
            station_col=station_col,
            date_col=date_col,
            allow_unseen_fallback=allow_unseen_threshold_fallback,
        )

        t_info = bundle["exceedance"]["thresh_info"]
        policy_name = (
            "area_scaling_prior"
            if t_info.get("area_scaling_prior") is not None
            else "global_thresholds"
        )
        metrics_df.insert(0, "fold_id", fold_count)
        metrics_df.insert(1, "val_station", held_out_id)
        metrics_df["n_train_stations"] = integrity["n_train_stations"]
        metrics_df["n_val_stations"] = integrity["n_val_stations"]
        metrics_df["n_train_rows"] = integrity["n_train_rows"]
        metrics_df["n_val_rows"] = integrity["n_val_rows"]
        metrics_df["unseen_threshold_policy"] = policy_name
        fold_dfs.append(metrics_df)
        fold_count += 1

    fold_metrics = pd.concat(fold_dfs, ignore_index=True) if fold_dfs else pd.DataFrame()
    summary_by_horizon, overall_summary = _summarize_fold_metrics(fold_metrics)

    return {
        "mode": "unseen_station",
        "fold_metrics": fold_metrics,
        "summary_by_horizon": summary_by_horizon,
        "overall_summary": overall_summary,
    }


def evaluate_basinshift(
    df: pd.DataFrame,
    features_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    val_years: Optional[List[int]] = None,
    val_stations: Optional[List[str]] = None,
    max_loso_stations: Optional[int] = None,
    gap_days: int = 30,
    horizons: Tuple[int, ...] = HORIZONS,
    quantiles: Tuple[float, ...] = QUANTILES,
    thresholds: Optional[Dict[str, float]] = None,
    params: Optional[Dict[str, Any]] = None,
    calibrate: bool = True,
    prob_threshold: float = 0.50,
    run_temporal: bool = True,
    run_unseen_station: bool = True,
    target_col: Optional[str] = None,
    station_col: str = STATION_COL,
    date_col: str = DATE_COL,
) -> Dict[str, Any]:
    """Run complete BASINSHIFT temporal and unseen-station (LOSO) validation suite."""
    if features_df is None:
        features_df = build_features(df)

    results: Dict[str, Any] = {}
    if run_temporal:
        results["temporal"] = validate_temporal(
            df,
            features_df=features_df,
            feature_cols=feature_cols,
            val_years=val_years,
            gap_days=gap_days,
            horizons=horizons,
            quantiles=quantiles,
            thresholds=thresholds,
            params=params,
            calibrate=calibrate,
            prob_threshold=prob_threshold,
            target_col=target_col,
            station_col=station_col,
            date_col=date_col,
        )

    if run_unseen_station:
        results["unseen_station"] = validate_unseen_stations(
            df,
            features_df=features_df,
            feature_cols=feature_cols,
            val_stations=val_stations,
            max_stations=max_loso_stations,
            horizons=horizons,
            quantiles=quantiles,
            thresholds=thresholds,
            params=params,
            calibrate=calibrate,
            prob_threshold=prob_threshold,
            target_col=target_col,
            station_col=station_col,
            date_col=date_col,
        )

    return results


def attach_static_attributes(
    df: pd.DataFrame,
    static_df: Optional[pd.DataFrame] = None,
    station_col: str = STATION_COL,
) -> pd.DataFrame:
    """Attach static catchment attributes by `station_id` without modifying Lane B or leaking targets.

    Bridges the interface between Caravan (`df, static_df`) and Lane B (`build_features(df)`):
      - Rejects or strips any target columns (`discharge_cfs`, `streamflow`) from `static_df`
        so static attributes for unseen stations never carry target information.
      - Preserves exact row order and index of `df`.
    """
    if static_df is None or len(static_df) == 0:
        return df

    if station_col not in df.columns:
        raise KeyError(f"Missing station column '{station_col}' in df.")
    if station_col not in static_df.columns:
        raise KeyError(f"Missing station column '{station_col}' in static_df.")

    forbidden_targets = {TARGET_COL, CARAVAN_TARGET_COL, "y_origin", "z_origin"}
    leaked_cols = [c for c in static_df.columns if c in forbidden_targets]
    if leaked_cols:
        raise ValueError(
            f"Target column(s) {leaked_cols} detected inside static_df! "
            "Static catchment attributes must never contain flow target values."
        )

    new_cols = [c for c in static_df.columns if c != station_col and c not in df.columns]
    if not new_cols:
        return df

    static_dedup = static_df[[station_col, *new_cols]].drop_duplicates(subset=[station_col], keep="first")
    orig_index = df.index
    merged = df.merge(static_dedup, on=station_col, how="left", sort=False)
    merged.index = orig_index
    return merged


def get_feature_groups(
    features_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    static_df: Optional[pd.DataFrame] = None,
    station_col: str = STATION_COL,
    date_col: str = DATE_COL,
) -> Dict[str, List[str]]:
    """Partition feature columns into `dynamic_only` and `full_static_plus_dynamic` groups.

    Ensures:
      - `dynamic_only` excludes all raw static catchment attributes (`STATIC_RAW_FEATURE_COLS`),
        all engineered static morphology & static-interaction features (`STATIC_DERIVED_FEATURE_COLS`),
        and any extra static columns supplied via `static_df`.
      - `full_static_plus_dynamic` includes all valid feature columns (dynamic + static).
      - Forbidden identifier/target columns (`station_id`, `date`, `discharge_cfs`, `streamflow`)
        are excluded from both groups.
    """
    forbidden = {station_col, date_col, TARGET_COL, CARAVAN_TARGET_COL}
    if feature_cols is not None:
        base_cols = [c for c in feature_cols if c not in forbidden]
    elif features_df is not None:
        preferred = [c for c in FEATURE_COLS if c in features_df.columns and c not in forbidden]
        extra = [
            c
            for c in features_df.columns
            if c not in preferred and c not in forbidden and pd.api.types.is_numeric_dtype(features_df[c])
        ]
        base_cols = preferred + extra
    else:
        base_cols = [c for c in FEATURE_COLS if c not in forbidden]

    explicit_static = set(STATIC_FEATURE_COLS)
    if static_df is not None:
        explicit_static |= {c for c in static_df.columns if c not in forbidden}

    static_raw_only = [c for c in base_cols if c in STATIC_RAW_FEATURE_COLS]
    static_derived_only = [c for c in base_cols if c in STATIC_DERIVED_FEATURE_COLS]
    static_removed = [c for c in base_cols if c in explicit_static]
    dynamic_only = [c for c in base_cols if c not in explicit_static]

    if not dynamic_only:
        raise ValueError("dynamic_only feature group is empty after removing static features.")

    return {
        "full_static_plus_dynamic": list(base_cols),
        "dynamic_only": dynamic_only,
        "static_removed": static_removed,
        "static_raw_only": static_raw_only,
        "static_derived_only": static_derived_only,
    }


def _compare_ablation_mode(
    dyn_res: Dict[str, Any],
    full_res: Dict[str, Any],
    mode: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Verify fold consistency and compute per-horizon & overall metric differences."""
    dyn_folds = dyn_res["fold_metrics"]
    full_folds = full_res["fold_metrics"]

    if len(dyn_folds) != len(full_folds):
        raise ValueError(
            f"Ablation fold count mismatch in mode '{mode}': "
            f"dynamic_only has {len(dyn_folds)} rows vs full_static_plus_dynamic has {len(full_folds)} rows."
        )

    # Verify identical fold boundaries, row counts, and positive event counts
    check_cols = ["fold_id", "horizon", "n_train_rows", "n_val_rows", "n_eval", "n_pos_q_high", "n_pos_q_extreme"]
    if mode == "temporal":
        check_cols.extend(["val_year", "max_train_origin_date", "effective_train_cutoff", "min_val_origin_date"])
    elif mode == "unseen_station":
        check_cols.extend(["val_station", "n_train_stations", "n_val_stations"])

    for col in check_cols:
        if col in dyn_folds.columns and col in full_folds.columns:
            if not dyn_folds[col].equals(full_folds[col]):
                raise ValueError(
                    f"Fold consistency violation in mode '{mode}' on column '{col}': "
                    "dynamic_only and full_static_plus_dynamic were not evaluated on identical folds!"
                )

    dyn_sum = dyn_res["summary_by_horizon"].sort_values("horizon").reset_index(drop=True)
    full_sum = full_res["summary_by_horizon"].sort_values("horizon").reset_index(drop=True)

    lower_is_better = (
        "mae",
        "rmse",
        "pinball_q10",
        "pinball_q50",
        "pinball_q90",
        "mean_pinball_loss",
        "brier_q_high",
        "brier_raw_q_high",
        "cal_mae_q_high",
        "brier_q_extreme",
        "brier_raw_q_extreme",
        "cal_mae_q_extreme",
        "interval_width_80",
    )
    higher_is_better = (
        "skill_mae_vs_persistence",
        "skill_mae_vs_climatology",
        "interval_coverage_80",
        "precision_q_high",
        "recall_q_high",
        "f1_q_high",
        "precision_q_extreme",
        "recall_q_extreme",
        "f1_q_extreme",
    )

    records: List[Dict[str, Any]] = []
    for idx in range(len(dyn_sum)):
        d_row = dyn_sum.iloc[idx]
        f_row = full_sum.iloc[idx]
        h = int(d_row["horizon"])
        rec: Dict[str, Any] = {
            "mode": mode,
            "horizon": h,
            "n_eval": float(d_row.get("n_eval", np.nan)),
        }

        for m in lower_is_better:
            if m in d_row and m in f_row:
                d_val = float(d_row[m])
                f_val = float(f_row[m])
                rec[f"{m}_dynamic_only"] = d_val
                rec[f"{m}_full_static_plus_dynamic"] = f_val
                # delta = full - dynamic (negative means full reduced error/loss)
                rec[f"delta_{m}"] = float(f_val - d_val)
                rec[f"rel_gain_pct_{m}"] = (
                    float(100.0 * (d_val - f_val) / abs(d_val)) if abs(d_val) > 1e-12 else 0.0
                )

        for m in higher_is_better:
            if m in d_row and m in f_row:
                d_val = float(d_row[m])
                f_val = float(f_row[m])
                rec[f"{m}_dynamic_only"] = d_val
                rec[f"{m}_full_static_plus_dynamic"] = f_val
                # delta = full - dynamic (positive means full improved skill/F1/coverage)
                rec[f"delta_{m}"] = float(f_val - d_val)

        records.append(rec)

    comp_df = pd.DataFrame(records)
    overall_diff: Dict[str, float] = {}
    for col in comp_df.select_dtypes(include=[np.number]).columns:
        if col in ("horizon", "n_eval"):
            continue
        overall_diff[f"mean_{col}"] = float(comp_df[col].mean(skipna=True))

    return comp_df, overall_diff


def run_static_ablation(
    df: pd.DataFrame,
    features_df: Optional[pd.DataFrame] = None,
    static_df: Optional[pd.DataFrame] = None,
    val_years: Optional[List[int]] = None,
    val_stations: Optional[List[str]] = None,
    max_loso_stations: Optional[int] = None,
    gap_days: int = 30,
    horizons: Tuple[int, ...] = HORIZONS,
    quantiles: Tuple[float, ...] = QUANTILES,
    thresholds: Optional[Dict[str, float]] = None,
    params: Optional[Dict[str, Any]] = None,
    calibrate: bool = True,
    prob_threshold: float = 0.50,
    run_temporal: bool = True,
    run_unseen_station: bool = True,
    target_col: Optional[str] = None,
    station_col: str = STATION_COL,
    date_col: str = DATE_COL,
) -> Dict[str, Any]:
    """Compare BASINSHIFT performance with and without static catchment attributes.

    Evaluates two feature variants on identical temporal and unseen-station (LOSO) folds
    across all horizons (1, 3, 7 days):
      1. `dynamic_only`: strictly meteorological, soil moisture, temperature, and seasonality
         features (all 14 static and static-derived/interaction features removed).
      2. `full_static_plus_dynamic`: all dynamic + static catchment features.
    """
    work_df = attach_static_attributes(df, static_df=static_df, station_col=station_col)
    if features_df is None:
        features_df = build_features(work_df)

    feature_groups = get_feature_groups(
        features_df=features_df,
        static_df=static_df,
        station_col=station_col,
        date_col=date_col,
    )

    dyn_cols = feature_groups["dynamic_only"]
    full_cols = feature_groups["full_static_plus_dynamic"]

    dyn_eval = evaluate_basinshift(
        work_df,
        features_df=features_df,
        feature_cols=dyn_cols,
        val_years=val_years,
        val_stations=val_stations,
        max_loso_stations=max_loso_stations,
        gap_days=gap_days,
        horizons=horizons,
        quantiles=quantiles,
        thresholds=thresholds,
        params=params,
        calibrate=calibrate,
        prob_threshold=prob_threshold,
        run_temporal=run_temporal,
        run_unseen_station=run_unseen_station,
        target_col=target_col,
        station_col=station_col,
        date_col=date_col,
    )

    full_eval = evaluate_basinshift(
        work_df,
        features_df=features_df,
        feature_cols=full_cols,
        val_years=val_years,
        val_stations=val_stations,
        max_loso_stations=max_loso_stations,
        gap_days=gap_days,
        horizons=horizons,
        quantiles=quantiles,
        thresholds=thresholds,
        params=params,
        calibrate=calibrate,
        prob_threshold=prob_threshold,
        run_temporal=run_temporal,
        run_unseen_station=run_unseen_station,
        target_col=target_col,
        station_col=station_col,
        date_col=date_col,
    )

    comparison_frames: List[pd.DataFrame] = []
    overall_comparison: Dict[str, Dict[str, float]] = {}

    for mode in ("temporal", "unseen_station"):
        if mode in dyn_eval and mode in full_eval:
            comp_df, overall_diff = _compare_ablation_mode(dyn_eval[mode], full_eval[mode], mode=mode)
            comparison_frames.append(comp_df)
            overall_comparison[mode] = overall_diff

    comparison_by_horizon = (
        pd.concat(comparison_frames, ignore_index=True) if comparison_frames else pd.DataFrame()
    )

    return {
        "feature_groups": feature_groups,
        "variants": {
            "dynamic_only": dyn_eval,
            "full_static_plus_dynamic": full_eval,
        },
        "comparison_by_horizon": comparison_by_horizon,
        "overall_comparison": overall_comparison,
    }


def _get_git_commit() -> str:
    """Return short git commit hash or 'unknown' if unavailable."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        commit = res.stdout.strip()
        return commit if commit else "unknown"
    except Exception:
        return "unknown"


def _append_rows_to_experiment_log(
    new_rows_df: pd.DataFrame,
    log_path: str | Path = EXPERIMENTS_PATH,
) -> pd.DataFrame:
    """Append new experiment records to `log_path` without overwriting any previous records."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure all standard EXPERIMENT_LOG_COLUMNS exist in new_rows_df
    for col in EXPERIMENT_LOG_COLUMNS:
        if col not in new_rows_df.columns:
            new_rows_df[col] = np.nan

    if path.exists() and path.stat().st_size > 0:
        existing_df = pd.read_csv(path, encoding="utf-8")
        if len(existing_df) > 0:
            # Preserve all existing rows and columns while aligning with EXPERIMENT_LOG_COLUMNS
            combined_cols = list(EXPERIMENT_LOG_COLUMNS) + [
                c for c in existing_df.columns if c not in EXPERIMENT_LOG_COLUMNS
            ]
            for col in combined_cols:
                if col not in existing_df.columns:
                    existing_df[col] = np.nan
                if col not in new_rows_df.columns:
                    new_rows_df[col] = np.nan
            combined_df = pd.concat(
                [existing_df[combined_cols], new_rows_df[combined_cols]],
                ignore_index=True,
            )
            combined_df.to_csv(path, index=False, encoding="utf-8")
            return new_rows_df[list(EXPERIMENT_LOG_COLUMNS)]

    ordered_new = new_rows_df[list(EXPERIMENT_LOG_COLUMNS)]
    ordered_new.to_csv(path, index=False, encoding="utf-8")
    return ordered_new


def _validate_experiment_inputs(
    description: str,
    model_config: Optional[Dict[str, Any] | str],
    feature_set: str,
    validation_strategy: str,
    random_seed: int,
) -> Tuple[str, str]:
    """Enforce required experiment metadata, seed=42, and prohibition on holdout/fabricated logs."""
    if not description or not str(description).strip():
        raise ValueError("Experiment `description` is required and cannot be empty.")
    if not feature_set or not str(feature_set).strip():
        raise ValueError("Experiment `feature_set` is required and cannot be empty.")
    if not validation_strategy or not str(validation_strategy).strip():
        raise ValueError("Experiment `validation_strategy` is required and cannot be empty.")
    if model_config is None or (isinstance(model_config, (dict, str)) and len(model_config) == 0):
        raise ValueError("Experiment `model_config` is required and cannot be empty.")

    if int(random_seed) != int(RANDOM_STATE):
        raise ValueError(
            f"Reproducibility violation: random_seed must be {RANDOM_STATE}, got {random_seed}."
        )

    # Reject any attempt to log holdout/test labels or fabricated/synthetic metrics
    forbidden_tokens = ("holdout", "test_label", "test_set", "fabricated", "synthetic", "fake")
    check_text = f"{validation_strategy} {description} {feature_set}".lower()
    for token in forbidden_tokens:
        if token in check_text:
            raise ValueError(
                f"Forbidden experiment source or token '{token}' detected! "
                "Never log test-label, holdout-label, or fabricated results."
            )

    if isinstance(model_config, str):
        config_str = model_config
    else:
        config_dict = dict(model_config)
        if "n_jobs" in config_dict:
            config_dict["n_jobs"] = _sanitize_n_jobs(config_dict["n_jobs"])
        config_str = json.dumps(config_dict, sort_keys=True, default=str)

    return config_str, str(validation_strategy).strip()


def log_basinshift_experiment(
    eval_result: Optional[Dict[str, Any] | pd.DataFrame] = None,
    *,
    description: str,
    model_config: Dict[str, Any] | str,
    feature_set: str = "full_static_plus_dynamic",
    n_features: Optional[int] = None,
    validation_strategy: Optional[str] = None,
    random_seed: int = RANDOM_STATE,
    experiment_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    commit: Optional[str] = None,
    status: str = "completed",
    error_message: str = "",
    allow_missing_metrics: bool = False,
    log_path: str | Path = EXPERIMENTS_PATH,
) -> pd.DataFrame:
    """Log BASINSHIFT validation or static-ablation results to `experiments/` without overwriting.

    Supports logging:
      1. Single validation mode output (`validate_temporal` or `validate_unseen_stations` dict,
         or a `summary_by_horizon` DataFrame).
      2. Combined `evaluate_basinshift` output (`{'temporal': ..., 'unseen_station': ...}`).
      3. Static-feature ablation output (`run_static_ablation` dict), recording both
         `dynamic_only` and `full_static_plus_dynamic` horizon rows plus `ablation_delta_*` fields.
      4. Failed experiments (`status='failed'`), recording exact metadata and `error_message`
         with NaN metrics for full auditability.
    """
    if status not in ("completed", "failed"):
        raise ValueError(f"Invalid experiment status '{status}'. Expected 'completed' or 'failed'.")

    # Infer validation_strategy if not explicitly provided and eval_result is a dict
    if validation_strategy is None and isinstance(eval_result, dict):
        if "variants" in eval_result and "comparison_by_horizon" in eval_result:
            validation_strategy = "static_ablation"
        elif "mode" in eval_result:
            validation_strategy = str(eval_result["mode"])
        elif "temporal" in eval_result or "unseen_station" in eval_result:
            validation_strategy = "temporal_and_unseen_station"

    config_str, val_strat = _validate_experiment_inputs(
        description=description,
        model_config=model_config,
        feature_set=feature_set,
        validation_strategy=validation_strategy or "",
        random_seed=random_seed,
    )

    ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    git_commit = commit or _get_git_commit()
    exp_id = experiment_id or f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    if status == "failed":
        failed_row: Dict[str, Any] = {col: np.nan for col in EXPERIMENT_LOG_COLUMNS}
        failed_row.update(
            {
                "experiment_id": exp_id,
                "timestamp": ts,
                "status": "failed",
                "error_message": str(error_message or "unspecified_error"),
                "commit": git_commit,
                "description": str(description).strip(),
                "model_config": config_str,
                "feature_set": str(feature_set).strip(),
                "n_features": int(n_features) if n_features is not None else np.nan,
                "validation_strategy": val_strat,
                "random_seed": int(random_seed),
                "target_col": TARGET_COL,
                "target_units": DATA_V2_TARGET_UNITS,
            }
        )
        return _append_rows_to_experiment_log(pd.DataFrame([failed_row]), log_path=log_path)

    if eval_result is None:
        raise ValueError("`eval_result` cannot be None when logging a completed experiment.")

    required_metric_cols = (
        "mae",
        "rmse",
        "mean_pinball_loss",
        "interval_coverage_80",
        "f1_q_high",
        "brier_q_high",
        "f1_q_extreme",
        "brier_q_extreme",
    )
    metric_copy_cols = (
        "horizon",
        "n_eval",
        "mae",
        "rmse",
        "persistence_mae",
        "persistence_rmse",
        "climatology_mae",
        "climatology_rmse",
        "skill_mae_vs_persistence",
        "skill_mae_vs_climatology",
        "pinball_q10",
        "pinball_q50",
        "pinball_q90",
        "mean_pinball_loss",
        "interval_coverage_80",
        "interval_width_80",
        "n_pos_q_high",
        "pos_rate_q_high",
        "precision_q_high",
        "recall_q_high",
        "f1_q_high",
        "brier_q_high",
        "brier_raw_q_high",
        "cal_mae_q_high",
        "n_pos_q_extreme",
        "pos_rate_q_extreme",
        "precision_q_extreme",
        "recall_q_extreme",
        "f1_q_extreme",
        "brier_q_extreme",
        "brier_raw_q_extreme",
        "cal_mae_q_extreme",
    )

    def _build_rows_from_summary(
        summary_df: pd.DataFrame,
        strat_label: str,
        feat_label: str,
        feat_count: Optional[int],
        ablation_comp_df: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Any]]:
        if summary_df is None or len(summary_df) == 0:
            raise ValueError(f"Empty summary_by_horizon DataFrame for strategy '{strat_label}'.")
        if "horizon" not in summary_df.columns:
            raise ValueError("summary_by_horizon DataFrame is missing required 'horizon' column.")

        missing_req = [c for c in required_metric_cols if c not in summary_df.columns]
        if missing_req and not allow_missing_metrics:
            raise ValueError(
                f"Completed experiment is missing required metric columns: {missing_req}. "
                "Pass `allow_missing_metrics=True` only when logging partial diagnostic tables."
            )

        rows: List[Dict[str, Any]] = []
        for _, s_row in summary_df.sort_values("horizon").iterrows():
            h_val = int(s_row["horizon"])
            n_eval_val = float(s_row.get("n_eval", np.nan))
            if not allow_missing_metrics and (not np.isfinite(n_eval_val) or n_eval_val <= 0):
                raise ValueError(f"Invalid or zero n_eval ({n_eval_val}) at horizon h={h_val}.")

            rec: Dict[str, Any] = {col: np.nan for col in EXPERIMENT_LOG_COLUMNS}
            rec.update(
                {
                    "experiment_id": exp_id,
                    "timestamp": ts,
                    "status": "completed",
                    "error_message": "",
                    "commit": git_commit,
                    "description": str(description).strip(),
                    "model_config": config_str,
                    "feature_set": feat_label,
                    "n_features": int(feat_count) if feat_count is not None else np.nan,
                    "validation_strategy": strat_label,
                    "random_seed": int(random_seed),
                    "target_col": str(s_row.get("target_col", TARGET_COL)),
                    "target_units": str(s_row.get("target_units", DATA_V2_TARGET_UNITS)),
                }
            )
            for m_col in metric_copy_cols:
                if m_col in s_row:
                    val = s_row[m_col]
                    if not allow_missing_metrics and m_col in required_metric_cols and not np.isfinite(float(val)):
                        raise ValueError(
                            f"Non-finite required metric '{m_col}'={val} at horizon h={h_val}."
                        )
                    rec[m_col] = int(val) if m_col == "horizon" else float(val)

            if ablation_comp_df is not None and len(ablation_comp_df) > 0:
                sub_comp = ablation_comp_df[ablation_comp_df["horizon"] == h_val]
                if len(sub_comp) > 0:
                    c_row = sub_comp.iloc[0]
                    rec["ablation_delta_mae"] = float(c_row.get("delta_mae", np.nan))
                    rec["ablation_rel_gain_pct_mae"] = float(c_row.get("rel_gain_pct_mae", np.nan))
                    rec["ablation_delta_rmse"] = float(c_row.get("delta_rmse", np.nan))
                    rec["ablation_delta_mean_pinball_loss"] = float(
                        c_row.get("delta_mean_pinball_loss", np.nan)
                    )
                    rec["ablation_delta_f1_q_high"] = float(c_row.get("delta_f1_q_high", np.nan))
                    rec["ablation_delta_f1_q_extreme"] = float(c_row.get("delta_f1_q_extreme", np.nan))
                    rec["ablation_delta_brier_q_high"] = float(c_row.get("delta_brier_q_high", np.nan))
                    rec["ablation_delta_brier_q_extreme"] = float(
                        c_row.get("delta_brier_q_extreme", np.nan)
                    )
                    rec["ablation_delta_interval_coverage_80"] = float(
                        c_row.get("delta_interval_coverage_80", np.nan)
                    )

            rows.append(rec)
        return rows

    all_new_rows: List[Dict[str, Any]] = []

    if isinstance(eval_result, pd.DataFrame):
        all_new_rows.extend(
            _build_rows_from_summary(
                eval_result,
                strat_label=val_strat,
                feat_label=str(feature_set).strip(),
                feat_count=n_features,
            )
        )
    elif isinstance(eval_result, dict):
        # Case 1: Static-feature ablation result dictionary
        if "variants" in eval_result and "comparison_by_horizon" in eval_result:
            f_groups = eval_result.get("feature_groups", {})
            comp_all = eval_result["comparison_by_horizon"]
            for var_name in ("dynamic_only", "full_static_plus_dynamic"):
                if var_name not in eval_result["variants"]:
                    continue
                var_eval = eval_result["variants"][var_name]
                var_n_feat = len(f_groups.get(var_name, [])) if var_name in f_groups else n_features
                for mode in ("temporal", "unseen_station"):
                    if mode in var_eval and "summary_by_horizon" in var_eval[mode]:
                        mode_comp = (
                            comp_all[comp_all["mode"] == mode]
                            if len(comp_all) > 0 and "mode" in comp_all.columns
                            else None
                        )
                        all_new_rows.extend(
                            _build_rows_from_summary(
                                var_eval[mode]["summary_by_horizon"],
                                strat_label=f"static_ablation_{mode}",
                                feat_label=var_name,
                                feat_count=var_n_feat,
                                ablation_comp_df=mode_comp,
                            )
                        )
        # Case 2: Single validation mode output (`validate_temporal` or `validate_unseen_stations`)
        elif "summary_by_horizon" in eval_result:
            strat = str(eval_result.get("mode", val_strat))
            all_new_rows.extend(
                _build_rows_from_summary(
                    eval_result["summary_by_horizon"],
                    strat_label=strat,
                    feat_label=str(feature_set).strip(),
                    feat_count=n_features,
                )
            )
        # Case 3: Combined `evaluate_basinshift` output (`{'temporal': ..., 'unseen_station': ...}`)
        elif "temporal" in eval_result or "unseen_station" in eval_result:
            for mode in ("temporal", "unseen_station"):
                if mode in eval_result and "summary_by_horizon" in eval_result[mode]:
                    all_new_rows.extend(
                        _build_rows_from_summary(
                            eval_result[mode]["summary_by_horizon"],
                            strat_label=mode,
                            feat_label=str(feature_set).strip(),
                            feat_count=n_features,
                        )
                    )
        else:
            raise ValueError("Unrecognized `eval_result` dictionary structure in log_basinshift_experiment.")
    else:
        raise TypeError(f"Unsupported `eval_result` type: {type(eval_result)}.")

    if not all_new_rows:
        raise ValueError("No valid experiment rows were extracted from `eval_result`.")

    return _append_rows_to_experiment_log(pd.DataFrame(all_new_rows), log_path=log_path)


def run_and_log_experiment(
    runner_fn: Callable[..., Dict[str, Any] | pd.DataFrame],
    *runner_args: Any,
    description: str,
    model_config: Dict[str, Any] | str,
    feature_set: str = "full_static_plus_dynamic",
    n_features: Optional[int] = None,
    validation_strategy: str = "temporal",
    random_seed: int = RANDOM_STATE,
    experiment_id: Optional[str] = None,
    log_path: str | Path = EXPERIMENTS_PATH,
    raise_on_error: bool = False,
    **runner_kwargs: Any,
) -> Tuple[Optional[Dict[str, Any] | pd.DataFrame], pd.DataFrame]:
    """Execute `runner_fn` and log either the completed metrics or a failed experiment record."""
    # Pre-validate metadata and seed before running
    _validate_experiment_inputs(
        description=description,
        model_config=model_config,
        feature_set=feature_set,
        validation_strategy=validation_strategy,
        random_seed=random_seed,
    )

    try:
        result = runner_fn(*runner_args, **runner_kwargs)
        logged_df = log_basinshift_experiment(
            result,
            description=description,
            model_config=model_config,
            feature_set=feature_set,
            n_features=n_features,
            validation_strategy=validation_strategy,
            random_seed=random_seed,
            experiment_id=experiment_id,
            status="completed",
            log_path=log_path,
        )
        return result, logged_df
    except Exception as exc:
        logged_df = log_basinshift_experiment(
            None,
            description=description,
            model_config=model_config,
            feature_set=feature_set,
            n_features=n_features,
            validation_strategy=validation_strategy,
            random_seed=random_seed,
            experiment_id=experiment_id,
            status="failed",
            error_message=f"{type(exc).__name__}: {exc}",
            log_path=log_path,
        )
        if raise_on_error:
            raise
        return None, logged_df

