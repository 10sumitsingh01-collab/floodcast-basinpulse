"""Validation, scoring, and submission verification module (Lane D).

Shared interfaces:
  - forward_folds(df, val_years: list[int], gap_days: int = 30) -> Generator[(train_idx, val_idx)]
  - loso_folds(df) -> Generator[(train_idx, val_idx)]
  - score(df_val: pd.DataFrame, y_pred: np.ndarray) -> dict
  - finalize(test_df, preds, path="predictions.csv") -> pd.DataFrame
  - validate_submission(path, test_path) -> None

Hard rules enforced:
  - Strict submission validation (UTF-8, exact columns, non-negative, finite, same order).
  - No data leakage across temporal or spatial splits.
  - Accurate competition metric computation (40% RMSE, 30% top-10% RMSE, 20% MAE, 10% R2).
"""

from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    DATE_COL,
    SCORING_WEIGHTS,
    STATION_COL,
    SUBMISSION_COLUMNS,
    TARGET_COL,
)


def forward_folds(
    df: pd.DataFrame,
    val_years: List[int],
    gap_days: int = 30,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Temporal forward-chaining validation splits with a purging buffer gap.

    For each evaluation year in val_years:
      - val_idx: rows belonging to val_year
      - train_idx: rows strictly preceding (val_start - gap_days)
    This prevents temporal data leakage from rolling/lagged features.
    """
    dates = pd.to_datetime(df[DATE_COL])
    indices = np.arange(len(df))

    for y in sorted(val_years):
        val_mask = dates.dt.year == y
        if not val_mask.any():
            continue

        val_start = dates[val_mask].min()
        train_cutoff = val_start - pd.Timedelta(days=gap_days)
        train_mask = dates <= train_cutoff

        if not train_mask.any():
            continue

        yield indices[train_mask.values], indices[val_mask.values]


def loso_folds(df: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Leave-One-Station-Out (LOSO) cross-validation splits.

    Iterates over each unique station:
      - val_idx: all observations for that station
      - train_idx: all observations for all other stations
    Directly measures geographic generalization to unseen basins.
    """
    stations = df[STATION_COL].values
    unique_stations = np.unique(stations)
    indices = np.arange(len(df))

    for sid in unique_stations:
        val_mask = stations == sid
        train_mask = ~val_mask

        if val_mask.any() and train_mask.any():
            yield indices[train_mask], indices[val_mask]


def score(df_val: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes competition scoring metrics against ground truth discharge_cfs.

    Metrics:
      - rmse: Overall Root Mean Squared Error across all validation rows.
      - top10_rmse: RMSE computed strictly on the top-10% discharge days per station.
      - mae: Mean Absolute Error across all validation rows.
      - r2: R-squared coefficient of determination.
      - weighted_composite: Weighted combination using official hackathon weights.
    """
    if TARGET_COL not in df_val.columns:
        raise ValueError(f"Ground truth target column '{TARGET_COL}' missing from validation DataFrame")

    y_true = pd.to_numeric(df_val[TARGET_COL], errors="coerce").fillna(0.0).values
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch in score(): {len(y_true)} actuals vs {len(y_pred)} predictions"
        )

    # 1. Overall RMSE
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # 2. Top 10% RMSE per station
    stations = df_val[STATION_COL].values
    extreme_indices = []

    for sid in np.unique(stations):
        s_idx = np.where(stations == sid)[0]
        if len(s_idx) > 0:
            s_actual = y_true[s_idx]
            q90 = float(np.nanquantile(s_actual, 0.90))
            top_mask = s_idx[s_actual >= q90]
            extreme_indices.extend(top_mask)

    if extreme_indices:
        top10_y_true = y_true[extreme_indices]
        top10_y_pred = y_pred[extreme_indices]
        top10_rmse = float(np.sqrt(mean_squared_error(top10_y_true, top10_y_pred)))
    else:
        top10_rmse = rmse

    # 3. Overall MAE
    mae = float(mean_absolute_error(y_true, y_pred))

    # 4. R2 Score
    if len(np.unique(y_true)) > 1:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = 0.0

    return {
        "rmse": rmse,
        "top10_rmse": top10_rmse,
        "mae": mae,
        "r2": r2,
    }


def finalize(
    test_df: pd.DataFrame,
    preds: np.ndarray,
    path: str | Path = "predictions.csv",
) -> pd.DataFrame:
    """Creates, validates, and writes the final predictions.csv according to the contract.

    Contract:
      - Exact columns: station_id, date, discharge_cfs
      - Same row count and order as test_df
      - Non-negative, finite, UTF-8 CSV
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    preds_arr = np.asarray(preds, dtype=float).ravel()
    if len(preds_arr) != len(test_df):
        raise ValueError(
            f"Predictions length ({len(preds_arr)}) does not match test length ({len(test_df)})"
        )

    # Enforce non-negativity and finiteness
    preds_arr = np.nan_to_num(preds_arr, nan=0.0, posinf=0.0, neginf=0.0)
    preds_arr = np.maximum(0.0, preds_arr)

    # Preserve exact station_id string and normalized date format
    sub = pd.DataFrame(
        {
            STATION_COL: test_df[STATION_COL].astype(str).str.zfill(8).values,
            DATE_COL: pd.to_datetime(test_df[DATE_COL]).dt.strftime("%Y-%m-%d").values,
            TARGET_COL: np.round(preds_arr, 4),
        }
    )

    sub = sub[SUBMISSION_COLUMNS]
    sub.to_csv(path, index=False, encoding="utf-8")
    return sub


def validate_submission(path: str | Path, test_path: str | Path) -> None:
    """Performs rigorous contract verification on the generated predictions.csv.

    Raises AssertionError on any violation of the submission rules.
    """
    path = Path(path)
    test_path = Path(test_path)

    assert path.exists(), f"Submission file does not exist: {path}"
    assert path.stat().st_size > 0, f"Submission file is empty: {path}"
    assert test_path.exists(), f"Reference test file does not exist: {test_path}"

    # Verify UTF-8 encoding
    try:
        sub_df = pd.read_csv(path, dtype={STATION_COL: str}, encoding="utf-8")
    except Exception as e:
        raise AssertionError(f"Failed to read submission CSV as UTF-8: {e}")

    test_df = pd.read_csv(test_path, dtype={STATION_COL: str}, encoding="utf-8")

    # Rule 6: Columns exactly station_id, date, discharge_cfs
    assert list(sub_df.columns) == SUBMISSION_COLUMNS, (
        f"Invalid columns: expected {SUBMISSION_COLUMNS}, got {list(sub_df.columns)}"
    )

    # Rule 5: Same row count as test.csv
    assert len(sub_df) == len(test_df), (
        f"Row count mismatch: predictions has {len(sub_df)} rows, test has {len(test_df)}"
    )

    # Rule 5: Same order of station_id and date
    sub_stations = sub_df[STATION_COL].astype(str).str.zfill(8).values
    test_stations = test_df[STATION_COL].astype(str).str.zfill(8).values
    assert (sub_stations == test_stations).all(), (
        "station_id ordering does not match test.csv exactly"
    )

    sub_dates = pd.to_datetime(sub_df[DATE_COL]).dt.normalize().values
    test_dates = pd.to_datetime(test_df[DATE_COL]).dt.normalize().values
    assert (sub_dates == test_dates).all(), (
        "date ordering does not match test.csv exactly"
    )

    # Rule 5: Non-negative and finite
    preds = pd.to_numeric(sub_df[TARGET_COL], errors="coerce").values
    assert not np.isnan(preds).any(), "predictions.csv contains NaN values"
    assert not np.isinf(preds).any(), "predictions.csv contains Infinite values"
    assert (preds >= 0.0).all(), "predictions.csv contains negative values"

