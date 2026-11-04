"""Data loading, ingestion, and validation module (Lane A).

Interfaces:
  - load_train() -> pd.DataFrame
  - load_test() -> pd.DataFrame
  - load_station_metadata() -> pd.DataFrame
  - last_train_date() -> pd.Timestamp
  - test_start_date() -> pd.Timestamp
  - missing_report(df: pd.DataFrame) -> pd.DataFrame

Hard rules enforced:
  - Real data only. No synthetic/generator code anywhere.
  - station_id is ALWAYS a string (dtype={'station_id': str}).
  - date is ALWAYS pandas datetime64[ns], normalized to midnight.
  - Strict assertions: no duplicates, no negative discharge, test has no discharge,
    train max date < test min date.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from src.config import (
    DATE_COL,
    HOLDOUT_LABELS_PATH,
    PROCESSED_DIR,
    STATION_COL,
    STATION_METADATA_COLUMNS,
    STATION_METADATA_PARQUET_PATH,
    STATION_METADATA_PATH,
    TARGET_COL,
    TEST_COLUMNS,
    TEST_PARQUET_PATH,
    TEST_PATH,
    TRAIN_COLUMNS,
    TRAIN_PARQUET_PATH,
    TRAIN_PATH,
)


def _enforce_contract_types(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    """Enforce strict dtypes, string station_id, and datetime64[ns] normalized to midnight."""
    df = df.copy()

    # Rule 3: station_id is string
    df[STATION_COL] = df[STATION_COL].astype(str).str.strip().str.zfill(8)

    # Rule 4: date is datetime64[ns] normalized to midnight
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.normalize().astype("datetime64[ns]")

    # Numeric columns
    num_cols = [
        "precipitation_mm",
        "temp_max_c",
        "temp_min_c",
        "soil_moisture",
        "drainage_area_sqmi",
        "mean_elevation_ft",
        "slope_pct",
        "forest_cover_pct",
        "elev_range_ft",
        "elev_std_ft",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    if not is_test and TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").astype(float)

    return df


def last_train_date() -> pd.Timestamp:
    """Return the maximum date present in the training set."""
    train = load_train()
    return pd.to_datetime(train[DATE_COL].max()).normalize()


def test_start_date() -> pd.Timestamp:
    """Return the minimum date present in the test set."""
    test = load_test()
    return pd.to_datetime(test[DATE_COL].min()).normalize()


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate missingness report across all columns."""
    total = len(df)
    missing_cnt = df.isna().sum()
    pct = (missing_cnt / total) * 100.0 if total > 0 else 0.0

    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_cnt.values,
        "missing_pct": np.round(pct.values, 2),
        "dtype": [str(df[c].dtype) for c in df.columns],
    })
    return report


def load_train() -> pd.DataFrame:
    """Load training data (real data only).

    Columns: station_id, date, discharge_cfs, precipitation_mm,
             temp_max_c, temp_min_c, soil_moisture, drainage_area_sqmi,
             mean_elevation_ft, slope_pct, forest_cover_pct
    """
    if TRAIN_PARQUET_PATH.exists():
        df = pd.read_parquet(TRAIN_PARQUET_PATH)
    elif TRAIN_PATH.exists():
        df = pd.read_csv(TRAIN_PATH, dtype={STATION_COL: str})
    else:
        raise FileNotFoundError(
            f"Real training data not found at {TRAIN_PARQUET_PATH} or {TRAIN_PATH}. "
            f"Run `python -m src.build_dataset --stations all` first."
        )

    df = _enforce_contract_types(df, is_test=False)

    # Assertions
    missing = [c for c in TRAIN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"train data missing required columns: {missing}")

    assert not df.duplicated(subset=[STATION_COL, DATE_COL]).any(), "Duplicate (station_id, date) in train!"
    assert pd.api.types.is_string_dtype(df[STATION_COL]), "station_id must be string in train!"
    assert (df[TARGET_COL] >= 0.0).all(), "Negative discharge in train!"

    return df[TRAIN_COLUMNS]


def load_test() -> pd.DataFrame:
    """Load test features (same columns minus discharge_cfs)."""
    if TEST_PARQUET_PATH.exists():
        df = pd.read_parquet(TEST_PARQUET_PATH)
    elif TEST_PATH.exists():
        df = pd.read_csv(TEST_PATH, dtype={STATION_COL: str})
    else:
        raise FileNotFoundError(
            f"Real test data not found at {TEST_PARQUET_PATH} or {TEST_PATH}. "
            f"Run `python -m src.build_dataset --stations all` first."
        )

    df = _enforce_contract_types(df, is_test=True)

    # Rule 2: NEVER include discharge in test features
    if TARGET_COL in df.columns:
        raise AssertionError("discharge_cfs leaked into test features!")

    missing = [c for c in TEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"test data missing required columns: {missing}")

    assert not df.duplicated(subset=[STATION_COL, DATE_COL]).any(), "Duplicate (station_id, date) in test!"
    assert pd.api.types.is_string_dtype(df[STATION_COL]), "station_id must be string in test!"

    # Verify temporal separation against train
    if TRAIN_PARQUET_PATH.exists() or TRAIN_PATH.exists():
        train = load_train()
        assert train[DATE_COL].max() < df[DATE_COL].min(), (
            f"Temporal leakage: train max date ({train[DATE_COL].max()}) "
            f">= test min date ({df[DATE_COL].min()})"
        )

    # Verify holdout keys match test keys exactly
    if HOLDOUT_LABELS_PATH.exists():
        holdout = pd.read_parquet(HOLDOUT_LABELS_PATH)
        assert len(df) == len(holdout), "Test rows != Holdout labels rows!"
        assert (df[STATION_COL].values == holdout[STATION_COL].values).all(), "Test and holdout stations do not match!"
        assert (df[DATE_COL].values == holdout[DATE_COL].values).all(), "Test and holdout dates do not match!"

    return df[TEST_COLUMNS]


def load_station_metadata() -> pd.DataFrame:
    """Load station metadata (station_id, station_name, latitude, longitude, state, river_name)."""
    if STATION_METADATA_PARQUET_PATH.exists():
        df = pd.read_parquet(STATION_METADATA_PARQUET_PATH)
    elif STATION_METADATA_PATH.exists():
        df = pd.read_csv(STATION_METADATA_PATH, dtype={STATION_COL: str})
    else:
        raise FileNotFoundError(
            f"Station metadata not found at {STATION_METADATA_PARQUET_PATH} or {STATION_METADATA_PATH}."
        )

    df[STATION_COL] = df[STATION_COL].astype(str).str.strip().str.zfill(8)

    missing = [c for c in STATION_METADATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"station_metadata missing columns: {missing}")

    return df[STATION_METADATA_COLUMNS]
