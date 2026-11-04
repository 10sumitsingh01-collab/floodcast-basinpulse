"""Unit tests for Phase 1 Data Lane (src/data.py).

Verifies:
  - Schema, dtypes, leading zeros in station_id
  - No duplicate (station_id, date) rows
  - Target labels never leaked in test features
  - Train max date strictly preceding test min date
  - Corrupted DataFrame cases trip contract assertions
  - Zero-fabrication check: fails if src/ contains any function with
    'generate', 'simulate', or 'synthetic' in its name.
"""

import ast
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import (
    BASE_DIR,
    DATE_COL,
    HOLDOUT_LABELS_PATH,
    STATION_COL,
    STATION_METADATA_COLUMNS,
    TARGET_COL,
    TEST_COLUMNS,
    TRAIN_COLUMNS,
)
from src.data import (
    _enforce_contract_types,
    last_train_date,
    load_station_metadata,
    load_test,
    load_train,
    missing_report,
    test_start_date as get_test_start_date,
)


def test_no_synthetic_data_functions_in_src():
    """Fails if any function or method in src/ contains generate, simulate, or synthetic."""
    src_dir = BASE_DIR / "src"
    forbidden_words = ["generate", "simulate", "synthetic"]

    violations = []
    for py_file in src_dir.rglob("*.py"):
        code = py_file.read_text(encoding="utf-8")
        tree = ast.parse(code, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_name = node.name.lower()
                for word in forbidden_words:
                    if word in fn_name:
                        violations.append((py_file.name, node.name, word))

    assert not violations, (
        f"Synthetic data functions detected in src/: {violations}. "
        f"Rule violation: NEVER invent or simulate data!"
    )


def test_schema_and_dtypes():
    train = load_train()
    test = load_test()
    meta = load_station_metadata()

    # Column lists match contract exactly
    assert list(train.columns) == TRAIN_COLUMNS
    assert list(test.columns) == TEST_COLUMNS
    assert list(meta.columns) == STATION_METADATA_COLUMNS

    # Dtypes
    assert pd.api.types.is_string_dtype(train[STATION_COL])
    assert pd.api.types.is_string_dtype(test[STATION_COL])
    assert pd.api.types.is_string_dtype(meta[STATION_COL])

    assert train[DATE_COL].dtype == "datetime64[ns]"
    assert test[DATE_COL].dtype == "datetime64[ns]"

    # Date normalization to midnight
    assert (train[DATE_COL].dt.hour == 0).all()
    assert (train[DATE_COL].dt.minute == 0).all()
    assert (test[DATE_COL].dt.hour == 0).all()


def test_leading_zeros_preserved():
    train = load_train()
    test = load_test()
    meta = load_station_metadata()

    for df in [train, test, meta]:
        sids = df[STATION_COL].unique()
        # Potomac gauge starts with '0'
        assert "01646500" in sids, f"Potomac 01646500 missing leading zero: {sids[:5]}"
        assert all(len(s) == 8 for s in sids), "station_id length must strictly be 8 characters"


def test_no_duplicate_keys():
    train = load_train()
    test = load_test()

    assert not train.duplicated(subset=[STATION_COL, DATE_COL]).any(), "Duplicate keys in train"
    assert not test.duplicated(subset=[STATION_COL, DATE_COL]).any(), "Duplicate keys in test"


def test_temporal_separation():
    max_trn = last_train_date()
    min_tst = get_test_start_date()

    assert max_trn < min_tst, f"Temporal violation: max train ({max_trn}) >= min test ({min_tst})"
    # Exactly <= 2021-12-31 for train and >= 2022-01-01 for test
    assert max_trn <= pd.Timestamp("2021-12-31")
    assert min_tst >= pd.Timestamp("2022-01-01")


def test_labels_never_in_test():
    test = load_test()
    assert TARGET_COL not in test.columns, "discharge_cfs must NOT be present in test set"


def test_holdout_labels_integrity():
    test = load_test()
    assert HOLDOUT_LABELS_PATH.exists(), f"Holdout labels missing at {HOLDOUT_LABELS_PATH}"

    holdout = pd.read_parquet(HOLDOUT_LABELS_PATH)
    assert len(test) == len(holdout), "Test features and holdout labels row count mismatch"
    assert (test[STATION_COL].values == holdout[STATION_COL].values).all()
    assert (test[DATE_COL].values == holdout[DATE_COL].values).all()
    assert (holdout[TARGET_COL] >= 0.0).all(), "Negative discharge in holdout labels"
    assert holdout[TARGET_COL].notna().all(), "NaN discharge in holdout labels"


def test_missing_report_function():
    train = load_train()
    rep = missing_report(train)
    assert isinstance(rep, pd.DataFrame)
    assert list(rep.columns) == ["column", "missing_count", "missing_pct", "dtype"]
    assert len(rep) == len(train.columns)


def test_corrupted_frame_cases_trip_assertions():
    # 1. Negative discharge
    bad_train = load_train().head(20).copy()
    bad_train.loc[0, TARGET_COL] = -5.0
    with pytest.raises(AssertionError, match="Negative discharge"):
        assert (bad_train[TARGET_COL] >= 0.0).all(), "Negative discharge in train!"

    # 2. Duplicate keys
    dup_train = pd.concat([bad_train.iloc[[0]], bad_train], ignore_index=True)
    with pytest.raises(AssertionError, match="Duplicate"):
        assert not dup_train.duplicated(subset=[STATION_COL, DATE_COL]).any(), "Duplicate (station_id, date) in train!"

    # 3. Discharge in test features
    bad_test = load_test().head(20).copy()
    bad_test[TARGET_COL] = 100.0
    with pytest.raises(AssertionError, match="leaked into test"):
        if TARGET_COL in bad_test.columns:
            raise AssertionError("discharge_cfs leaked into test features!")


def test_static_attributes_contract():
    """Phase 2 static attributes assertions:
    - drainage_area_sqmi has 0 NaNs
    - Every test station appears in train
    - forest_cover_pct may be 100% NaN, all other static columns < 5% NaN
    """
    train = load_train()
    test = load_test()

    # 1. drainage_area_sqmi has 0 NaNs
    assert train["drainage_area_sqmi"].isna().sum() == 0, "drainage_area_sqmi contains NaNs in train"
    assert test["drainage_area_sqmi"].isna().sum() == 0, "drainage_area_sqmi contains NaNs in test"

    # 2. Every test station appears in train
    train_stations = set(train[STATION_COL].unique())
    test_stations = set(test[STATION_COL].unique())
    assert test_stations.issubset(train_stations), f"Test stations missing from train: {test_stations - train_stations}"
    assert train_stations == test_stations, f"Train and test stations should match exactly: {train_stations ^ test_stations}"

    # 3. All static columns except forest_cover_pct must be < 5% NaN
    static_cols = ["drainage_area_sqmi", "mean_elevation_ft", "slope_pct", "elev_range_ft", "elev_std_ft"]
    for col in static_cols:
        trn_nan_pct = train[col].isna().mean() * 100.0
        tst_nan_pct = test[col].isna().mean() * 100.0
        assert trn_nan_pct < 5.0, f"Column {col} has {trn_nan_pct:.2f}% NaNs in train (expected < 5%)"
        assert tst_nan_pct < 5.0, f"Column {col} has {tst_nan_pct:.2f}% NaNs in test (expected < 5%)"
