"""Unit tests for Lane A: src/data.py"""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    DATE_COL,
    STATION_COL,
    STATION_METADATA_COLUMNS,
    TARGET_COL,
    TEST_COLUMNS,
    TRAIN_COLUMNS,
)
from src.data import load_station_metadata, load_test, load_train


def test_load_train_schema_and_types():
    train = load_train()
    assert isinstance(train, pd.DataFrame)
    assert not train.empty

    # Columns match contract exactly
    assert list(train.columns) == TRAIN_COLUMNS

    # Rule 3: station_id is string
    assert pd.api.types.is_string_dtype(train[STATION_COL])
    assert all(isinstance(val, str) for val in train[STATION_COL].iloc[:20])

    # Rule 4: date is datetime64[ns] normalized to midnight
    assert train[DATE_COL].dtype == "datetime64[ns]"
    assert (train[DATE_COL].dt.hour == 0).all()
    assert (train[DATE_COL].dt.minute == 0).all()

    # Target is present and numeric
    assert TARGET_COL in train.columns
    assert np.issubdtype(train[TARGET_COL].dtype, np.floating)
    assert (train[TARGET_COL] >= 0).all()


def test_load_test_schema_and_types():
    test = load_test()
    assert isinstance(test, pd.DataFrame)
    assert not test.empty

    # Columns match contract exactly (same minus discharge_cfs)
    assert list(test.columns) == TEST_COLUMNS
    assert TARGET_COL not in test.columns

    # Rule 3: station_id is string
    assert pd.api.types.is_string_dtype(test[STATION_COL])

    # Rule 4: date is datetime64[ns] normalized to midnight
    assert test[DATE_COL].dtype == "datetime64[ns]"
    assert (test[DATE_COL].dt.hour == 0).all()


def test_load_station_metadata():
    meta = load_station_metadata()
    assert isinstance(meta, pd.DataFrame)
    assert list(meta.columns) == STATION_METADATA_COLUMNS
    assert pd.api.types.is_string_dtype(meta[STATION_COL])
