"""Unit tests for Lane D: src/validate.py"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import DATE_COL, STATION_COL, SUBMISSION_COLUMNS, TARGET_COL
from src.data import load_test, load_train
from src.validate import finalize, forward_folds, loso_folds, score, validate_submission


def test_forward_folds():
    train = load_train()
    years = [2021, 2022]
    folds = list(forward_folds(train, val_years=years, gap_days=30))

    assert len(folds) > 0
    for train_idx, val_idx in folds:
        assert len(train_idx) > 0
        assert len(val_idx) > 0

        # Disjoint indices
        assert len(set(train_idx).intersection(set(val_idx))) == 0

        # Purging gap: train max date must be <= val min date - 30 days
        train_max = train.iloc[train_idx][DATE_COL].max()
        val_min = train.iloc[val_idx][DATE_COL].min()
        assert train_max <= val_min - pd.Timedelta(days=30)


def test_loso_folds():
    train = load_train()
    folds = list(loso_folds(train))

    unique_stations = train[STATION_COL].nunique()
    assert len(folds) == unique_stations

    for train_idx, val_idx in folds:
        train_stations = set(train.iloc[train_idx][STATION_COL].unique())
        val_stations = set(train.iloc[val_idx][STATION_COL].unique())

        assert len(val_stations) == 1
        assert len(train_stations.intersection(val_stations)) == 0


def test_score_metrics():
    train = load_train().iloc[:100].copy()
    y_true = train[TARGET_COL].values
    # Perfect predictions
    metrics_perf = score(train, y_true)
    assert metrics_perf["rmse"] == 0.0
    assert metrics_perf["mae"] == 0.0
    assert np.isclose(metrics_perf["r2"], 1.0)
    assert metrics_perf["top10_rmse"] == 0.0

    # Perturbed predictions
    noisy = y_true + 10.0
    metrics_noisy = score(train, noisy)
    assert metrics_noisy["rmse"] > 0.0
    assert metrics_noisy["mae"] > 0.0
    assert metrics_noisy["top10_rmse"] > 0.0


def test_finalize_and_validate_submission(tmp_path):
    test = load_test()
    preds = np.random.uniform(50.0, 500.0, len(test))

    out_file = tmp_path / "predictions.csv"
    test_file = tmp_path / "test.csv"
    test.to_csv(test_file, index=False)

    final_df = finalize(test, preds, path=out_file)
    assert isinstance(final_df, pd.DataFrame)
    assert list(final_df.columns) == SUBMISSION_COLUMNS
    assert len(final_df) == len(test)

    # Validate against reference test file
    validate_submission(out_file, test_file)

    # Test failure detection: negative prediction
    bad_preds = preds.copy()
    bad_preds[0] = -1.0
    bad_file = tmp_path / "bad_preds.csv"
    final_df_bad = pd.DataFrame(
        {
            STATION_COL: test[STATION_COL].values,
            DATE_COL: test[DATE_COL].values,
            TARGET_COL: bad_preds,
        }
    )
    final_df_bad.to_csv(bad_file, index=False)

    with pytest.raises(AssertionError):
        validate_submission(bad_file, test_file)
