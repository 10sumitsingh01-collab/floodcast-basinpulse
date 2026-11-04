"""Smoke test suite (Lane D).

Runs fast end-to-end sanity checks across all lanes:
  - Lane A: data loading and contract schemas
  - Lane B: feature engineering and leakage immunity
  - Lane C: sample weighting and model ensemble fitting
  - Lane D: validation folds, scoring, finalize, and submission verification

Can be run directly via:
    python tests/smoke.py
or via pytest:
    pytest tests/smoke.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATE_COL,
    RANDOM_STATE,
    STATION_COL,
    SUBMISSION_COLUMNS,
    TARGET_COL,
    TEST_COLUMNS,
    TEST_PATH,
    TRAIN_COLUMNS,
)
from src.data import load_station_metadata, load_test, load_train
from src.features import FEATURE_COLS, build_features
from src.models import fit_models, make_weights, predict
from src.validate import finalize, forward_folds, loso_folds, score, validate_submission


def smoke_test_data_lane():
    print("[1/5] Checking Lane A: Data loading and contracts...")
    train = load_train()
    test = load_test()
    meta = load_station_metadata()

    assert not train.empty, "train DataFrame is empty"
    assert not test.empty, "test DataFrame is empty"
    assert not meta.empty, "station metadata is empty"

    assert list(train.columns) == TRAIN_COLUMNS, "train columns do not match contract"
    assert list(test.columns) == TEST_COLUMNS, "test columns do not match contract"

    # Rule 3: station_id is string
    assert pd.api.types.is_string_dtype(train[STATION_COL]), "station_id must be string dtype in train"
    assert pd.api.types.is_string_dtype(test[STATION_COL]), "station_id must be string dtype in test"

    # Rule 4: date is datetime64[ns] normalized to midnight
    assert train[DATE_COL].dtype == "datetime64[ns]", f"train date must be datetime64[ns], got {train[DATE_COL].dtype}"
    assert test[DATE_COL].dtype == "datetime64[ns]", f"test date must be datetime64[ns], got {test[DATE_COL].dtype}"
    assert (train[DATE_COL].dt.hour == 0).all(), "train date not normalized to midnight"
    assert (test[DATE_COL].dt.hour == 0).all(), "test date not normalized to midnight"

    # Rule 2: No discharge in test
    assert TARGET_COL not in test.columns, "test.csv must NOT have discharge_cfs"
    print("      -> Lane A verified!")


def smoke_test_features_lane():
    print("[2/5] Checking Lane B: Feature engineering and leakage immunity...")
    train = load_train().head(200)
    test = load_test().head(50)

    combined = pd.concat([train, test], ignore_index=True)
    feats = build_features(combined)

    assert len(feats) == len(combined), "Row count changed during feature engineering"
    for col in FEATURE_COLS:
        assert col in feats.columns, f"Feature column '{col}' missing from engineered DataFrame"
        assert not np.isnan(feats[col]).any(), f"Feature column '{col}' contains NaN"

    # Target leakage verification
    c1 = combined.copy()
    c2 = combined.copy()
    c2[TARGET_COL] = np.nan
    f1 = build_features(c1)[FEATURE_COLS]
    f2 = build_features(c2)[FEATURE_COLS]
    diff = float((f1 - f2).abs().max().max())
    assert diff == 0.0, f"Features depend on target variable! Diff: {diff}"
    print("      -> Lane B verified!")


def smoke_test_models_lane():
    print("[3/5] Checking Lane C: Extreme event weighting and ensemble models...")
    train = load_train().head(300)
    weights = make_weights(train, q=0.90, boost=5.0)

    assert len(weights) == len(train), "Weight array length mismatch"
    assert (weights >= 1.0).all(), "Weights must be >= 1.0"

    feats = build_features(train)
    X = feats[FEATURE_COLS]
    y = train[TARGET_COL]

    bundle = fit_models(X, y, train, params={"n_estimators": 5, "learning_rate": 0.1})
    preds = predict(bundle, X, train)

    assert len(preds) == len(train), "Predictions length mismatch"
    assert not np.isnan(preds).any(), "Predictions contain NaN"
    assert not np.isinf(preds).any(), "Predictions contain Inf"
    assert (preds >= 0.0).all(), "Predictions contain negative values"
    print("      -> Lane C verified!")


def smoke_test_validate_lane():
    print("[4/5] Checking Lane D: CV folds, scoring, finalize and validation...")
    train = load_train()

    # Test forward folds
    years = sorted(train[DATE_COL].dt.year.unique())
    if len(years) > 1:
        folds = list(forward_folds(train, val_years=[years[-1]], gap_days=10))
        assert len(folds) >= 1, "Failed to generate forward fold"

    # Test LOSO folds
    loso = list(loso_folds(train))
    assert len(loso) >= 1, "Failed to generate LOSO fold"

    # Test scoring
    metrics = score(train, train[TARGET_COL].values)
    for m in ["rmse", "top10_rmse", "mae", "r2"]:
        assert m in metrics, f"Metric '{m}' missing from score output"
        assert not np.isnan(metrics[m]), f"Metric '{m}' is NaN"

    print("      -> Lane D verified!")


def smoke_test_end_to_end_pipeline():
    print("[5/5] Running mini end-to-end pipeline test...")
    train = load_train().head(200)
    test = load_test().head(40)

    train_len = len(train)
    combined = pd.concat([train, test], ignore_index=True)
    all_feats = build_features(combined)

    X_train = all_feats.iloc[:train_len][FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_test = all_feats.iloc[train_len:][FEATURE_COLS]

    bundle = fit_models(X_train, y_train, train, params={"n_estimators": 10, "learning_rate": 0.1})
    preds = predict(bundle, X_test, test)

    tmp_sub = Path("experiments") / "smoke_predictions.csv"
    tmp_test = Path("experiments") / "smoke_test.csv"
    tmp_sub.parent.mkdir(parents=True, exist_ok=True)

    test.to_csv(tmp_test, index=False)
    finalize(test, preds, path=tmp_sub)
    validate_submission(tmp_sub, tmp_test)

    # Clean up smoke files
    if tmp_sub.exists():
        tmp_sub.unlink()
    if tmp_test.exists():
        tmp_test.unlink()

    print("      -> End-to-end pipeline passed successfully!")


def main():
    print("=" * 60)
    print("FLOODCAST SMOKE TESTS: RUNNING ALL LANES")
    print("=" * 60)
    smoke_test_data_lane()
    smoke_test_features_lane()
    smoke_test_models_lane()
    smoke_test_validate_lane()
    smoke_test_end_to_end_pipeline()
    print("=" * 60)
    print("ALL SMOKE TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
