"""Unit tests for Lane C: src/models.py"""

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_COL
from src.data import load_test, load_train
from src.features import FEATURE_COLS, build_features
from src.models import fit_models, make_weights, predict


def test_make_weights():
    train = load_train()
    weights = make_weights(train, q=0.90, boost=5.0)

    assert isinstance(weights, np.ndarray)
    assert len(weights) == len(train)
    assert (weights >= 1.0).all()

    # Boosted rows should exist
    boosted_count = (weights == 5.0).sum()
    assert boosted_count > 0, "No rows were boosted for top 10% events"

    # Approximately 10% of rows per station should be boosted
    ratio = boosted_count / len(train)
    assert 0.05 <= ratio <= 0.15, f"Unexpected ratio of boosted rows: {ratio}"


def test_fit_and_predict_interface():
    train = load_train()
    test = load_test()

    train_feats = build_features(train)
    test_feats = build_features(test)

    X_train = train_feats[FEATURE_COLS]
    y_train = train[TARGET_COL]

    # Fast fit with small trees for unit testing
    params = {"n_estimators": 10, "learning_rate": 0.1}
    bundle = fit_models(X_train, y_train, train, params=params)

    assert "lgb_raw" in bundle
    assert "xgb_raw" in bundle
    assert "lgb_log" in bundle
    assert "smearing_factor" in bundle
    assert bundle["smearing_factor"] > 0

    X_test = test_feats[FEATURE_COLS]
    preds = predict(bundle, X_test, test)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(test)

    # Rule 5: Non-negative, finite, no NaNs
    assert not np.isnan(preds).any()
    assert not np.isinf(preds).any()
    assert (preds >= 0.0).all()
