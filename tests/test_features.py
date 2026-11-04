"""Unit tests for Lane B: src/features.py"""

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_COL
from src.data import load_test, load_train
from src.features import FEATURE_COLS, build_features


def test_build_features_interface_and_presence():
    train = load_train()
    feats = build_features(train)

    assert isinstance(feats, pd.DataFrame)
    assert len(feats) == len(train)
    assert (feats.index == train.index).all()

    # Every column named in FEATURE_COLS exists
    for col in FEATURE_COLS:
        assert col in feats.columns, f"Missing feature {col}"
        # All feature values must be finite numbers
        assert not np.isnan(feats[col]).any(), f"Feature {col} contains NaN"
        assert not np.isinf(feats[col]).any(), f"Feature {col} contains Inf"


def test_build_features_no_target_leakage():
    """Rule 1 & 2: discharge_cfs must not be used or leak into features."""
    train = load_train()

    # Case A: normal target
    feats_a = build_features(train)[FEATURE_COLS]

    # Case B: target corrupted / all zeros
    train_b = train.copy()
    train_b[TARGET_COL] = 0.0
    feats_b = build_features(train_b)[FEATURE_COLS]

    # Case C: target set to NaN
    train_c = train.copy()
    train_c[TARGET_COL] = np.nan
    feats_c = build_features(train_c)[FEATURE_COLS]

    # Case D: target removed entirely (as in test.csv)
    train_d = train.drop(columns=[TARGET_COL])
    feats_d = build_features(train_d)[FEATURE_COLS]

    # Features must be byte-for-byte or numerically identical
    pd.testing.assert_frame_equal(feats_a, feats_b)
    pd.testing.assert_frame_equal(feats_a, feats_c)
    pd.testing.assert_frame_equal(feats_a, feats_d)


def test_build_features_row_order_preservation():
    """Verify that scrambled row indices / order are faithfully preserved."""
    train = load_train().sample(frac=0.3, random_state=42)
    feats = build_features(train)

    assert len(feats) == len(train)
    pd.testing.assert_index_equal(feats.index, train.index)
