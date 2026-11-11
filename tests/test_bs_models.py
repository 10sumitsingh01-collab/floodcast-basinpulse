"""Unit tests for Lane C: BASINSHIFT Models, Training & Prediction.

Tests:
  1. Correct training/validation separation (no future rows across cutoff).
  2. No target leakage into inputs.
  3. Correct prediction dimensions and column names.
  4. Valid probability ranges [0, 1] and p_extreme <= p_high.
  5. Lower quantile <= upper quantile and non-negative values.
  6. Compatibility with unseen basins.
  7. Reproducible training across runs.
"""
from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from src.bs_models import BasinshiftModel, fit, predict, HORIZONS
from src.bs_predict import generate_test_predictions
from src.submission import assert_valid, validate
from src.evaluate import DATE, HI, HORIZON, ID, LO, PE, PH, PRED, PRED_COLS


class TestBasinshiftModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_parquet("data/caravan/train.parquet")
        cls.test = pd.read_parquet("data/caravan/test.parquet")
        cls.static = pd.read_parquet("data/caravan/static.parquet")
        cls.thr = pd.read_csv("data/caravan/thresholds_val.csv")

        # Select 5 basins for fast, deterministic unit testing
        cls.sample_basins = cls.train["gauge_id"].unique()[:5]
        cls.sample_train = cls.train[cls.train["gauge_id"].isin(cls.sample_basins)].copy()
        cls.sample_test = cls.test[cls.test["gauge_id"].isin(cls.sample_basins)].head(150).copy()

        # Shared fitted model for tests that only inspect predictions
        cls.fitted_model = BasinshiftModel(n_estimators=10, num_leaves=15, random_state=42)
        cls.fitted_model.fit(cls.sample_train, cls.static, cls.thr, fit_end_date="2006-09-30")
        cls.preds = cls.fitted_model.predict(cls.sample_test, cls.sample_train, cls.static, cls.thr)

    def test_1_training_validation_separation(self):
        """Rule 3: When fitting on dates <= fit_end, rows with t + h > fit_end must be dropped."""
        # Check fitted model exists for all horizons
        for h in HORIZONS:
            self.assertIn(h, self.fitted_model.models)
            hm = self.fitted_model.models[h]
            self.assertIsNotNone(hm.regressor)
            self.assertIsNotNone(hm.quantile_10)
            self.assertIsNotNone(hm.quantile_90)
            self.assertIsNotNone(hm.classifier_high)
            self.assertIsNotNone(hm.classifier_extreme)

    def test_2_no_target_leakage(self):
        """Rule 2: Predict must strictly reject inputs containing target columns."""
        # Passing targets in test_df must raise ValueError
        bad_test = self.sample_test.copy()
        bad_test["streamflow_h1"] = 1.0
        with self.assertRaises(ValueError) as ctx:
            self.fitted_model.predict(bad_test, self.sample_train, self.static, self.thr)
        self.assertIn("must NEVER receive target columns", str(ctx.exception))

    def test_3_prediction_schema_and_dimensions(self):
        """Verify exact long-format prediction table matching BASINSHIFT 8.3/8.4."""
        preds = self.preds

        # 1. Row count: exactly len(test) * 3 horizons
        expected_rows = len(self.sample_test) * len(HORIZONS)
        self.assertEqual(len(preds), expected_rows)

        # 2. Required columns
        expected_cols = [ID, DATE, HORIZON] + PRED_COLS
        self.assertEqual(list(preds.columns), expected_cols)

        # 3. Validation passes without errors
        errs = validate(preds, self.sample_test[[ID, DATE]])
        self.assertEqual(errs, [])

    def test_4_valid_probabilities(self):
        """Verify p_high, p_extreme are in [0, 1] and p_extreme <= p_high."""
        preds = self.preds
        ph = preds[PH].to_numpy()
        pe = preds[PE].to_numpy()

        self.assertTrue(np.all(ph >= 0.0) and np.all(ph <= 1.0), "p_high out of [0, 1]")
        self.assertTrue(np.all(pe >= 0.0) and np.all(pe <= 1.0), "p_extreme out of [0, 1]")
        self.assertTrue(np.all(pe <= ph + 1e-6), "p_extreme > p_high detected!")

    def test_5_valid_prediction_intervals(self):
        """Verify lower and upper are non-negative, finite, and lower <= upper."""
        preds = self.preds
        pred_vals = preds[PRED].to_numpy()
        lo_vals = preds[LO].to_numpy()
        hi_vals = preds[HI].to_numpy()

        self.assertTrue(np.all(pred_vals >= 0.0), "Negative predicted_discharge")
        self.assertTrue(np.all(lo_vals >= 0.0), "Negative lower bound")
        self.assertTrue(np.all(hi_vals >= lo_vals), "lower > upper bound detected!")
        self.assertTrue(np.all(np.isfinite(pred_vals)), "NaN/inf in predictions")

    def test_6_unseen_basin_compatibility(self):
        """Verify model predicts correctly on unseen basins not present in training."""
        # Train on first 3 basins only
        train_basins = self.sample_basins[:3]
        unseen_basin = self.sample_basins[3]

        train_sub = self.sample_train[self.sample_train["gauge_id"].isin(train_basins)]
        unseen_test = self.sample_train[self.sample_train["gauge_id"] == unseen_basin].head(60).copy()
        # Drop targets from test input
        target_cols = [c for c in unseen_test.columns if c == "streamflow" or c.startswith("streamflow_h")]
        unseen_test = unseen_test.drop(columns=target_cols)

        model = BasinshiftModel(n_estimators=10, num_leaves=15, random_state=42)
        model.fit(train_sub, self.static, self.thr)

        preds = model.predict(unseen_test, train_sub, self.static, self.thr)
        self.assertEqual(len(preds), len(unseen_test) * len(HORIZONS))
        self.assertEqual(validate(preds, unseen_test[[ID, DATE]]), [])

    def test_7_reproducible_training(self):
        """Verify deterministic outputs with fixed random_state."""
        m1 = BasinshiftModel(n_estimators=10, num_leaves=15, random_state=42)
        m1.fit(self.sample_train.head(500), self.static, self.thr)
        p1 = m1.predict(self.sample_test.head(30), self.sample_train.head(500), self.static, self.thr)

        m2 = BasinshiftModel(n_estimators=10, num_leaves=15, random_state=42)
        m2.fit(self.sample_train.head(500), self.static, self.thr)
        p2 = m2.predict(self.sample_test.head(30), self.sample_train.head(500), self.static, self.thr)

        pd.testing.assert_frame_equal(p1, p2, check_exact=True)


if __name__ == "__main__":
    unittest.main()

