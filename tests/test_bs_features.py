"""Unit tests for Lane B: BASINSHIFT Feature Engineering (src/bs_features.py).

Tests on real Caravan v1.6 / CAMELS-US data:
  1. Truncation: features for dates <= cut are identical whether or not later rows exist.
  2. build_features raises if streamflow or streamflow_h* columns are passed in.
  3. No FEATURE_COLS column equals a streamflow_h* column on train rows.
  4. Row count and order preserved; gauge_id stays a string; output float32.
  5. Runtime on full train + test printed (must finish comfortably on a 7 GB laptop).
"""
from __future__ import annotations

import time
import unittest
import numpy as np
import pandas as pd

from src.bs_features import (
    FEATURE_COLS,
    INPUT_COLS,
    build_features,
)


class TestBasinshiftFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load real Caravan data
        cls.train = pd.read_parquet("data/caravan/train.parquet")
        cls.test = pd.read_parquet("data/caravan/test.parquet")
        cls.static = pd.read_parquet("data/caravan/static.parquet")

        # Select 3 real basins for detailed property testing
        cls.sample_basins = cls.train["gauge_id"].unique()[:3]
        cls.sample_train = cls.train[cls.train["gauge_id"].isin(cls.sample_basins)].copy()
        cls.sample_inputs = cls.sample_train[INPUT_COLS].copy()

    def test_1_truncation_causality(self):
        """Rule 1: Features for dates <= cut are identical whether or not later rows exist."""
        basin_id = self.sample_basins[0]
        basin_data = self.sample_inputs[self.sample_inputs["gauge_id"] == basin_id].sort_values("date")
        dates = pd.to_datetime(basin_data["date"]).values
        t_cut = dates[len(dates) // 2]

        # Full features
        feats_full = build_features(basin_data, self.static)

        # Truncated features: input only up to t_cut
        truncated_input = basin_data[basin_data["date"] <= t_cut].copy()
        feats_truncated = build_features(truncated_input, self.static)

        # Compare on dates <= t_cut
        mask_full = feats_full["date"] <= t_cut
        f_full_past = feats_full.loc[mask_full, FEATURE_COLS].reset_index(drop=True)
        f_trunc_past = feats_truncated[FEATURE_COLS].reset_index(drop=True)

        pd.testing.assert_frame_equal(
            f_trunc_past,
            f_full_past,
            check_exact=False,
            rtol=1e-5,
            atol=1e-5,
            obj="Causality verification (dates <= t_cut)",
        )

    def test_2_raises_on_target_columns(self):
        """Rule 2: build_features raises ValueError if streamflow or streamflow_h* columns are passed."""
        # Case A: streamflow target present
        bad_input_a = self.sample_train.copy()
        with self.assertRaises(ValueError) as ctx_a:
            build_features(bad_input_a, self.static)
        self.assertIn("streamflow", str(ctx_a.exception))

        # Case B: streamflow_h1 present
        bad_input_b = self.sample_inputs.copy()
        bad_input_b["streamflow_h1"] = 1.0
        with self.assertRaises(ValueError) as ctx_b:
            build_features(bad_input_b, self.static)
        self.assertIn("streamflow_h1", str(ctx_b.exception))

        # Case C: streamflow_h7 present
        bad_input_c = self.sample_inputs.copy()
        bad_input_c["streamflow_h7"] = 1.0
        with self.assertRaises(ValueError) as ctx_c:
            build_features(bad_input_c, self.static)
        self.assertIn("streamflow_h7", str(ctx_c.exception))

    def test_3_no_target_leakage_on_train(self):
        """Hard rule: No FEATURE_COLS column equals or leaks future streamflow_h* targets."""
        feats = build_features(self.sample_inputs, self.static)
        
        for h in [1, 3, 7]:
            target_col = f"streamflow_h{h}"
            target_vals = self.sample_train[target_col].to_numpy(dtype=float)
            ok = np.isfinite(target_vals)

            for col in FEATURE_COLS:
                feat_vals = feats[col].to_numpy(dtype=float)
                if not np.all(np.isnan(feat_vals[ok])):
                    # Check maximum absolute correlation or exact match
                    diff = np.nanmax(np.abs(feat_vals[ok] - target_vals[ok]))
                    self.assertGreater(
                        diff,
                        1e-4,
                        f"Feature '{col}' is suspiciously identical to target '{target_col}'!",
                    )

    def test_4_row_preservation_and_dtypes(self):
        """Verify row count and order preserved, gauge_id stays string, output float32."""
        # Shuffle inputs to test preservation of original order
        shuffled = self.sample_inputs.sample(frac=1.0, random_state=42)
        feats = build_features(shuffled, self.static)

        # 1. Exact row count & index
        self.assertEqual(len(feats), len(shuffled))
        pd.testing.assert_index_equal(feats.index, shuffled.index)

        # 2. Key columns exist and preserve original values
        self.assertTrue((feats["gauge_id"].values == shuffled["gauge_id"].values).all())
        self.assertTrue((feats["date"].values == shuffled["date"].values).all())

        # 3. gauge_id remains string
        self.assertTrue(
            isinstance(feats["gauge_id"].iloc[0], str),
            "gauge_id must be a string",
        )

        # 4. All feature columns are float32
        for col in FEATURE_COLS:
            self.assertIn(col, feats.columns, f"Missing feature column {col}")
            self.assertEqual(
                feats[col].dtype,
                np.float32,
                f"Feature column '{col}' has dtype {feats[col].dtype}, expected float32",
            )

    def test_5_runtime_full_train_and_test(self):
        """Verify runtime on full train + test inputs (must finish on a 7 GB laptop)."""
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK: FULL TRAIN + TEST FEATURES")
        print("=" * 60)

        # Train features
        train_inputs = self.train[INPUT_COLS]
        t0 = time.time()
        train_feats = build_features(train_inputs, self.static)
        train_time = time.time() - t0
        print(f"Full Train: {len(train_feats):,} rows in {train_time:.2f}s ({len(train_feats)/train_time:,.0f} rows/s)")

        # Test features
        test_inputs = self.test[INPUT_COLS]
        t0 = time.time()
        test_feats = build_features(test_inputs, self.static)
        test_time = time.time() - t0
        print(f"Full Test:  {len(test_feats):,} rows in {test_time:.2f}s ({len(test_feats)/test_time:,.0f} rows/s)")

        total_rows = len(train_feats) + len(test_feats)
        total_time = train_time + test_time
        print(f"Total:      {total_rows:,} rows in {total_time:.2f}s ({total_rows/total_time:,.0f} rows/s)")
        print(f"Memory:     {train_feats.memory_usage(deep=True).sum() / 1e9:.2f} GB (Train), {test_feats.memory_usage(deep=True).sum() / 1e9:.2f} GB (Test)")
        print("=" * 60)

        # Ensure reasonable benchmark
        self.assertLess(
            total_time,
            120.0,
            f"Feature generation took {total_time:.2f}s, exceeding 120s limit!",
        )


if __name__ == "__main__":
    unittest.main()



def test_unsorted_input_rows_get_their_own_features():
    """Regression test: rows given in any order must receive the features of their own (basin, date)."""
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import pytest
    from src.bs_features import FEATURE_COLS, INPUT_COLS, build_features
    data = Path(__file__).resolve().parents[1] / "data" / "caravan"
    if not (data / "train.parquet").exists():
        pytest.skip("data/caravan not present")
    tr = pd.read_parquet(data / "train.parquet", columns=INPUT_COLS)
    st = pd.read_parquet(data / "static.parquet")
    s = tr[tr["gauge_id"].isin(sorted(tr["gauge_id"].unique())[:3])].sort_values(["gauge_id", "date"])
    s = s.reset_index(drop=True)
    ref = build_features(s, st).set_index(["gauge_id", "date"])
    sh = s.sample(frac=1.0, random_state=3).reset_index(drop=True)
    got = build_features(sh, st)
    exp = ref.loc[list(zip(sh["gauge_id"], sh["date"]))]
    assert np.allclose(got["flow_obs"].to_numpy(float), sh["streamflow_obs"].to_numpy(float), equal_nan=True)
    assert np.allclose(got[FEATURE_COLS].to_numpy(float), exp[FEATURE_COLS].to_numpy(float), equal_nan=True)
