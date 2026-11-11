"""Lane C Models for BASINSHIFT (src/bs_models.py).

Provides GBDT (LightGBM) model architecture for multi-horizon streamflow forecasting:
  - Horizons: t+1, t+3, t+7
  - Point forecast: predicted_discharge >= 0.0 (Huber / L2 regression)
  - Prediction intervals: lower (10th pct) and upper (90th pct) with lower <= upper
  - Event probabilities: p_high (exceeding q_high) and p_extreme (exceeding q_extreme) with p_extreme <= p_high

Compatible with the BASINSHIFT evaluation interface (basinshift_run.py):
    model = fit(train, static, thresholds)
    preds = predict(model, test, train, static, thresholds) -> long format DataFrame
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.bs_features import FEATURE_COLS, INPUT_COLS, build_features
from src.evaluate import DATE, HI, HORIZON, ID, LO, PE, PH, PRED, PRED_COLS

HORIZONS = (1, 3, 7)
RANDOM_STATE = 42


@dataclass
class HorizonModels:
    """Fitted model bundle for a single horizon."""
    regressor: lgb.LGBMRegressor
    quantile_10: lgb.LGBMRegressor
    quantile_90: lgb.LGBMRegressor
    classifier_high: lgb.LGBMClassifier
    classifier_extreme: lgb.LGBMClassifier


class BasinshiftModel:
    """LightGBM Multi-Horizon Predictor for BASINSHIFT."""

    def __init__(
        self,
        n_estimators: int = 80,
        num_leaves: int = 31,
        learning_rate: float = 0.08,
        random_state: int = RANDOM_STATE,
    ):
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.models: Dict[int, HorizonModels] = {}
        self.feature_cols: List[str] = list(FEATURE_COLS)

    def fit(
        self,
        train_df: pd.DataFrame,
        static_df: pd.DataFrame,
        thresholds_df: pd.DataFrame,
        fit_end_date: Optional[str] = None,
    ) -> BasinshiftModel:
        """Fit models for all horizons on training data.

        If fit_end_date is provided (temporal validation), drops training rows where
        t + h > fit_end_date to strictly prevent temporal leakage into validation.
        """
        # Ensure thresholds mapping as dicts for instantaneous O(1) lookups
        q_high_dict = thresholds_df.set_index("gauge_id")["q_high"].to_dict()
        q_extreme_dict = thresholds_df.set_index("gauge_id")["q_extreme"].to_dict()

        # Generate Lane B features on train inputs
        feats_df = build_features(train_df[INPUT_COLS], static_df)
        X_all = feats_df[self.feature_cols]

        dates = pd.to_datetime(train_df["date"])
        fit_end = pd.to_datetime(fit_end_date) if fit_end_date is not None else None

        for h in HORIZONS:
            target_col = f"streamflow_h{h}"
            y_raw = train_df[target_col].to_numpy(dtype=float)

            # Apply temporal cutoff if specified (Rule: t + h <= fit_end)
            if fit_end is not None:
                cutoff_date = fit_end - pd.Timedelta(days=h)
                valid_mask = (dates <= cutoff_date) & np.isfinite(y_raw)
            else:
                valid_mask = np.isfinite(y_raw)

            X_h = X_all.loc[valid_mask]
            y_h = y_raw[valid_mask]

            # Construct binary event labels using basin-specific thresholds (O(1) dictionary lookups)
            gids = train_df.loc[valid_mask, "gauge_id"].astype(str).values
            q_high_vals = np.array([q_high_dict.get(g, np.nan) for g in gids], dtype=float)
            q_extreme_vals = np.array([q_extreme_dict.get(g, np.nan) for g in gids], dtype=float)

            y_high = (y_h > q_high_vals).astype(int)
            y_extreme = (y_h > q_extreme_vals).astype(int)

            # 1. Point regressor (Huber / L2)
            reg = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                num_leaves=self.num_leaves,
                learning_rate=self.learning_rate,
                objective="regression",
                random_state=self.random_state + h,
                n_jobs=-1,
                verbose=-1,
            )
            reg.fit(X_h, y_h)

            # 2. Lower quantile regressor (alpha=0.10)
            q10 = lgb.LGBMRegressor(
                n_estimators=max(30, self.n_estimators // 2),
                num_leaves=self.num_leaves,
                learning_rate=self.learning_rate,
                objective="quantile",
                alpha=0.10,
                random_state=self.random_state + 10 + h,
                n_jobs=-1,
                verbose=-1,
            )
            q10.fit(X_h, y_h)

            # 3. Upper quantile regressor (alpha=0.90)
            q90 = lgb.LGBMRegressor(
                n_estimators=max(30, self.n_estimators // 2),
                num_leaves=self.num_leaves,
                learning_rate=self.learning_rate,
                objective="quantile",
                alpha=0.90,
                random_state=self.random_state + 20 + h,
                n_jobs=-1,
                verbose=-1,
            )
            q90.fit(X_h, y_h)

            # 4. Binary classifier for high flow (p_high)
            clf_hi = lgb.LGBMClassifier(
                n_estimators=max(30, self.n_estimators // 2),
                num_leaves=self.num_leaves,
                learning_rate=self.learning_rate,
                objective="binary",
                random_state=self.random_state + 30 + h,
                n_jobs=-1,
                verbose=-1,
            )
            clf_hi.fit(X_h, y_high)

            # 5. Binary classifier for extreme flow (p_extreme)
            clf_ex = lgb.LGBMClassifier(
                n_estimators=max(30, self.n_estimators // 2),
                num_leaves=self.num_leaves,
                learning_rate=self.learning_rate,
                objective="binary",
                random_state=self.random_state + 40 + h,
                n_jobs=-1,
                verbose=-1,
            )
            clf_ex.fit(X_h, y_extreme)

            self.models[h] = HorizonModels(
                regressor=reg,
                quantile_10=q10,
                quantile_90=q90,
                classifier_high=clf_hi,
                classifier_extreme=clf_ex,
            )

        return self

    def predict(
        self,
        test_df: pd.DataFrame,
        train_df: pd.DataFrame,
        static_df: pd.DataFrame,
        thresholds_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate long-format predictions for test_df.

        Concatenates train and test INPUT_COLS to ensure full historical lookback
        for early test dates, then extracts features strictly for test rows.
        """
        # Strict validation: prevent target leakage
        target_cols = [f"streamflow_h{h}" for h in HORIZONS] + ["streamflow"]
        leakage = [c for c in target_cols if c in test_df.columns]
        if leakage:
            raise ValueError(f"test_df must NEVER receive target columns: {leakage}")

        base_id = test_df["gauge_id"].astype(str).values
        base_date = pd.to_datetime(test_df["date"]).values

        # Relevant historical lookback from train_df:
        # Keep only basins present in test_df, and only dates strictly prior to each basin's earliest test date
        test_basins = set(base_id)
        train_rel = train_df[train_df["gauge_id"].astype(str).isin(test_basins)]
        if len(train_rel) > 0:
            min_test_dates = test_df.groupby(test_df["gauge_id"].astype(str))["date"].apply(
                lambda s: pd.to_datetime(s).min()
            ).to_dict()
            t_gids = train_rel["gauge_id"].astype(str).values
            t_dates = pd.to_datetime(train_rel["date"]).values
            keep_mask = np.array([d < min_test_dates[g] for g, d in zip(t_gids, t_dates)], dtype=bool)
            train_lookback = train_rel.loc[keep_mask]
        else:
            train_lookback = train_rel.iloc[0:0]

        # Concatenate lookback and test inputs for continuous calendar features
        if len(train_lookback) > 0:
            combined_inputs = pd.concat([train_lookback[INPUT_COLS], test_df[INPUT_COLS]], ignore_index=True)
            all_feats = build_features(combined_inputs, static_df)
            test_feats = all_feats.iloc[len(train_lookback):].reset_index(drop=True)
        else:
            test_feats = build_features(test_df[INPUT_COLS], static_df).reset_index(drop=True)

        assert (test_feats["gauge_id"].values == base_id).all(), "Feature basin ID alignment mismatch!"
        assert (test_feats["date"].values == base_date).all(), "Feature date alignment mismatch!"

        X_test = test_feats[self.feature_cols]

        parts = []
        for h in HORIZONS:
            hm = self.models[h]

            # 1. Point prediction (non-negative)
            raw_pred = hm.regressor.predict(X_test)
            pred_val = np.maximum(0.0, raw_pred)

            # 2. Prediction intervals (lower <= upper, non-negative)
            raw_lo = hm.quantile_10.predict(X_test)
            raw_hi = hm.quantile_90.predict(X_test)
            lo_val = np.maximum(0.0, raw_lo)
            hi_val = np.maximum(lo_val, raw_hi)

            # 3. Probabilities (bounded in [0, 1] and p_extreme <= p_high)
            raw_ph = hm.classifier_high.predict_proba(X_test)[:, 1]
            raw_pe = hm.classifier_extreme.predict_proba(X_test)[:, 1]
            ph_val = np.clip(raw_ph, 0.0, 1.0)
            pe_val = np.clip(np.minimum(raw_pe, ph_val), 0.0, 1.0)

            part = pd.DataFrame({
                ID: base_id,
                DATE: base_date,
                HORIZON: h,
                PRED: pred_val.astype(np.float32),
                LO: lo_val.astype(np.float32),
                HI: hi_val.astype(np.float32),
                PH: ph_val.astype(np.float32),
                PE: pe_val.astype(np.float32),
            })
            parts.append(part)

        return pd.concat(parts, ignore_index=True)


# Module-level functions to plug directly into basinshift_run.py
def fit(train: pd.DataFrame, static: pd.DataFrame, thresholds: pd.DataFrame) -> BasinshiftModel:
    model = BasinshiftModel()
    return model.fit(train, static, thresholds)


def predict(
    model: BasinshiftModel,
    test: pd.DataFrame,
    train: pd.DataFrame,
    static: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    return model.predict(test, train, static, thresholds)

