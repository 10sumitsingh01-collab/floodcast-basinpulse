"""Model training, weighting, and inference module (Lane C).

Shared interfaces:
  - make_weights(train_df: pd.DataFrame, q: float = 0.90, boost: float = 5.0) -> np.ndarray
  - fit_models(X_train, y_train, train_df, params: dict | None = None) -> dict
  - predict(bundle: dict, X: pd.DataFrame, df: pd.DataFrame) -> np.ndarray

Hard rules enforced:
  - Rule 5: Predictions must be non-negative, finite, one row per test row, same order.
  - Rule 7: All randomness uses random_state=42.
  - Rule 8: Anything computed from target (station percentiles, smearing factors)
            must be computed strictly on TRAIN rows of the current CV fold only.
"""

from typing import Any, Dict, Optional
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import (
    DEFAULT_WEIGHT_BOOST,
    DEFAULT_WEIGHT_Q,
    RANDOM_STATE,
    STATION_COL,
    TARGET_COL,
)


def make_weights(
    train_df: pd.DataFrame,
    q: float = DEFAULT_WEIGHT_Q,
    boost: float = DEFAULT_WEIGHT_BOOST,
) -> np.ndarray:
    """Compute per-station sample weights to emphasize extreme flood events (Rule 8).

    For each station in train_df, calculates the q-th quantile of discharge_cfs.
    Rows at or above this threshold receive a weight of `boost`, while lower rows
    receive a weight of 1.0.

    Input:
      train_df: DataFrame containing at least 'station_id' and 'discharge_cfs'.
      q: Quantile threshold (default 0.90 for top-10% flood days).
      boost: Weight multiplier for extreme events (default 5.0).
    Output:
      np.ndarray of shape (len(train_df),) with sample weights >= 1.0.
    """
    if len(train_df) == 0:
        return np.array([], dtype=float)

    if TARGET_COL not in train_df.columns:
        return np.ones(len(train_df), dtype=float)

    weights = np.ones(len(train_df), dtype=float)
    y = pd.to_numeric(train_df[TARGET_COL], errors="coerce").fillna(0.0).values
    stations = train_df[STATION_COL].values

    unique_stations = np.unique(stations)
    for sid in unique_stations:
        mask = stations == sid
        station_y = y[mask]
        if len(station_y) > 0:
            threshold = float(np.nanquantile(station_y, q))
            extreme_mask = mask & (y >= threshold)
            weights[extreme_mask] = float(boost)

    return weights


def fit_models(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    train_df: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fits an ensemble of gradient-boosted trees targeting raw cfs and log-calibrated surge.

    Inputs:
      X_train: pd.DataFrame of feature columns.
      y_train: target array or Series in raw discharge_cfs.
      train_df: full train DataFrame of the current fold (contains station_id, date, etc.).
      params: optional dictionary of model hyperparameters.
    Returns:
      bundle: dict containing fitted models, smearing factors, and feature specifications.
    """
    params = params or {}
    q = params.get("q", DEFAULT_WEIGHT_Q)
    boost = params.get("boost", DEFAULT_WEIGHT_BOOST)
    n_estimators = params.get("n_estimators", 150)
    learning_rate = params.get("learning_rate", 0.05)

    y_arr = np.asarray(y_train, dtype=float).ravel()
    y_arr = np.maximum(0.0, np.nan_to_num(y_arr, nan=0.0))

    # Compute sample weights strictly from train_df (Rule 8)
    sample_weights = make_weights(train_df, q=q, boost=boost)
    if len(sample_weights) != len(y_arr):
        sample_weights = np.ones(len(y_arr), dtype=float)

    # 1. LightGBM on raw cfs (RMSE with top-peak sample weights)
    lgb_params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "force_col_wise": True,
        "n_jobs": -1,
    }
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_arr, sample_weight=sample_weights)

    # 2. XGBoost on raw cfs (squared error with sample weights)
    xgb_params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "tree_method": "hist",
        "n_jobs": -1,
    }
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_arr, sample_weight=sample_weights)

    # 3. LightGBM on log1p(y) with Duan's Smearing Calibration
    y_log = np.log1p(y_arr)
    lgb_log_model = lgb.LGBMRegressor(**lgb_params)
    lgb_log_model.fit(X_train, y_log, sample_weight=sample_weights)

    # Duan's Smearing Factor computed strictly on train residuals (Rule 8)
    log_preds_train = lgb_log_model.predict(X_train)
    log_residuals = y_log - log_preds_train
    smearing_factor = float(np.mean(np.exp(log_residuals)))

    # Per-station historical summaries on train fold (Rule 8)
    station_stats = {}
    stations = train_df[STATION_COL].values
    for sid in np.unique(stations):
        s_y = y_arr[stations == sid]
        if len(s_y) > 0:
            station_stats[sid] = {
                "median": float(np.nanmedian(s_y)),
                "mean": float(np.nanmean(s_y)),
                "q90": float(np.nanquantile(s_y, 0.90)),
                "max": float(np.nanmax(s_y)),
            }

    global_stats = {
        "median": float(np.nanmedian(y_arr)) if len(y_arr) > 0 else 100.0,
        "mean": float(np.nanmean(y_arr)) if len(y_arr) > 0 else 200.0,
        "q90": float(np.nanquantile(y_arr, 0.90)) if len(y_arr) > 0 else 500.0,
    }

    bundle = {
        "lgb_raw": lgb_model,
        "xgb_raw": xgb_model,
        "lgb_log": lgb_log_model,
        "smearing_factor": smearing_factor,
        "station_stats": station_stats,
        "global_stats": global_stats,
        "feature_cols": list(X_train.columns),
        "weights": {"lgb": 0.45, "xgb": 0.45, "log": 0.10},
    }
    return bundle


def predict(bundle: Dict[str, Any], X: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    """Generate final discharge predictions adhering strictly to project constraints.

    Inputs:
      bundle: fitted model dictionary from fit_models.
      X: feature matrix matching bundle['feature_cols'].
      df: reference DataFrame (e.g. test_df) with station_id, date to guarantee alignment.
    Returns:
      1D np.ndarray of non-negative, finite float predictions in raw cfs.
    """
    feature_cols = bundle.get("feature_cols", list(X.columns))
    X_mat = X[feature_cols]

    # Model predictions
    p_lgb = bundle["lgb_raw"].predict(X_mat)
    p_xgb = bundle["xgb_raw"].predict(X_mat)

    p_log_raw = bundle["lgb_log"].predict(X_mat)
    smearing = bundle.get("smearing_factor", 1.0)
    # Expm1 with Duan's smearing factor
    p_log = np.expm1(np.maximum(0.0, p_log_raw)) * smearing

    weights = bundle.get("weights", {"lgb": 0.45, "xgb": 0.45, "log": 0.10})
    preds = weights["lgb"] * p_lgb + weights["xgb"] * p_xgb + weights["log"] * p_log

    # Rule 5: Non-negative and finite
    preds = np.asarray(preds, dtype=float).ravel()
    fallback_val = bundle.get("global_stats", {}).get("median", 100.0)
    preds = np.nan_to_num(preds, nan=fallback_val, posinf=fallback_val, neginf=0.0)
    preds = np.maximum(0.0, preds)

    # Validate output length matches reference df
    if len(preds) != len(df):
        raise ValueError(
            f"Prediction length mismatch: generated {len(preds)} predictions for {len(df)} rows"
        )

    return preds
