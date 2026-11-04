"""FLOODCAST Hackathon Solution Orchestrator (Lane D).

Runs the complete end-to-end ML pipeline:
  1. Load train, test, and station metadata (Lane A)
  2. Build hydrological features without target leakage (Lane B)
  3. Validate using forward temporal chaining and LOSO folds (Lane D)
  4. Fit final calibrated ensemble models with peak weighting (Lane C)
  5. Generate non-negative predictions for test.csv (Lane C)
  6. Finalize predictions.csv and verify submission contract (Lane D)
  7. Append results to experiments/results.csv (Lane D)

Usage:
  python solution.py
  python solution.py --quick
"""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd

from src.config import (
    DATE_COL,
    EXPERIMENTS_PATH,
    PREDICTIONS_PATH,
    RANDOM_STATE,
    SCORING_WEIGHTS,
    STATION_COL,
    TARGET_COL,
    TEST_PATH,
)
from src.data import load_station_metadata, load_test, load_train
from src.features import FEATURE_COLS, build_features
from src.models import fit_models, predict
from src.validate import (
    finalize,
    forward_folds,
    loso_folds,
    score,
    validate_submission,
)


def get_git_commit_hash() -> str:
    """Retrieve current git commit hash if in a git repo."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return res.stdout.strip() if res.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def log_experiment(metrics: dict, description: str = "production_ensemble") -> None:
    """Append experiment metrics and run metadata to experiments/results.csv."""
    EXPERIMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec="seconds")
    commit = get_git_commit_hash()

    row = {
        "timestamp": timestamp,
        "commit": commit,
        "description": description,
        "val_rmse": metrics.get("rmse", np.nan),
        "val_top10_rmse": metrics.get("top10_rmse", np.nan),
        "val_mae": metrics.get("mae", np.nan),
        "val_r2": metrics.get("r2", np.nan),
    }

    results_df = pd.DataFrame([row])
    if EXPERIMENTS_PATH.exists():
        results_df.to_csv(EXPERIMENTS_PATH, mode="a", header=False, index=False)
    else:
        results_df.to_csv(EXPERIMENTS_PATH, mode="w", header=True, index=False)


def run_pipeline(quick: bool = False, run_loso: bool = False) -> None:
    print("=" * 70)
    print("FLOODCAST PIPELINE EXECUTION")
    print("=" * 70)

    # 1. Load Data (Lane A)
    print("\n[Step 1/6] Loading data...")
    train_df = load_train()
    test_df = load_test()
    meta_df = load_station_metadata()

    print(f"  Loaded Train: {len(train_df):,} rows, {train_df[STATION_COL].nunique()} stations")
    print(f"  Train Date Range: {train_df[DATE_COL].min().date()} to {train_df[DATE_COL].max().date()}")
    print(f"  Loaded Test:  {len(test_df):,} rows, {test_df[STATION_COL].nunique()} stations")
    print(f"  Test Date Range:  {test_df[DATE_COL].min().date()} to {test_df[DATE_COL].max().date()}")

    # 2. Build Features (Lane B)
    print("\n[Step 2/6] Building features (zero target leakage)...")
    train_len = len(train_df)
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    all_features_df = build_features(combined_df)

    X_train = all_features_df.iloc[:train_len][FEATURE_COLS]
    y_train = train_df[TARGET_COL].values
    X_test = all_features_df.iloc[train_len:][FEATURE_COLS]

    print(f"  Total features engineered: {len(FEATURE_COLS)}")
    print(f"  X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # Model hyperparams
    n_estimators = 50 if quick else 200
    learning_rate = 0.08 if quick else 0.04
    model_params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "q": 0.90,
        "boost": 5.0,
    }

    # 3. Cross-Validation (Lane D)
    print("\n[Step 3/6] Running Forward-Chaining Temporal Cross-Validation...")
    val_years = sorted(train_df[DATE_COL].dt.year.unique())
    # Evaluate on the last 2 available training years
    eval_years = val_years[-2:] if len(val_years) >= 3 else val_years[-1:]

    oof_actuals = []
    oof_preds = []
    cv_records = []

    for fold_i, (trn_idx, val_idx) in enumerate(forward_folds(train_df, val_years=eval_years, gap_days=30)):
        fold_train_df = train_df.iloc[trn_idx]
        fold_val_df = train_df.iloc[val_idx]

        fold_X_train = X_train.iloc[trn_idx]
        fold_y_train = y_train[trn_idx]
        fold_X_val = X_train.iloc[val_idx]

        bundle = fit_models(fold_X_train, fold_y_train, fold_train_df, params=model_params)
        fold_pred = predict(bundle, fold_X_val, fold_val_df)

        fold_scores = score(fold_val_df, fold_pred)
        val_year_label = fold_val_df[DATE_COL].dt.year.iloc[0]
        print(
            f"  Fold {fold_i + 1} (Year {val_year_label}) | "
            f"RMSE: {fold_scores['rmse']:.2f} cfs | "
            f"Top-10% RMSE: {fold_scores['top10_rmse']:.2f} cfs | "
            f"MAE: {fold_scores['mae']:.2f} cfs | "
            f"R2: {fold_scores['r2']:.4f}"
        )

        oof_actuals.append(fold_val_df)
        oof_preds.append(fold_pred)
        cv_records.append(fold_scores)

    if oof_actuals:
        pooled_val_df = pd.concat(oof_actuals, ignore_index=True)
        pooled_preds = np.concatenate(oof_preds)
        overall_cv = score(pooled_val_df, pooled_preds)
    else:
        overall_cv = {"rmse": 0.0, "top10_rmse": 0.0, "mae": 0.0, "r2": 0.0}

    print("-" * 50)
    print("  POOLED TEMPORAL CV METRICS:")
    print(f"    -> Overall RMSE:        {overall_cv['rmse']:.2f} cfs")
    print(f"    -> Top-10% Flood RMSE:  {overall_cv['top10_rmse']:.2f} cfs")
    print(f"    -> Overall MAE:         {overall_cv['mae']:.2f} cfs")
    print(f"    -> R2 Score:            {overall_cv['r2']:.4f}")
    print("-" * 50)

    if run_loso:
        print("\n  Running Leave-One-Station-Out (LOSO) CV for Geographic Generalization...")
        loso_scores_list = []
        for sid, (trn_idx, val_idx) in enumerate(loso_folds(train_df)):
            s_name = train_df.iloc[val_idx][STATION_COL].iloc[0]
            fold_bundle = fit_models(X_train.iloc[trn_idx], y_train[trn_idx], train_df.iloc[trn_idx], params=model_params)
            s_pred = predict(fold_bundle, X_train.iloc[val_idx], train_df.iloc[val_idx])
            s_metrics = score(train_df.iloc[val_idx], s_pred)
            print(f"    Station {s_name} | RMSE: {s_metrics['rmse']:.2f} cfs | Top10: {s_metrics['top10_rmse']:.2f} cfs | R2: {s_metrics['r2']:.4f}")
            loso_scores_list.append(s_metrics)

    # 4. Fit Production Ensemble on Full Train Data (Lane C)
    print("\n[Step 4/6] Training production ensemble on 100% of train data...")
    final_bundle = fit_models(X_train, y_train, train_df, params=model_params)
    print("  Models trained: LightGBM (raw cfs), XGBoost (raw cfs), LightGBM (log-calibrated)")
    print(f"  Duan's Smearing Factor: {final_bundle['smearing_factor']:.4f}")

    # 5. Predict on Test Set (Lane C)
    print("\n[Step 5/6] Generating predictions for test.csv...")
    test_preds = predict(final_bundle, X_test, test_df)
    print(f"  Predictions generated: {len(test_preds):,}")
    print(f"  Summary stats - Min: {test_preds.min():.2f}, Median: {np.median(test_preds):.2f}, "
          f"Mean: {test_preds.mean():.2f}, Max: {test_preds.max():.2f}")

    # 6. Finalize & Validate Submission (Lane D)
    print("\n[Step 6/6] Finalizing predictions.csv and validating submission contract...")
    finalize(test_df, test_preds, path=PREDICTIONS_PATH)
    validate_submission(PREDICTIONS_PATH, TEST_PATH)
    print(f"  SUCCESS! Verified submission at: {PREDICTIONS_PATH}")

    # Log to experiments/results.csv
    log_experiment(overall_cv, description=f"ensemble_n{n_estimators}_lr{learning_rate}")
    print(f"  Logged validation metrics to: {EXPERIMENTS_PATH}")

    print("\n" + "=" * 70)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="FLOODCAST Hackathon Solution Orchestrator")
    parser.add_argument("--quick", action="store_true", help="Run with fewer trees for fast iteration")
    parser.add_argument("--loso", action="store_true", help="Run Leave-One-Station-Out CV")
    args = parser.parse_args()

    run_pipeline(quick=args.quick, run_loso=args.loso)


if __name__ == "__main__":
    main()

