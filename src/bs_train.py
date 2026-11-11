"""Training workflows and validation splits for BASINSHIFT (src/bs_train.py).

Implements official basin-aware split evaluations:
  1. Temporal Validation: fit on <= 2006-09-30, validate on [2006-10-01, 2008-09-30].
     Rule: t + h <= fit_end ensures zero target leakage into the validation window.
  2. Spatial Validation: fit on 4 folds, validate on 1 held-out fold of unseen basins.
  3. Full Training: fit on all training basins up to 2008-09-30 for test evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.bs_features import INPUT_COLS
from src.bs_models import BasinshiftModel, HORIZONS
from src.evaluate import DATE, HI, HORIZON, ID, LO, PE, PH, PRED, PRED_COLS, score

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "caravan"


def evaluate_predictions(
    preds_long: pd.DataFrame,
    labels_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
) -> Dict[str, Dict[int, float]]:
    """Compute standard metrics per horizon on long-format predictions."""
    q_high_dict = thresholds_df.set_index("gauge_id")["q_high"].to_dict()
    q_extreme_dict = thresholds_df.set_index("gauge_id")["q_extreme"].to_dict()

    # Merge labels with predictions
    metrics: Dict[str, Dict[int, float]] = {
        "mae_pooled": {},
        "nse_pooled": {},
        "nse_median": {},
        "kge_median": {},
        "interval_coverage": {},
        "interval_width": {},
        "auprc_high": {},
        "roc_auc_high": {},
        "brier_high": {},
        "brier_skill_high": {},
        "auprc_extreme": {},
        "roc_auc_extreme": {},
        "brier_extreme": {},
        "brier_skill_extreme": {},
    }

    for h in HORIZONS:
        target_col = f"streamflow_h{h}"
        p_sub = preds_long[preds_long["horizon"] == h].copy()

        # Join true target
        merged = p_sub.merge(
            labels_df[["gauge_id", "date", target_col]],
            on=["gauge_id", "date"],
            how="inner",
        )
        merged = merged.dropna(subset=[target_col])
        if len(merged) == 0:
            continue

        y_true = merged[target_col].to_numpy(dtype=float)
        y_pred = merged["predicted_discharge"].to_numpy(dtype=float)
        lo = merged["lower"].to_numpy(dtype=float)
        hi = merged["upper"].to_numpy(dtype=float)
        ph = merged["p_high"].to_numpy(dtype=float)
        pe = merged["p_extreme"].to_numpy(dtype=float)

        # 1. MAE & Pooled NSE
        mae = float(np.mean(np.abs(y_true - y_pred)))
        denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
        nse_pooled = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else np.nan

        # 2. Per-basin NSE and KGE median
        basin_nses = []
        basin_kges = []
        for _, grp in merged.groupby("gauge_id"):
            yt = grp[target_col].values
            yp = grp["predicted_discharge"].values
            if len(yt) >= 30 and np.var(yt) > 1e-6:
                basin_nse = 1.0 - np.sum((yt - yp) ** 2) / np.sum((yt - np.mean(yt)) ** 2)
                basin_nses.append(basin_nse)

                r = np.corrcoef(yt, yp)[0, 1] if np.std(yp) > 0 else 0.0
                alpha = np.std(yp) / np.std(yt)
                beta = np.mean(yp) / np.mean(yt) if np.mean(yt) != 0 else np.nan
                kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
                if np.isfinite(kge):
                    basin_kges.append(kge)

        nse_median = float(np.median(basin_nses)) if basin_nses else np.nan
        kge_median = float(np.median(basin_kges)) if basin_kges else np.nan

        # 3. Interval coverage and width
        covered = (y_true >= lo) & (y_true <= hi)
        coverage = float(np.mean(covered))
        width = float(np.mean(hi - lo))

        # 4. Event metrics: High flow exceedance (95th pct threshold)
        gids = merged["gauge_id"].astype(str).values
        q_high_arr = np.array([q_high_dict.get(g, np.nan) for g in gids], dtype=float)
        y_high = (y_true > q_high_arr).astype(int)

        if np.sum(y_high) > 0 and len(np.unique(y_high)) > 1:
            auprc_h = float(average_precision_score(y_high, ph))
            roc_h = float(roc_auc_score(y_high, ph))
            brier_h = float(brier_score_loss(y_high, ph))
            ref_var_h = 0.05 * 0.95
            bss_h = float(1.0 - brier_h / ref_var_h)
        else:
            auprc_h = np.nan
            roc_h = np.nan
            brier_h = np.nan
            bss_h = np.nan

        # 5. Event metrics: Extreme flow exceedance (99th pct threshold)
        q_extreme_arr = np.array([q_extreme_dict.get(g, np.nan) for g in gids], dtype=float)
        y_extreme = (y_true > q_extreme_arr).astype(int)

        if np.sum(y_extreme) > 0 and len(np.unique(y_extreme)) > 1:
            auprc_e = float(average_precision_score(y_extreme, pe))
            roc_e = float(roc_auc_score(y_extreme, pe))
            brier_e = float(brier_score_loss(y_extreme, pe))
            ref_var_e = 0.01 * 0.99
            bss_e = float(1.0 - brier_e / ref_var_e)
        else:
            auprc_e = np.nan
            roc_e = np.nan
            brier_e = np.nan
            bss_e = np.nan

        metrics["mae_pooled"][h] = mae
        metrics["nse_pooled"][h] = nse_pooled
        metrics["nse_median"][h] = nse_median
        metrics["kge_median"][h] = kge_median
        metrics["interval_coverage"][h] = coverage
        metrics["interval_width"][h] = width
        metrics["auprc_high"][h] = auprc_h
        metrics["roc_auc_high"][h] = roc_h
        metrics["brier_high"][h] = brier_h
        metrics["brier_skill_high"][h] = bss_h
        metrics["auprc_extreme"][h] = auprc_e
        metrics["roc_auc_extreme"][h] = roc_e
        metrics["brier_extreme"][h] = brier_e
        metrics["brier_skill_extreme"][h] = bss_e

    return metrics


def train_temporal_val(
    train_df: pd.DataFrame,
    static_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    splits_path: Optional[Path] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[BasinshiftModel, pd.DataFrame, Dict[str, Dict[int, float]]]:
    """Train on fit period (<= 2006-09-30) and evaluate on validation period (>= 2006-10-01)."""
    p = splits_path or (DATA_DIR / "splits.json")
    with open(p, "r") as f:
        splits = json.load(f)

    fit_end = splits["temporal_validation"]["fit_end_inclusive"]
    val_start = splits["temporal_validation"]["val_start_inclusive"]

    print(f"Temporal validation setup: fit_end={fit_end}, val_start={val_start}")
    print(f"Total input basins: {train_df['gauge_id'].nunique()}, total rows: {len(train_df):,}")

    params = params or {}
    model = BasinshiftModel(**params)

    # 1. Fit on training rows up to fit_end (with t + h <= fit_end enforced)
    model.fit(train_df, static_df, thresholds_df, fit_end_date=fit_end)

    # 2. Partition validation rows
    val_mask = pd.to_datetime(train_df["date"]) >= pd.to_datetime(val_start)
    val_df = train_df[val_mask].copy()
    val_inputs = val_df[INPUT_COLS].copy()
    print(f"Validation partition: {len(val_df):,} rows from {val_df['date'].min()} to {val_df['date'].max()} across {val_df['gauge_id'].nunique()} basins")

    # 3. Predict on validation partition using training history (fit period) for lookback
    fit_mask = pd.to_datetime(train_df["date"]) <= pd.to_datetime(fit_end)
    fit_df = train_df[fit_mask].copy()
    preds_val = model.predict(val_inputs, fit_df, static_df, thresholds_df)

    # 4. Evaluate metrics
    metrics = evaluate_predictions(preds_val, val_df, thresholds_df)
    return model, preds_val, metrics


def train_spatial_val(
    train_df: pd.DataFrame,
    static_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    val_fold: int = 4,
    splits_path: Optional[Path] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[BasinshiftModel, pd.DataFrame, Dict[str, Dict[int, float]]]:
    """Train on 4 folds of basins and evaluate on 1 held-out fold of unseen basins."""
    p = splits_path or (DATA_DIR / "splits.json")
    with open(p, "r") as f:
        splits = json.load(f)

    station_folds = splits["station_folds"]
    val_basins = sorted([gid for gid, fold in station_folds.items() if fold == val_fold and gid in train_df["gauge_id"].values])
    train_basins = sorted([gid for gid, fold in station_folds.items() if fold != val_fold and gid in train_df["gauge_id"].values])

    print(f"Spatial validation setup: held-out fold={val_fold}")
    print(f"Train (seen) basins count: {len(train_basins)}")
    print(f"Val (unseen) basins count: {len(val_basins)}")
    print(f"Val (unseen) basin IDs: {val_basins}")

    train_fold_df = train_df[train_df["gauge_id"].isin(train_basins)].copy()
    val_fold_df = train_df[train_df["gauge_id"].isin(val_basins)].copy()
    val_inputs = val_fold_df[INPUT_COLS].copy()
    print(f"Train rows: {len(train_fold_df):,}, Val rows: {len(val_fold_df):,}")

    params = params or {}
    model = BasinshiftModel(**params)

    # Fit strictly on seen basins
    model.fit(train_fold_df, static_df, thresholds_df)

    # Predict on unseen basins
    preds_val = model.predict(val_inputs, train_fold_df, static_df, thresholds_df)

    # Evaluate metrics on unseen basins
    metrics = evaluate_predictions(preds_val, val_fold_df, thresholds_df)
    return model, preds_val, metrics


if __name__ == "__main__":
    import argparse
    import time
    import tracemalloc

    parser = argparse.ArgumentParser(description="BASINSHIFT Validation Runner")
    parser.add_argument("--mode", choices=["temporal", "spatial", "both"], default="temporal")
    parser.add_argument("--n-basins", type=int, default=None, help="Subset number of basins for fast run")
    parser.add_argument("--n-estimators", type=int, default=60)
    args = parser.parse_args()

    tracemalloc.start()
    t0 = time.time()

    print(f"Loading data from {DATA_DIR}...")
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    static_df = pd.read_parquet(DATA_DIR / "static.parquet")
    thr_val = pd.read_csv(DATA_DIR / "thresholds_val.csv")

    if args.n_basins is not None:
        sel_basins = train_df["gauge_id"].unique()[:args.n_basins]
        train_df = train_df[train_df["gauge_id"].isin(sel_basins)].copy()
        print(f"Subsetting to {args.n_basins} basins ({len(train_df):,} rows)...")

    model_params = {"n_estimators": args.n_estimators}

    if args.mode in ("temporal", "both"):
        print("\n=== Running Temporal Validation (<= 2006-09-30 fit, 2006-2008 val) ===")
        t_start = time.time()
        _, preds_temp, metrics_temp = train_temporal_val(
            train_df, static_df, thr_val, params=model_params
        )
        t_dur = time.time() - t_start
        _, peak_bytes = tracemalloc.get_traced_memory()
        print(f"Temporal Validation finished in {t_dur:.1f}s | Traced Peak RAM: {peak_bytes / (1024**3):.2f} GB")
        print("Metrics per horizon:")
        for metric_name, h_dict in metrics_temp.items():
            vals = ", ".join([f"h={h}: {val:.4f}" if np.isfinite(val) else f"h={h}: nan" for h, val in h_dict.items()])
            print(f"  {metric_name:<20}: {vals}")

    if args.mode in ("spatial", "both"):
        print("\n=== Running Spatial Validation (Fold 4 unseen basins) ===")
        t_start = time.time()
        _, preds_spat, metrics_spat = train_spatial_val(
            train_df, static_df, thr_val, val_fold=4, params=model_params
        )
        t_dur = time.time() - t_start
        _, peak_bytes = tracemalloc.get_traced_memory()
        print(f"Spatial Validation finished in {t_dur:.1f}s | Traced Peak RAM: {peak_bytes / (1024**3):.2f} GB")
        print("Metrics per horizon:")
        for metric_name, h_dict in metrics_spat.items():
            vals = ", ".join([f"h={h}: {val:.4f}" if np.isfinite(val) else f"h={h}: nan" for h, val in h_dict.items()])
            print(f"  {metric_name:<20}: {vals}")

    total_time = time.time() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    print(f"\nTotal elapsed: {total_time:.1f}s | Traced Peak RAM: {peak_bytes / (1024**3):.2f} GB")

