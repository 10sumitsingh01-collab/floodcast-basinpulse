import time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from src.bs_models import BasinshiftModel, HORIZONS
from src.bs_train import train_temporal_val, train_spatial_val, evaluate_predictions
from src.bs_features import INPUT_COLS

print("Loading data...")
train_df = pd.read_parquet("data/caravan/train.parquet")
static_df = pd.read_parquet("data/caravan/static.parquet")
thr_val = pd.read_csv("data/caravan/thresholds_val.csv")

sel_basins = train_df["gauge_id"].unique()[:40]
train_sub = train_df[train_df["gauge_id"].isin(sel_basins)].copy()

# ==============================================================================
# Baseline
# ==============================================================================
print("\n--- Running Baseline ---")
t0 = time.time()
_, p_temp_base, m_temp_base = train_temporal_val(train_sub, static_df, thr_val, params={"n_estimators": 40})
_, p_spat_base, m_spat_base = train_spatial_val(train_sub, static_df, thr_val, val_fold=4, params={"n_estimators": 40})
t_base = time.time() - t0
print(f"Baseline done in {t_base:.1f}s")

# ==============================================================================
# Experiment 1: Horizon-specific regularization & feature subsampling for h=7
# In LightGBM: colsample_bytree=0.8, subsample=0.8, min_child_samples=50 for h=7
# ==============================================================================
print("\n--- Experiment 1: Regularization & Subsampling ---")
# Test modified BasinshiftModel with colsample_bytree=0.8, subsample=0.8
class BasinshiftModelExp1(BasinshiftModel):
    def __init__(self, n_estimators=40, num_leaves=31, learning_rate=0.08, random_state=42):
        super().__init__(n_estimators, num_leaves, learning_rate, random_state)

    def fit(self, train_df, static_df, thresholds_df, fit_end_date=None):
        import lightgbm as lgb
        q_high_dict = thresholds_df.set_index("gauge_id")["q_high"].to_dict()
        q_extreme_dict = thresholds_df.set_index("gauge_id")["q_extreme"].to_dict()
        from src.bs_features import build_features
        feats_df = build_features(train_df[INPUT_COLS], static_df)
        X_all = feats_df[self.feature_cols]

        dates = pd.to_datetime(train_df["date"])
        fit_end = pd.to_datetime(fit_end_date) if fit_end_date is not None else None

        for h in HORIZONS:
            target_col = f"streamflow_h{h}"
            y_raw = train_df[target_col].to_numpy(dtype=float)
            if fit_end is not None:
                cutoff_date = fit_end - pd.Timedelta(days=h)
                valid_mask = (dates <= cutoff_date) & np.isfinite(y_raw)
            else:
                valid_mask = np.isfinite(y_raw)

            X_h = X_all.loc[valid_mask]
            y_h = y_raw[valid_mask]

            gids = train_df.loc[valid_mask, "gauge_id"].astype(str).values
            q_high_vals = np.array([q_high_dict.get(g, np.nan) for g in gids], dtype=float)
            q_extreme_vals = np.array([q_extreme_dict.get(g, np.nan) for g in gids], dtype=float)
            y_high = (y_h > q_high_vals).astype(int)
            y_extreme = (y_h > q_extreme_vals).astype(int)

            # Horizon-specific subsampling/regularization for h=7
            colsample = 0.8 if h == 7 else 1.0
            subsample = 0.8 if h == 7 else 1.0
            subsample_freq = 1 if h == 7 else 0

            reg = lgb.LGBMRegressor(
                n_estimators=self.n_estimators, num_leaves=self.num_leaves,
                learning_rate=self.learning_rate, objective="regression",
                colsample_bytree=colsample, subsample=subsample, subsample_freq=subsample_freq,
                random_state=self.random_state + h, n_jobs=-1, verbose=-1,
            )
            reg.fit(X_h, y_h)

            q10 = lgb.LGBMRegressor(
                n_estimators=max(30, self.n_estimators // 2), num_leaves=self.num_leaves,
                learning_rate=self.learning_rate, objective="quantile", alpha=0.10,
                colsample_bytree=colsample, subsample=subsample, subsample_freq=subsample_freq,
                random_state=self.random_state + 10 + h, n_jobs=-1, verbose=-1,
            )
            q10.fit(X_h, y_h)

            q90 = lgb.LGBMRegressor(
                n_estimators=max(30, self.n_estimators // 2), num_leaves=self.num_leaves,
                learning_rate=self.learning_rate, objective="quantile", alpha=0.90,
                colsample_bytree=colsample, subsample=subsample, subsample_freq=subsample_freq,
                random_state=self.random_state + 20 + h, n_jobs=-1, verbose=-1,
            )
            q90.fit(X_h, y_h)

            clf_hi = lgb.LGBMClassifier(
                n_estimators=max(30, self.n_estimators // 2), num_leaves=self.num_leaves,
                learning_rate=self.learning_rate, objective="binary",
                colsample_bytree=colsample, subsample=subsample, subsample_freq=subsample_freq,
                random_state=self.random_state + 30 + h, n_jobs=-1, verbose=-1,
            )
            clf_hi.fit(X_h, y_high)

            clf_ex = lgb.LGBMClassifier(
                n_estimators=max(30, self.n_estimators // 2), num_leaves=self.num_leaves,
                learning_rate=self.learning_rate, objective="binary",
                colsample_bytree=colsample, subsample=subsample, subsample_freq=subsample_freq,
                random_state=self.random_state + 40 + h, n_jobs=-1, verbose=-1,
            )
            clf_ex.fit(X_h, y_extreme)

            from src.bs_models import HorizonModels
            self.models[h] = HorizonModels(reg, q10, q90, clf_hi, clf_ex)
        return self

# Run Exp 1
t0 = time.time()
m_exp1_temp = BasinshiftModelExp1(n_estimators=40)
m_exp1_temp.fit(train_sub, static_df, thr_val, fit_end_date="2006-09-30")
val_mask = pd.to_datetime(train_sub["date"]) >= pd.to_datetime("2006-10-01")
val_df = train_sub[val_mask].copy()
fit_df = train_sub[pd.to_datetime(train_sub["date"]) <= pd.to_datetime("2006-09-30")].copy()
p_temp_exp1 = m_exp1_temp.predict(val_df[INPUT_COLS], fit_df, static_df, thr_val)
m_temp_exp1 = evaluate_predictions(p_temp_exp1, val_df, thr_val)

# Spatial val Exp 1
import json
splits = json.load(open("data/caravan/splits.json"))
val_basins = sorted([gid for gid, fold in splits["station_folds"].items() if fold == 4 and gid in train_sub["gauge_id"].values])
train_basins = sorted([gid for gid, fold in splits["station_folds"].items() if fold != 4 and gid in train_sub["gauge_id"].values])
train_fold_df = train_sub[train_sub["gauge_id"].isin(train_basins)].copy()
val_fold_df = train_sub[train_sub["gauge_id"].isin(val_basins)].copy()

m_exp1_spat = BasinshiftModelExp1(n_estimators=40)
m_exp1_spat.fit(train_fold_df, static_df, thr_val)
p_spat_exp1 = m_exp1_spat.predict(val_fold_df[INPUT_COLS], train_fold_df, static_df, thr_val)
m_spat_exp1 = evaluate_predictions(p_spat_exp1, val_fold_df, thr_val)
t_exp1 = time.time() - t0
print(f"Exp 1 done in {t_exp1:.1f}s")

# ==============================================================================
# Experiment 2: Extreme probability shrinkage towards climatological prior (0.01)
# Brier score penalizes unconfident false alarms in rare events.
# Shrinking: p_ext = (1 - w)*p_ext + w*0.01 for h=3 (w=0.15) and h=7 (w=0.30)
# ==============================================================================
print("\n--- Experiment 2: Probability Shrinkage ---")
p_temp_exp2 = p_temp_base.copy()
for h, w in [(3, 0.15), (7, 0.30)]:
    mask = p_temp_exp2["horizon"] == h
    p_temp_exp2.loc[mask, "p_extreme"] = (1 - w) * p_temp_exp2.loc[mask, "p_extreme"] + w * 0.01
    # Maintain p_extreme <= p_high
    p_temp_exp2.loc[mask, "p_extreme"] = np.minimum(p_temp_exp2.loc[mask, "p_extreme"], p_temp_exp2.loc[mask, "p_high"])
m_temp_exp2 = evaluate_predictions(p_temp_exp2, val_df, thr_val)

p_spat_exp2 = p_spat_base.copy()
for h, w in [(3, 0.15), (7, 0.30)]:
    mask = p_spat_exp2["horizon"] == h
    p_spat_exp2.loc[mask, "p_extreme"] = (1 - w) * p_spat_exp2.loc[mask, "p_extreme"] + w * 0.01
    p_spat_exp2.loc[mask, "p_extreme"] = np.minimum(p_spat_exp2.loc[mask, "p_extreme"], p_spat_exp2.loc[mask, "p_high"])
m_spat_exp2 = evaluate_predictions(p_spat_exp2, val_fold_df, thr_val)

# ==============================================================================
# Experiment 3: Interval Calibration (Tuning alpha for h=1 to achieve ~80% coverage)
# alpha=0.12 and alpha=0.88 for h=1 (leaving h=3 and h=7 at nominal 0.10/0.90)
# ==============================================================================
print("\n--- Experiment 3: Calibrated Quantiles (h=1 alpha 0.12/0.88) ---")
class BasinshiftModelExp3(BasinshiftModel):
    def fit(self, train_df, static_df, thresholds_df, fit_end_date=None):
        import lightgbm as lgb
        q_high_dict = thresholds_df.set_index("gauge_id")["q_high"].to_dict()
        q_extreme_dict = thresholds_df.set_index("gauge_id")["q_extreme"].to_dict()
        from src.bs_features import build_features
        feats_df = build_features(train_df[INPUT_COLS], static_df)
        X_all = feats_df[self.feature_cols]

        dates = pd.to_datetime(train_df["date"])
        fit_end = pd.to_datetime(fit_end_date) if fit_end_date is not None else None

        for h in HORIZONS:
            target_col = f"streamflow_h{h}"
            y_raw = train_df[target_col].to_numpy(dtype=float)
            if fit_end is not None:
                cutoff_date = fit_end - pd.Timedelta(days=h)
                valid_mask = (dates <= cutoff_date) & np.isfinite(y_raw)
            else:
                valid_mask = np.isfinite(y_raw)

            X_h = X_all.loc[valid_mask]
            y_h = y_raw[valid_mask]

            gids = train_df.loc[valid_mask, "gauge_id"].astype(str).values
            q_high_vals = np.array([q_high_dict.get(g, np.nan) for g in gids], dtype=float)
            q_extreme_vals = np.array([q_extreme_dict.get(g, np.nan) for g in gids], dtype=float)
            y_high = (y_h > q_high_vals).astype(int)
            y_extreme = (y_h > q_extreme_vals).astype(int)

            reg = lgb.LGBMRegressor(n_estimators=self.n_estimators, num_leaves=self.num_leaves, learning_rate=self.learning_rate, objective="regression", random_state=self.random_state + h, n_jobs=-1, verbose=-1)
            reg.fit(X_h, y_h)

            a_lo = 0.12 if h == 1 else 0.10
            a_hi = 0.88 if h == 1 else 0.90

            q10 = lgb.LGBMRegressor(n_estimators=max(30, self.n_estimators // 2), num_leaves=self.num_leaves, learning_rate=self.learning_rate, objective="quantile", alpha=a_lo, random_state=self.random_state + 10 + h, n_jobs=-1, verbose=-1)
            q10.fit(X_h, y_h)

            q90 = lgb.LGBMRegressor(n_estimators=max(30, self.n_estimators // 2), num_leaves=self.num_leaves, learning_rate=self.learning_rate, objective="quantile", alpha=a_hi, random_state=self.random_state + 20 + h, n_jobs=-1, verbose=-1)
            q90.fit(X_h, y_h)

            clf_hi = lgb.LGBMClassifier(n_estimators=max(30, self.n_estimators // 2), num_leaves=self.num_leaves, learning_rate=self.learning_rate, objective="binary", random_state=self.random_state + 30 + h, n_jobs=-1, verbose=-1)
            clf_hi.fit(X_h, y_high)

            clf_ex = lgb.LGBMClassifier(n_estimators=max(30, self.n_estimators // 2), num_leaves=self.num_leaves, learning_rate=self.learning_rate, objective="binary", random_state=self.random_state + 40 + h, n_jobs=-1, verbose=-1)
            clf_ex.fit(X_h, y_extreme)

            from src.bs_models import HorizonModels
            self.models[h] = HorizonModels(reg, q10, q90, clf_hi, clf_ex)
        return self

t0 = time.time()
m_exp3_temp = BasinshiftModelExp3(n_estimators=40)
m_exp3_temp.fit(train_sub, static_df, thr_val, fit_end_date="2006-09-30")
p_temp_exp3 = m_exp3_temp.predict(val_df[INPUT_COLS], fit_df, static_df, thr_val)
m_temp_exp3 = evaluate_predictions(p_temp_exp3, val_df, thr_val)

m_exp3_spat = BasinshiftModelExp3(n_estimators=40)
m_exp3_spat.fit(train_fold_df, static_df, thr_val)
p_spat_exp3 = m_exp3_spat.predict(val_fold_df[INPUT_COLS], train_fold_df, static_df, thr_val)
m_spat_exp3 = evaluate_predictions(p_spat_exp3, val_fold_df, thr_val)
t_exp3 = time.time() - t0
print(f"Exp 3 done in {t_exp3:.1f}s")

# ==============================================================================
# Print Comparison Tables
# ==============================================================================
print("\n" + "="*80)
print("EXPERIMENT COMPARISON RESULTS")
print("="*80)

print("\n--- 1. NSE Median (Forecast Accuracy) ---")
print(f"{'Configuration':<25} | {'Temp h=1':<9} {'Temp h=3':<9} {'Temp h=7':<9} | {'Spat h=1':<9} {'Spat h=3':<9} {'Spat h=7':<9}")
print(f"{'Baseline':<25} | {m_temp_base['nse_median'][1]:<9.4f} {m_temp_base['nse_median'][3]:<9.4f} {m_temp_base['nse_median'][7]:<9.4f} | {m_spat_base['nse_median'][1]:<9.4f} {m_spat_base['nse_median'][3]:<9.4f} {m_spat_base['nse_median'][7]:<9.4f}")
print(f"{'Exp 1 (Regul h=7)':<25} | {m_temp_exp1['nse_median'][1]:<9.4f} {m_temp_exp1['nse_median'][3]:<9.4f} {m_temp_exp1['nse_median'][7]:<9.4f} | {m_spat_exp1['nse_median'][1]:<9.4f} {m_spat_exp1['nse_median'][3]:<9.4f} {m_spat_exp1['nse_median'][7]:<9.4f}")

print("\n--- 2. Extreme Flow Brier Skill Score (BSS) ---")
print(f"{'Configuration':<25} | {'Temp h=1':<9} {'Temp h=3':<9} {'Temp h=7':<9} | {'Spat h=1':<9} {'Spat h=3':<9} {'Spat h=7':<9}")
print(f"{'Baseline':<25} | {m_temp_base['brier_skill_extreme'][1]:<9.4f} {m_temp_base['brier_skill_extreme'][3]:<9.4f} {m_temp_base['brier_skill_extreme'][7]:<9.4f} | {m_spat_base['brier_skill_extreme'][1]:<9.4f} {m_spat_base['brier_skill_extreme'][3]:<9.4f} {m_spat_base['brier_skill_extreme'][7]:<9.4f}")
print(f"{'Exp 2 (Prob Shrinkage)':<25} | {m_temp_exp2['brier_skill_extreme'][1]:<9.4f} {m_temp_exp2['brier_skill_extreme'][3]:<9.4f} {m_temp_exp2['brier_skill_extreme'][7]:<9.4f} | {m_spat_base['brier_skill_extreme'][1]:<9.4f} {m_spat_exp2['brier_skill_extreme'][3]:<9.4f} {m_spat_exp2['brier_skill_extreme'][7]:<9.4f}")

print("\n--- 3. Interval Coverage & Width ---")
print(f"{'Configuration':<25} | {'Cov h=1':<9} {'Width h=1':<9} | {'Cov h=7':<9} {'Width h=7':<9}")
print(f"{'Baseline (Temp)':<25} | {m_temp_base['interval_coverage'][1]:<9.4f} {m_temp_base['interval_width'][1]:<9.4f} | {m_temp_base['interval_coverage'][7]:<9.4f} {m_temp_base['interval_width'][7]:<9.4f}")
print(f"{'Exp 3 (Calibrated Temp)':<25} | {m_temp_exp3['interval_coverage'][1]:<9.4f} {m_temp_exp3['interval_width'][1]:<9.4f} | {m_temp_exp3['interval_coverage'][7]:<9.4f} {m_temp_base['interval_width'][7]:<9.4f}")
print(f"{'Baseline (Spat)':<25} | {m_spat_base['interval_coverage'][1]:<9.4f} {m_spat_base['interval_width'][1]:<9.4f} | {m_spat_base['interval_coverage'][7]:<9.4f} {m_spat_base['interval_width'][7]:<9.4f}")
print(f"{'Exp 3 (Calibrated Spat)':<25} | {m_spat_exp3['interval_coverage'][1]:<9.4f} {m_spat_exp3['interval_width'][1]:<9.4f} | {m_spat_exp3['interval_coverage'][7]:<9.4f} {m_spat_base['interval_width'][7]:<9.4f}")

print("\n--- 4. Extreme Flow PR-AUC and ROC-AUC under Exp 2 (Protection Check) ---")
print(f"Base Temp h=3 PR-AUC: {m_temp_base['auprc_extreme'][3]:.4f}, Exp 2: {m_temp_exp2['auprc_extreme'][3]:.4f}")
print(f"Base Temp h=7 PR-AUC: {m_temp_base['auprc_extreme'][7]:.4f}, Exp 2: {m_temp_exp2['auprc_extreme'][7]:.4f}")
print(f"Base Spat h=3 PR-AUC: {m_spat_base['auprc_extreme'][3]:.4f}, Exp 2: {m_spat_exp2['auprc_extreme'][3]:.4f}")
print(f"Base Spat h=7 PR-AUC: {m_spat_base['auprc_extreme'][7]:.4f}, Exp 2: {m_spat_exp2['auprc_extreme'][7]:.4f}")

