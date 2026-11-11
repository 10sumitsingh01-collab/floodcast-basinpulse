import pandas as pd
import numpy as np
from src.bs_models import BasinshiftModel, HORIZONS
from src.bs_features import INPUT_COLS

print("Loading data...")
train = pd.read_parquet("data/caravan/train.parquet")
static = pd.read_parquet("data/caravan/static.parquet")
thr = pd.read_csv("data/caravan/thresholds_val.csv")

sel = train["gauge_id"].unique()[:40]
train40 = train[train["gauge_id"].isin(sel)].copy()

m = BasinshiftModel(n_estimators=40, num_leaves=31)
m.fit(train40, static, thr, fit_end_date="2006-09-30")

print("\n=== Feature Importances per Horizon ===")
for h in HORIZONS:
    reg = m.models[h].regressor
    imp = pd.Series(reg.feature_importances_, index=m.feature_cols).sort_values(ascending=False)
    print(f"\n--- Horizon h={h} (Top 10 features) ---")
    for feat, val in imp.head(10).items():
        print(f"  {feat:<35}: {val}")

# Validation analysis
val = train40[pd.to_datetime(train40["date"]) >= pd.to_datetime("2006-10-01")].copy()
fit_df = train40[pd.to_datetime(train40["date"]) <= pd.to_datetime("2006-09-30")].copy()
preds = m.predict(val[INPUT_COLS], fit_df, static, thr)

print("\n=== Probability Calibration & Brier Decomposition ===")
q_extreme_dict = thr.set_index("gauge_id")["q_extreme"].to_dict()
gids = val["gauge_id"].astype(str).values
for h in HORIZONS:
    p_h = preds[preds["horizon"] == h].merge(val[["gauge_id", "date", f"streamflow_h{h}"]], on=["gauge_id", "date"])
    yt = p_h[f"streamflow_h{h}"].values
    pe = p_h["p_extreme"].values
    qe = np.array([q_extreme_dict.get(g, np.nan) for g in p_h["gauge_id"].values])
    y_ext = (yt > qe).astype(int)
    
    base_rate = np.mean(y_ext)
    pred_mean = np.mean(pe)
    bs = np.mean((pe - y_ext)**2)
    bs_ref = 0.01 * 0.99
    bss = 1.0 - bs / bs_ref
    print(f"h={h}: Base rate={base_rate:.4f}, Mean predicted p_extreme={pred_mean:.4f}, Brier={bs:.5f}, Ref={bs_ref:.5f}, BSS={bss:.4f}")

print("\n=== Lower vs Upper Interval Coverage Analysis ===")
for h in HORIZONS:
    p_h = preds[preds["horizon"] == h].merge(val[["gauge_id", "date", f"streamflow_h{h}"]], on=["gauge_id", "date"])
    yt = p_h[f"streamflow_h{h}"].values
    lo = p_h["lower"].values
    hi = p_h["upper"].values
    
    under = np.mean(yt < lo)
    over = np.mean(yt > hi)
    total_cov = np.mean((yt >= lo) & (yt <= hi))
    print(f"h={h}: Under lower bound={under:.4f} (target 0.10), Over upper bound={over:.4f} (target 0.10), Total coverage={total_cov:.4f} (target 0.80)")

