"""Full-dataset training, prediction, validation, and submission artifact generation."""
import hashlib
import pickle
import time
import tracemalloc
from pathlib import Path
import numpy as np
import pandas as pd

from src.bs_models import BasinshiftModel
from src.bs_predict import generate_test_predictions
from src.submission import assert_valid, validate

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "caravan"
OUTPUT_DIR = ROOT_DIR / "outputs" / "final_submission"
MODEL_DIR = ROOT_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

tracemalloc.start()
t_total_start = time.time()

print("=" * 80)
print("BASINSHIFT PHASE 4: FULL TRAINING AND SUBMISSION PIPELINE")
print("=" * 80)

# 1. Load Data
print("\n[1/5] Loading official Caravan datasets...")
t0 = time.time()
train_df = pd.read_parquet(DATA_DIR / "train.parquet")
test_df = pd.read_parquet(DATA_DIR / "test.parquet")
static_df = pd.read_parquet(DATA_DIR / "static.parquet")
thr_df = pd.read_csv(DATA_DIR / "thresholds.csv")
print(f"Loaded in {time.time() - t0:.1f}s:")
print(f"  train.parquet : {len(train_df):,} rows across {train_df['gauge_id'].nunique()} basins")
print(f"  test.parquet  : {len(test_df):,} rows across {test_df['gauge_id'].nunique()} basins")
print(f"  static.parquet: {len(static_df):,} basins x {len(static_df.columns)} attributes")
print(f"  thresholds.csv: {len(thr_df):,} basins")

# Confirm counts
assert train_df["gauge_id"].nunique() == 239, "Expected 239 training basins"
assert test_df["gauge_id"].nunique() == 299, "Expected 299 test basins"
assert len(train_df) == 2357018, f"Expected 2,357,018 train rows, got {len(train_df):,}"
assert len(test_df) == 1246829, f"Expected 1,246,829 test rows, got {len(test_df):,}"

# 2. Train Model
print("\n[2/5] Training Baseline BasinshiftModel on complete training dataset...")
t_train_start = time.time()
# Baseline configuration: n_estimators=80, num_leaves=31, learning_rate=0.08, random_state=42
model = BasinshiftModel(n_estimators=80, num_leaves=31, learning_rate=0.08, random_state=42)
model.fit(train_df, static_df, thr_df)
t_train = time.time() - t_train_start
_, peak_train = tracemalloc.get_traced_memory()
print(f"Model fitted successfully in {t_train:.1f}s | Traced RAM: {peak_train / (1024**3):.2f} GB")

# 3. Save Model Artifact
print("\n[3/5] Saving model artifacts...")
model_path = MODEL_DIR / "basinshift_baseline.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
model_size_mb = model_path.stat().st_size / (1024**2)
print(f"Model saved to {model_path} ({model_size_mb:.2f} MB)")

# 4. Generate Final Test Predictions
print("\n[4/5] Generating test predictions across all 299 catchments...")
t_pred_start = time.time()
preds_long = generate_test_predictions(
    model=model,
    test_df=test_df,
    train_df=train_df,
    static_df=static_df,
    thresholds_df=thr_df,
    out_dir=OUTPUT_DIR,
)
t_pred = time.time() - t_pred_start
_, peak_all = tracemalloc.get_traced_memory()
print(f"Predictions generated and saved in {t_pred:.1f}s | Peak RAM: {peak_all / (1024**3):.2f} GB")

# 5. Validation and Verification
print("\n[5/5] Auditing submission integrity against official BASINSHIFT rules...")
pred_parquet_path = OUTPUT_DIR / "predictions.parquet"
submission_jsonl_path = OUTPUT_DIR / "submission.jsonl.gz"

# Verify files on disk
assert pred_parquet_path.exists(), "predictions.parquet does not exist"
assert submission_jsonl_path.exists(), "submission.jsonl.gz does not exist"

saved_preds = pd.read_parquet(pred_parquet_path)
errs = validate(saved_preds, test_df[["gauge_id", "date"]])
print(f"Submission validator errors: {errs}")
assert len(errs) == 0, f"Submission validation failed: {errs}"

# Checksums
def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

sha_parquet = compute_sha256(pred_parquet_path)
sha_jsonl = compute_sha256(submission_jsonl_path)
sha_model = compute_sha256(model_path)

total_elapsed = time.time() - t_total_start
final_current, final_peak = tracemalloc.get_traced_memory()

print("\n" + "=" * 80)
print("FINAL REPRODUCIBILITY AND EXECUTION SUMMARY")
print("=" * 80)
print(f"Total Rows in predictions.parquet : {len(saved_preds):,} (Expected: {len(test_df) * 3:,})")
print(f"Unique Basins Predicted           : {saved_preds['gauge_id'].nunique()} (239 seen + 60 unseen = 299)")
print(f"Forecast Horizons                 : {sorted(saved_preds['horizon'].unique())}")
print(f"Columns                           : {list(saved_preds.columns)}")
print(f"predictions.parquet Size          : {pred_parquet_path.stat().st_size / (1024**2):.2f} MB")
print(f"predictions.parquet SHA-256       : {sha_parquet}")
print(f"submission.jsonl.gz Size          : {submission_jsonl_path.stat().st_size / (1024**2):.2f} MB")
print(f"submission.jsonl.gz SHA-256       : {sha_jsonl}")
print(f"Model Artifact SHA-256            : {sha_model}")
print(f"Model Fit Runtime                 : {t_train:.1f}s")
print(f"Prediction & Submission Runtime   : {t_pred:.1f}s")
print(f"Total Pipeline Runtime            : {total_elapsed:.1f}s")
print(f"Peak Traced RAM                   : {final_peak / (1024**3):.2f} GB (Ceiling: 7.0 GB)")
print("=" * 80)
