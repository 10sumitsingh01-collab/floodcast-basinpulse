"""Inference and Submission Generation for BASINSHIFT (src/bs_predict.py).

Produces validated long-format predictions and outputs submission files adhering to
the official BASINSHIFT 8.3 schema and 8.4 validation rules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd

from src.bs_models import BasinshiftModel, fit, predict
from src.submission import assert_valid, write_jsonl

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "caravan"


def generate_test_predictions(
    model: BasinshiftModel,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    static_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    out_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate long-format predictions for test_df and optionally write submission files."""
    preds_long = model.predict(test_df, train_df, static_df, thresholds_df)

    # Validate against BASINSHIFT 8.4 rules
    expected_keys = test_df[["gauge_id", "date"]].copy()
    assert_valid(preds_long, expected_keys)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Write predictions parquet
        preds_long.to_parquet(out_dir / "predictions.parquet", index=False)
        # Write gzipped JSONL submission
        write_jsonl(preds_long, out_dir / "submission.jsonl.gz")

    return preds_long

