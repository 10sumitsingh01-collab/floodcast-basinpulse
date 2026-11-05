"""Train the SUBMITTED model on all prepared real data and save it for predict.py.

    python train_final.py --model src.tuning:T0 --data data/final_camels [data/final_lamah] \\
        --out models/basinpulse_final.joblib

The configuration (model class and settings) is chosen on validation beforehand (docs/EXPERIMENTS.md); this
script only refits it on every basin and year in the given training packages and records what it used.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from basinshift_run import git_commit, peak_memory_mb  # noqa: E402

ID = "gauge_id"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="module:Class chosen on validation")
    ap.add_argument("--data", type=Path, nargs="+", required=True, help="training packages (train/static/thresholds)")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "basinpulse_final.joblib")
    ap.add_argument("--kw", default="{}", help="extra settings as JSON, for example {\"shift_guard\": true}")
    a = ap.parse_args()
    t0 = time.time()
    tr, st, th = [], [], []
    for d in a.data:
        tr.append(pd.read_parquet(d / "train.parquet"))
        st.append(pd.read_parquet(d / "static.parquet"))
        th.append(pd.read_csv(d / "thresholds.csv", dtype={ID: str}))
    train = pd.concat(tr, ignore_index=True)
    static = pd.concat(st, ignore_index=True)
    thr = pd.concat(th, ignore_index=True)
    del tr
    mod, _, cls = a.model.partition(":")
    model = getattr(importlib.import_module(mod), cls)(**json.loads(a.kw))
    print(f"training {a.model} on {train[ID].nunique()} basins, {len(train):,} rows", flush=True)
    model.fit(train, static, thr)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, a.out, compress=3)
    info = {
        "model": a.model, "settings": json.loads(a.kw), "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit(), "training_packages": [str(d) for d in a.data],
        "n_basins": int(train[ID].nunique()), "n_rows": len(train),
        "period": [str(train["date"].min().date()), str(train["date"].max().date())],
        "fit_seconds": round(time.time() - t0, 1), "peak_memory_mb": peak_memory_mb(),
        "model_mb": round(a.out.stat().st_size / 1e6, 2),
        "model_sha256": hashlib.sha256(a.out.read_bytes()).hexdigest(),
        "python": platform.python_version(),
    }
    a.out.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
