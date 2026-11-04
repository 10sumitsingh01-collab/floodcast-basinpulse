"""BASINSHIFT end-to-end run (Lane D): data -> model -> submission -> validation -> scores -> stress -> report.

    python basinshift_run.py                                  # reference persistence model
    python basinshift_run.py --model src.models:BasinshiftModel   # Lane C's model, once it exists
    python basinshift_run.py --no-stress --n-boot 50          # faster check

A model module/class must provide
    fit(train, static, thresholds) -> fitted model
    predict(model, test, train, static, thresholds) -> long predictions (see src/submission.py)
as module-level functions (e.g. src.baselines) or as methods of a class "module:Class".

Data: data/caravan/ (unzipped Lane A package). Outputs go to outputs/<run_name>/ (gitignored).
Only this script and src/evaluate.py read test_labels.parquet, and only after predictions exist.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import evaluate, stress, submission  # noqa: E402

ID, DATE = "gauge_id", "date"


def git_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return (r.stdout.strip() or "unknown") + ("-dirty" if dirty else "")


def load_model(spec: str):
    mod_name, _, cls = spec.partition(":")
    mod = importlib.import_module(mod_name)
    if cls:
        obj = getattr(mod, cls)()
        return obj.fit, obj.predict
    return mod.fit, mod.predict


def peak_memory_mb() -> float | None:
    try:
        import psutil  # optional
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except ImportError:
        return None


def check_package(data: Path) -> list[str]:
    """Verify file hashes against the package manifest, so everyone scores the same data."""
    man = json.loads((data / "prep_manifest.json").read_text())
    bad = []
    for name, h in man["sha256"].items():
        f = data / name
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != h:
            bad.append(name)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "caravan")
    ap.add_argument("--model", default="src.baselines")
    ap.add_argument("--name", default=None, help="run name (default: model name)")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--no-stress", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    name = a.name or a.model.replace(":", "_").replace(".", "_")
    out = ROOT / "outputs" / name
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    bad = check_package(a.data)
    if bad:
        print(f"WARNING: these package files do not match prep_manifest.json hashes: {bad}")
    train = pd.read_parquet(a.data / "train.parquet")
    test = pd.read_parquet(a.data / "test.parquet")
    static = pd.read_parquet(a.data / "static.parquet")
    thr = pd.read_csv(a.data / "thresholds.csv", dtype={ID: str})
    fit, predict = load_model(a.model)

    # ---- fit + predict (no labels loaded yet) -----------------------------------------
    t0 = time.time()
    model = fit(train, static, thr)
    t_fit = time.time() - t0
    t0 = time.time()
    preds = predict(model, test, train, static, thr)
    t_pred = time.time() - t0
    preds[ID] = preds[ID].astype(str)

    # ---- submission: validate long table, write 8.3 JSON, re-read and validate the file -------
    submission.assert_valid(preds, test[[ID, DATE]])
    sub_path = submission.write_jsonl(preds, out / "submission.jsonl.gz")
    back = submission.read_jsonl(sub_path)
    submission.assert_valid(back, test[[ID, DATE]])
    preds.to_parquet(out / "predictions.parquet", index=False)
    print(f"submission valid: {len(back):,} predictions, {sub_path.stat().st_size / 1e6:.1f} MB")

    # ---- scoring (labels loaded only now) ----------------------------------------------
    labels = pd.read_parquet(a.data / "test_labels.parquet")
    metrics, rel = evaluate.score(preds, labels, thr, n_boot=a.n_boot, seed=a.seed)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.concat([r.assign(slice=k) for k, r in rel.items()]).to_csv(out / "reliability_high.csv", index=False)

    # ---- stress tests: same model, corrupted test inputs --------------------------------
    stress_rows = []
    if not a.no_stress:
        key = ["nse_median", "mae_pooled", "auprc_high", "brier_high", "interval_coverage"]
        clean = metrics[metrics.metric.isin(key)].set_index(["test_set", "horizon", "metric"])["value"]
        for sc, desc in stress.SCENARIOS.items():
            pt = stress.perturb(test, sc, seed=a.seed)
            ps = predict(model, pt, train, static, thr)
            ps[ID] = ps[ID].astype(str)
            submission.assert_valid(ps, test[[ID, DATE]])
            m, _ = evaluate.score(ps, labels, thr, n_boot=0, seed=a.seed)
            m = m[m.metric.isin(key)].set_index(["test_set", "horizon", "metric"])["value"]
            d = pd.DataFrame({"clean": clean, "stressed": m}).reset_index()
            d["scenario"] = sc
            d["relative_change"] = (d["stressed"] - d["clean"]) / d["clean"].abs()
            stress_rows.append(d)
            print(f"stress {sc}: done")
        pd.concat(stress_rows).to_csv(out / "stress.csv", index=False)

    run = {
        "run": name, "model": a.model, "commit": git_commit(),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_manifest_ok": not bad, "seed": a.seed, "n_boot": a.n_boot,
        "fit_seconds": round(t_fit, 1), "predict_seconds": round(t_pred, 1),
        "total_seconds": round(time.time() - t_all, 1), "memory_mb": peak_memory_mb(),
        "python": platform.python_version(), "machine": platform.platform(),
        "n_train_rows": len(train), "n_test_origins": len(test),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    write_report(out, run, metrics, pd.concat(stress_rows) if stress_rows else None)
    log = ROOT / "experiments" / "basinshift_results.csv"
    log.parent.mkdir(exist_ok=True)
    head = metrics[(metrics.test_set == "geo_temporal") & (metrics.horizon == 1)].set_index("metric")["value"]
    row = pd.DataFrame([{**{k: run[k] for k in ("utc", "commit", "run", "model", "total_seconds")},
                         **{f"geo_temporal_h1_{k}": head.get(k) for k in
                            ("nse_median", "kge_median", "auprc_high", "brier_skill_high", "interval_coverage")}}])
    row.to_csv(log, mode="a", header=not log.exists(), index=False)
    print(f"\nReport: {out / 'REPORT.md'}")
    return 0


def write_report(out: Path, run: dict, metrics: pd.DataFrame, stress_df: pd.DataFrame | None) -> None:
    show = ["nse_median", "kge_median", "mae_pooled", "auprc_high", "auprc_extreme", "brier_skill_high",
            "recall_high", "precision_high", "false_alarms_per_basin_year_high", "interval_coverage",
            "interval_width"]

    def fmt(r):
        ci = "" if pd.isna(r.ci_low) else f" [{r.ci_low:.3f}, {r.ci_high:.3f}]"
        return f"{r.value:.3f}{ci}"

    lines = [f"# BASINSHIFT evaluation: `{run['run']}`", "",
             f"Model `{run['model']}`, commit `{run['commit']}`, {run['utc']}. "
             f"Fit {run['fit_seconds']} s, predict {run['predict_seconds']} s, total {run['total_seconds']} s.", "",
             "Values with 95% bootstrap CIs (resampling basins). Event counts are horizon-specific.", ""]
    for ts in ["temporal", "geo", "geo_temporal"]:
        sub = metrics[metrics.test_set == ts]
        if sub.empty:
            continue
        nb = int(sub.n_basins.iloc[0])
        lines += [f"## {ts} ({nb} basins)", "", "| metric | h=1 | h=3 | h=7 |", "|---|---|---|---|"]
        for m in show:
            cells = []
            for h in (1, 3, 7):
                r = sub[(sub.metric == m) & (sub.horizon == h)]
                cells.append(fmt(r.iloc[0]) if len(r) else "")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        ev = sub[sub.metric == "auprc_high"].set_index("horizon")["n_events"]
        lines += ["", f"High-flow events: " + ", ".join(f"h={h}: {int(v):,}" for h, v in ev.items()), ""]
    if stress_df is not None:
        s = stress_df[(stress_df.test_set == "geo_temporal") & (stress_df.horizon == 1)]
        piv = s.pivot(index="scenario", columns="metric", values="relative_change")
        lines += ["## Stress tests (geo_temporal, h=1): relative change vs clean", "",
                  "| scenario | " + " | ".join(piv.columns) + " |", "|---|" + "---|" * len(piv.columns)]
        for sc, r in piv.iterrows():
            lines.append(f"| {sc} | " + " | ".join(f"{v:+.1%}" for v in r) + " |")
        lines += ["", "Scenarios: " + "; ".join(f"`{k}` = {v}" for k, v in stress.SCENARIOS.items()), ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
