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
    """Peak resident memory of this process so far, in MB (Windows: peak working set; Linux/macOS: ru_maxrss)."""
    try:
        import psutil
        mi = psutil.Process(os.getpid()).memory_info()
        if hasattr(mi, "peak_wset"):
            return round(mi.peak_wset / 1e6, 1)
    except ImportError:
        pass
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / 1e6 if platform.system() == "Darwin" else r / 1e3, 1)
    except ImportError:
        return None


def save_model(model, path: Path) -> float | None:
    """Pickle the fitted model (joblib, compressed) and return its size in MB."""
    try:
        import joblib
        joblib.dump(model, path, compress=3)
        return round(path.stat().st_size / 1e6, 2)
    except Exception as e:  # a model that cannot be pickled still runs; size is then unknown
        print(f"model not saved: {e}")
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
    ap.add_argument("--validate", action="store_true",
                    help="score on splits.json validation (unseen-basin folds + last 2 train years); "
                         "never touches test.parquet or test labels")
    ap.add_argument("--folds", default="all", help="station folds to use with --validate, e.g. 0,1")
    ap.add_argument("--no-temporal", action="store_true",
                    help="with --validate: unseen-basin folds only (faster, for tuning)")
    ap.add_argument("--val-stress", action="store_true",
                    help="with --validate: also run the 5 stress scenarios on the temporal validation split")
    ap.add_argument("--extra-train", type=Path, default=None,
                    help="folder with extra TRAINING basins (train.parquet, static.parquet, thresholds.csv) that are in "
                         "no test set; added to training only (scripts/caravan/prepare_extra_train.py)")
    ap.add_argument("--score-only", action="store_true",
                    help="re-score outputs/<name>/predictions.parquet with the current metrics (no refit; "
                         "keeps the run's stress.csv and run.json)")
    a = ap.parse_args()
    name = a.name or a.model.replace(":", "_").replace(".", "_")
    out = ROOT / "outputs" / name
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    if a.validate:
        return run_validation(a, out)
    if a.score_only:
        test = pd.read_parquet(a.data / "test.parquet")
        thr = pd.read_csv(a.data / "thresholds.csv", dtype={ID: str})
        preds = pd.read_parquet(out / "predictions.parquet")
        submission.assert_valid(preds, test[[ID, DATE]])
        labels = pd.read_parquet(a.data / "test_labels.parquet")
        metrics, rel = evaluate.score(preds, labels, thr, n_boot=a.n_boot, seed=a.seed)
        metrics.to_csv(out / "metrics.csv", index=False)
        pd.concat([r.assign(slice=k) for k, r in rel.items()]).to_csv(out / "reliability_high.csv", index=False)
        run = json.loads((out / "run.json").read_text())
        st = pd.read_csv(out / "stress.csv") if (out / "stress.csv").exists() else None
        write_report(out, run, metrics, st)
        print(f"re-scored {name}: {out / 'REPORT.md'}")
        return 0

    bad = check_package(a.data)
    if bad:
        print(f"WARNING: these package files do not match prep_manifest.json hashes: {bad}")
    train = pd.read_parquet(a.data / "train.parquet")
    test = pd.read_parquet(a.data / "test.parquet")
    static = pd.read_parquet(a.data / "static.parquet")
    thr = pd.read_csv(a.data / "thresholds.csv", dtype={ID: str})
    train, static, thr = add_extra(a, train, static, thr)
    fit, predict = load_model(a.model)

    # ---- fit + predict (no labels loaded yet) -----------------------------------------
    t0 = time.time()
    model = fit(train, static, thr)
    t_fit = time.time() - t0
    t0 = time.time()
    preds = predict(model, test, train, static, thr)
    t_pred = time.time() - t0
    model_mb = save_model(model, out / "model.joblib")
    mem_after_fit = peak_memory_mb()
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
        "total_seconds": round(time.time() - t_all, 1), "memory_mb": peak_memory_mb(), "peak_mb_fit_predict": mem_after_fit, "model_mb": model_mb,
        "predictions_per_second": round(len(preds) / max(t_pred, 1e-9)),
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


TARGETS = ["streamflow_h1", "streamflow_h3", "streamflow_h7"]


def add_extra(a, train, static, thr):
    """Append extra training basins (never in a test set). Returns train, static, thresholds."""
    if not a.extra_train:
        return train, static, thr
    x = a.extra_train
    et = pd.read_parquet(x / "train.parquet")
    overlap = set(et[ID]) & (set(train[ID]) | set(pd.read_parquet(a.data / "test.parquet", columns=[ID])[ID]))
    if overlap:
        raise ValueError(f"extra training basins overlap the benchmark: {sorted(overlap)[:3]}")
    print(f"extra training basins: {et[ID].nunique()} ({len(et):,} rows)", flush=True)
    return (pd.concat([train, et[train.columns]], ignore_index=True),
            pd.concat([static, pd.read_parquet(x / "static.parquet")], ignore_index=True),
            pd.concat([thr, pd.read_csv(x / "thresholds.csv", dtype={ID: str})], ignore_index=True))
HORIZON_DAYS = {"streamflow_h1": 1, "streamflow_h3": 3, "streamflow_h7": 7}


def run_validation(a, out: Path) -> int:
    """Validation on TRAIN data only (splits.json), so models can be compared without the test set.

    val_unseen_basins: for each station fold, fit on the other folds, predict the held-out basins.
    val_temporal: fit on dates <= fit_end (targets reaching past fit_end removed), predict the last 2 years.
    """
    t_all = time.time()
    train = pd.read_parquet(a.data / "train.parquet")
    static = pd.read_parquet(a.data / "static.parquet")
    thr = pd.read_csv(a.data / "thresholds.csv", dtype={ID: str})
    thr_val = pd.read_csv(a.data / "thresholds_val.csv", dtype={ID: str})
    n_main = len(train)
    train, static, thr = add_extra(a, train, static, thr)
    extra_ids = set(train[ID].iloc[n_main:])
    if extra_ids:   # extra basins' thresholds for the fit period (same rule as thresholds_val)
        fe = pd.Timestamp(json.loads((a.data / "splits.json").read_text())["temporal_validation"]["fit_end_inclusive"])
        ex = train[train[ID].isin(extra_ids) & (train[DATE] <= fe)].dropna(subset=["streamflow"]).groupby(ID)["streamflow"]
        thr_val = pd.concat([thr_val, pd.DataFrame({"q_high": ex.quantile(0.95), "q_extreme": ex.quantile(0.99),
                                                    "n_obs": ex.size()}).reset_index()], ignore_index=True)
    splits = json.loads((a.data / "splits.json").read_text())
    fit, predict = load_model(a.model)
    inputs = [c for c in train.columns if c not in ["streamflow"] + TARGETS]
    preds_all, labels_all, timings = [], [], {}

    folds = splits["station_folds"]
    use = sorted(set(folds.values())) if a.folds == "all" else [int(x) for x in a.folds.split(",")]
    for k in use:
        t0 = time.time()
        held = {g for g, f in folds.items() if f == k}
        tr = train[~train[ID].isin(held)]
        va = train[train[ID].isin(held)]     # held-out fold basins (never extra basins)
        model = fit(tr, static, thr)
        p = predict(model, va[inputs], tr, static, thr)
        p[ID] = p[ID].astype(str)
        submission.assert_valid(p, va[[ID, DATE]])
        preds_all.append(p)
        labels_all.append(va[[ID, DATE] + TARGETS].assign(test_set="val_unseen_basins"))
        timings[f"fold_{k}_seconds"] = round(time.time() - t0, 1)
        print(f"fold {k}: {len(held)} basins, {time.time() - t0:.0f} s", flush=True)

    if a.no_temporal:
        m1, _ = evaluate.score(pd.concat(preds_all), pd.concat(labels_all), thr, n_boot=a.n_boot, seed=a.seed)
        m1.to_csv(out / "validation_metrics.csv", index=False)
        run = {"run": a.name or a.model, "model": a.model, "commit": git_commit(), "mode": "validation-folds-only",
               "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "folds": use, **timings,
               "fit_seconds": "see folds", "predict_seconds": "see folds",
               "total_seconds": round(time.time() - t_all, 1)}
        (out / "validation_run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        write_report(out, run, m1, None, sets=["val_unseen_basins"], fname="VALIDATION.md")
        print(f"\nValidation report: {out / 'VALIDATION.md'}")
        return 0

    t0 = time.time()
    fit_end = pd.Timestamp(splits["temporal_validation"]["fit_end_inclusive"])
    tr = train[train[DATE] <= fit_end].copy()
    for col, h in HORIZON_DAYS.items():  # a target on t+h after fit_end would be a validation label
        tr.loc[tr[DATE] + pd.Timedelta(days=h) > fit_end, col] = float("nan")
    va = train[(train[DATE] > fit_end) & ~train[ID].isin(extra_ids)]   # score the benchmark basins only
    model = fit(tr, static, thr_val)
    p = predict(model, va[inputs], tr, static, thr_val)
    p[ID] = p[ID].astype(str)
    submission.assert_valid(p, va[[ID, DATE]])
    timings["temporal_seconds"] = round(time.time() - t0, 1)
    print(f"temporal: {time.time() - t0:.0f} s", flush=True)
    va_labels = va[[ID, DATE] + TARGETS].assign(test_set="val_temporal")

    stress_df = None
    if a.val_stress:   # robustness on validation data: same fitted model, corrupted validation inputs
        key = ["nse_median", "mae_pooled", "auprc_high", "brier_high", "interval_coverage"]
        clean_m, _ = evaluate.score(p, va_labels, thr_val, n_boot=0, seed=a.seed)
        clean = clean_m[clean_m.metric.isin(key)].set_index(["test_set", "horizon", "metric"])["value"]
        rows = []
        for sc in stress.SCENARIOS:
            ps = predict(model, stress.perturb(va[inputs], sc, seed=a.seed), tr, static, thr_val)
            ps[ID] = ps[ID].astype(str)
            submission.assert_valid(ps, va[[ID, DATE]])
            m, _ = evaluate.score(ps, va_labels, thr_val, n_boot=0, seed=a.seed)
            m = m[m.metric.isin(key)].set_index(["test_set", "horizon", "metric"])["value"]
            d = pd.DataFrame({"clean": clean, "stressed": m}).reset_index()
            d["scenario"] = sc
            d["relative_change"] = (d["stressed"] - d["clean"]) / d["clean"].abs()
            rows.append(d)
            print(f"validation stress {sc}: done", flush=True)
        stress_df = pd.concat(rows)
        stress_df.to_csv(out / "validation_stress.csv", index=False)

    # score unseen-basin folds and temporal split separately (different thresholds files)
    m1, r1 = evaluate.score(pd.concat(preds_all), pd.concat(labels_all), thr, n_boot=a.n_boot, seed=a.seed)
    m2, r2 = evaluate.score(p, va_labels, thr_val, n_boot=a.n_boot, seed=a.seed)
    metrics = pd.concat([m1, m2], ignore_index=True)
    metrics.to_csv(out / "validation_metrics.csv", index=False)
    run = {"run": a.name or a.model, "model": a.model, "commit": git_commit(),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mode": "validation",
           "folds": use, "seed": a.seed, "n_boot": a.n_boot, **timings,
           "fit_seconds": "see folds", "predict_seconds": "see folds",
           "total_seconds": round(time.time() - t_all, 1)}
    (out / "validation_run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    write_report(out, run, metrics, stress_df, sets=["val_unseen_basins", "val_temporal"], fname="VALIDATION.md",
                 stress_slice="val_temporal")
    log = ROOT / "experiments" / "basinshift_validation.csv"
    log.parent.mkdir(exist_ok=True)
    v = metrics.set_index(["test_set", "horizon", "metric"])["value"]
    row = {k: run[k] for k in ("utc", "commit", "run", "model", "total_seconds")}
    row["folds"] = ",".join(map(str, use))
    for ts in ("val_unseen_basins", "val_temporal"):
        for h in (1, 3, 7):
            for mname in ("nse_median", "auprc_high", "brier_skill_high", "interval_coverage"):
                row[f"{ts}_h{h}_{mname}"] = v.get((ts, h, mname))
    pd.DataFrame([row]).to_csv(log, mode="a", header=not log.exists(), index=False)
    print(f"\nValidation report: {out / 'VALIDATION.md'}")
    return 0


def write_report(out: Path, run: dict, metrics: pd.DataFrame, stress_df: pd.DataFrame | None,
                 sets=("temporal", "geo", "geo_temporal"), fname: str = "REPORT.md",
                 stress_slice: str = "geo_temporal") -> None:
    show = ["nse_median", "kge_median", "mae_pooled", "auprc_high", "auprc_extreme", "brier_skill_high",
            "recall_high", "precision_high", "false_alarms_per_basin_year_high",
            "recall_at_budget10_high", "precision_at_budget10_high", "recall_at_budget20_high",
            "precision_at_budget20_high", "interval_coverage", "interval_width"]

    def fmt(r):
        ci = "" if pd.isna(r.ci_low) else f" [{r.ci_low:.3f}, {r.ci_high:.3f}]"
        return f"{r.value:.3f}{ci}"

    lines = [f"# BASINSHIFT evaluation: `{run['run']}`", "",
             f"Model `{run['model']}`, commit `{run['commit']}`, {run['utc']}. "
             f"Fit {run['fit_seconds']} s, predict {run['predict_seconds']} s, total {run['total_seconds']} s"
             + (f"; peak memory {run['peak_mb_fit_predict']:.0f} MB, model {run['model_mb']} MB, "
                f"{run['predictions_per_second']:,} predictions/s" if run.get("model_mb") else "") + ".", "",
             "Values with 95% bootstrap CIs (resampling basins). Event counts are horizon-specific.", ""]
    for ts in sets:
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
        s = stress_df[(stress_df.test_set == stress_slice) & (stress_df.horizon == 1)]
        piv = s.pivot(index="scenario", columns="metric", values="relative_change")
        lines += [f"## Stress tests ({stress_slice}, h=1): relative change vs clean", "",
                  "| scenario | " + " | ".join(piv.columns) + " |", "|---|" + "---|" * len(piv.columns)]
        for sc, r in piv.iterrows():
            lines.append(f"| {sc} | " + " | ".join(f"{v:+.1%}" for v in r) + " |")
        lines += ["", "Scenarios: " + "; ".join(f"`{k}` = {v}" for k, v in stress.SCENARIOS.items()), ""]
    (out / fname).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
