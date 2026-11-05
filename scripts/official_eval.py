"""Judge-style evaluation, following the BASINSHIFT specification (sections 7, 8.4, 10 and 14.2).

    python scripts/official_eval.py                     # full run (about 15-30 min on a laptop)
    python scripts/official_eval.py --skip-predict      # reuse the hidden-run submission already written

What it does, in the order the specification's evaluator would (14.2):
  1. Hidden run. Hands the test INPUTS ONLY (no labels) to the command-line predictor `predict.py`, exactly as an
     organizer would, with the benchmark model (`outputs/final_T0/model.joblib`: trained without the 60 unseen
     basins and without 2008-2014). Records wall time, peak memory and artifact size (10.1 efficiency).
  2. Validation. Re-reads the written JSON Lines submission and checks it against the 8.4 rules.
  3. Scoring. Joins the labels only now, and computes every section 10.1 metric family with 95% basin-bootstrap
     intervals and event counts (10.4), next to persistence and climatology (9.1).
  4. Generalization (10.1, 10.4). Share of unseen basins where the model beats persistence, skill by region,
     and the familiar-vs-unseen gap (H2).
  5. Robustness (7.5). Six fixed corruption scenarios on the unseen-basin test period, for the model and for
     persistence; relative degradation.
  6. Machine score (10.2). The specification proposes weights but leaves the normalization unfrozen (10.3), so
     this script fixes its own formula BELOW, before any result is read, and applies it to the model and to
     persistence alike. It is our stand-in for the official formula, not the official score.

Writes docs/OFFICIAL_STYLE_EVALUATION.md and docs/results/official_eval_*.csv.

NOTE: `models/basinpulse_final.joblib` (the submitted model) was refitted on ALL basins and years including the
test set, so it must never be scored on this test set; this script refuses it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import baselines, evaluate, stress, submission  # noqa: E402

ID, DATE = "gauge_id", "date"
HEADLINE = "geo_temporal"
HS = (1, 3, 7)
TEST_START = pd.Timestamp("2008-10-01")
WARMUP_START = pd.Timestamp("2008-03-01")   # > 150 days of look-back before the test period

# ---------------------------------------------------------------------------------------------------------------
# Section 10.2 machine score: OUR pre-declared normalization (fixed before reading results; see module docstring).
# skill(m, b, best) = clip((m - b) / (best - b), 0, 1): 0 = no better than persistence, 1 = perfect.
#   Forecast skill        25  headline set, mean over horizons of [NSE skill + MAE skill] / 2
#   High-flow detection   25  headline set, mean over horizons of [AUPRC high skill + AUPRC extreme skill
#                             + recall-at-10-alert-days-per-year (high) skill] / 3
#   OOD generalization    20  10 x share of unseen basins beating persistence (NSE, mean over horizons)
#                             + 10 x mean over horizons of clip(NSE unseen / NSE familiar, 0, 1)
#   Calibration           15  7.5 x mean over horizons of clip(Brier skill score high, 0, 1)
#                             + 7.5 x mean over horizons of clip(1 - |coverage - 0.80| / 0.10, 0, 1)
#   Robustness            10  10 x mean over the 6 scenarios of clip(stressed / clean median NSE at 1 day, 0, 1)
#   Efficiency             5  1 point each: CPU only; no network; artifact < 100 MB; peak memory < 8 GB;
#                             whole hidden test predicted in < 30 min
# ---------------------------------------------------------------------------------------------------------------
WEIGHTS = {"Forecast skill": 25, "High-flow event detection": 25, "Out-of-distribution generalization": 20,
           "Calibration and uncertainty": 15, "Robustness": 10, "Efficiency": 5}
BUDGET = {"artifact_mb": 100, "memory_gb": 8, "minutes": 30}


def skill(m: float, b: float, best: float) -> float:
    if not np.isfinite(m) or not np.isfinite(b) or best == b:
        return 0.0
    return float(np.clip((m - b) / (best - b), 0, 1))


def hidden_run(model: Path, data: Path, work: Path) -> dict:
    inputs = pd.read_parquet(data / "test.parquet")
    leaked = [c for c in inputs.columns if c.startswith("streamflow_h") or c == "test_set"]
    assert not leaked, f"test inputs contain label columns: {leaked}"
    cmd = [sys.executable, str(ROOT / "predict.py"), "--inputs", str(data / "test.parquet"),
           "--history", str(data / "train.parquet"), "--static", str(data / "static.parquet"),
           "--thresholds", str(data / "thresholds.csv"), "--model", str(model),
           "--out", str(work / "submission.jsonl.gz")]
    t0 = time.time()
    proc = psutil.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    peak = 0
    while proc.poll() is None:
        try:
            peak = max(peak, proc.memory_info().rss + sum(c.memory_info().rss for c in proc.children(True)))
        except psutil.Error:
            pass
        time.sleep(0.5)
    out = proc.stdout.read()
    if proc.returncode != 0:
        raise SystemExit(f"predict.py failed:\n{out}")
    run = {"seconds": round(time.time() - t0, 1), "peak_memory_gb": round(peak / 1e9, 2),
           "artifact_mb": round(model.stat().st_size / 1e6, 1), "origins": int(len(inputs)),
           "basins": int(inputs[ID].nunique()), "predict_output": out.strip().splitlines()[-1],
           "command": " ".join(["python"] + [Path(c).relative_to(ROOT).as_posix() if Path(c).is_absolute()
                                              and Path(c).resolve().is_relative_to(ROOT) else c for c in cmd[1:]])}
    (work / "hidden_run.json").write_text(json.dumps(run, indent=2))
    return run


def per_basin_nse(preds: pd.DataFrame, labels: pd.DataFrame, thr: pd.DataFrame, ts: str) -> pd.DataFrame:
    df = evaluate.join(preds, labels[labels.test_set == ts], thr)
    return pd.concat({h: evaluate.basin_metrics(g)["nse"] for h, g in df.groupby(evaluate.HORIZON)}, axis=1)


def load_region():
    spec = importlib.util.spec_from_file_location("war", ROOT / "scripts" / "warning_and_regions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.region


def stress_slice(model, data: Path, labels, static, thr, seed: int = 42) -> pd.DataFrame:
    """Six fixed corruption scenarios (7.5) on the unseen basins' test period, model and persistence."""
    test = pd.read_parquet(data / "test.parquet")
    gt = labels[labels.test_set == HEADLINE]
    basins = gt[ID].unique()
    rows = test[test[ID].isin(basins) & (test[DATE] >= WARMUP_START)].reset_index(drop=True)
    train = pd.read_parquet(data / "train.parquet")
    pers = baselines.fit(train, static, thr)
    out = []
    for sc in ["clean"] + list(stress.SCENARIOS):
        r = rows if sc == "clean" else stress.perturb(rows, sc, seed=seed)
        hist, cur = r[r[DATE] < TEST_START].reset_index(drop=True), r[r[DATE] >= TEST_START].reset_index(drop=True)
        for name, fn in (("BasinPulse", lambda: model.predict(model, cur, hist, static, thr)),
                         ("persistence", lambda: baselines.predict(pers, cur, hist, static, thr))):
            m, _ = evaluate.score(fn(), gt, thr, n_boot=0)
            m = m.set_index(["horizon", "metric"])["value"]
            for h in HS:
                out.append({"model": name, "scenario": sc, "horizon": h, "nse_median": m[(h, "nse_median")],
                            "auprc_high": m[(h, "auprc_high")], "interval_coverage": m[(h, "interval_coverage")]})
        print(f"stress {sc}: done", flush=True)
    return pd.DataFrame(out)


def machine_score(M, P, win, gap, stress_df, run, who: str) -> dict:
    """M, P: metric lookups (test_set, horizon, metric) -> value for the scored model and for persistence."""
    g = lambda D, h, k, ts=HEADLINE: float(D.get((ts, h, k), np.nan))  # noqa: E731
    f = np.mean([(skill(g(M, h, "nse_median"), g(P, h, "nse_median"), 1)
                  + skill(-g(M, h, "mae_basin_mean"), -g(P, h, "mae_basin_mean"), 0)) / 2 for h in HS])
    e = np.mean([(skill(g(M, h, "auprc_high"), g(P, h, "auprc_high"), 1)
                  + skill(g(M, h, "auprc_extreme"), g(P, h, "auprc_extreme"), 1)
                  + skill(g(M, h, "recall_at_budget10_high"), g(P, h, "recall_at_budget10_high"), 1)) / 3
                 for h in HS])
    o = 0.5 * np.mean(win) + 0.5 * np.mean(gap)
    c = 0.5 * np.mean([np.clip(g(M, h, "brier_skill_high"), 0, 1) for h in HS]) + 0.5 * np.mean(
        [np.clip(1 - abs(g(M, h, "interval_coverage") - 0.8) / 0.1, 0, 1) for h in HS])
    s = stress_df[(stress_df.model == who) & (stress_df.horizon == 1)].set_index("scenario")["nse_median"]
    r = float(np.mean([np.clip(s[k] / s["clean"], 0, 1) for k in s.index if k != "clean"]))
    eff = (1 + 1 + (run["artifact_mb"] < BUDGET["artifact_mb"]) + (run["peak_memory_gb"] < BUDGET["memory_gb"])
           + (run["seconds"] < 60 * BUDGET["minutes"])) / 5 if run else 1.0
    parts = dict(zip(WEIGHTS, [f, e, o, c, r, eff]))
    return {k: round(WEIGHTS[k] * v, 1) for k, v in parts.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path, default=ROOT / "outputs" / "final_T0" / "model.joblib")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "caravan")
    ap.add_argument("--work", type=Path, default=ROOT / "outputs" / "official_eval")
    ap.add_argument("--skip-predict", action="store_true")
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args()
    if a.model.resolve() == (ROOT / "models" / "basinpulse_final.joblib").resolve():
        raise SystemExit("models/basinpulse_final.joblib was trained on the test set; use the benchmark model.")
    a.work.mkdir(parents=True, exist_ok=True)
    res = ROOT / "docs" / "results"

    # 1-2. hidden run + 8.4 validation of what was written ------------------------------------------------------
    run = (json.loads((a.work / "hidden_run.json").read_text()) if a.skip_predict
           else hidden_run(a.model, a.data, a.work))
    print("hidden run:", run, flush=True)
    keys = pd.read_parquet(a.data / "test.parquet", columns=[ID, DATE]).drop_duplicates()
    preds = submission.read_jsonl(a.work / "submission.jsonl.gz")
    problems = submission.validate(preds, keys)

    # 3. scoring: labels joined only now ------------------------------------------------------------------------
    labels = pd.read_parquet(a.data / "test_labels.parquet")
    thr = pd.read_csv(a.data / "thresholds.csv", dtype={ID: str})
    static = pd.read_parquet(a.data / "static.parquet")
    metrics, rel = evaluate.score(preds, labels, thr, n_boot=a.n_boot)
    metrics.to_csv(res / "official_eval_metrics.csv", index=False)
    logged = pd.read_csv(ROOT / "outputs" / "final_T0" / "metrics.csv")
    chk = metrics.merge(logged, on=["test_set", "horizon", "metric"], suffixes=("", "_logged"))
    same = float((chk["value"] - chk["value_logged"]).abs().max())
    M = metrics.set_index(["test_set", "horizon", "metric"])["value"].to_dict()
    P = pd.read_csv(ROOT / "outputs" / "persistence_baseline" / "metrics.csv").set_index(
        ["test_set", "horizon", "metric"])["value"].to_dict()
    C = pd.read_csv(ROOT / "outputs" / "climatology_baseline" / "metrics.csv").set_index(
        ["test_set", "horizon", "metric"])["value"].to_dict()

    # 4. generalization -------------------------------------------------------------------------------------------
    pers_preds = pd.read_parquet(ROOT / "outputs" / "persistence_baseline" / "predictions.parquet")
    bm, bp = per_basin_nse(preds, labels, thr, HEADLINE), per_basin_nse(pers_preds, labels, thr, HEADLINE)
    win = [float((bm[h] > bp[h].reindex(bm.index)).mean()) for h in HS]
    gap = [float(np.clip(M[(HEADLINE, h, "nse_median")] / M[("temporal", h, "nse_median")], 0, 1)) for h in HS]
    gap_p = [float(np.clip(P[(HEADLINE, h, "nse_median")] / P[("temporal", h, "nse_median")], 0, 1))
             if P[("temporal", h, "nse_median")] > 0 else 0.0 for h in HS]
    region = load_region()
    reg = pd.DataFrame({"region": [region(x) for x in bm.index], "nse_3d": bm[3].values,
                        "beats_persistence_3d": (bm[3] > bp[3].reindex(bm.index)).values})
    reg = reg.groupby("region").agg(basins=("nse_3d", "size"), nse_3d_median=("nse_3d", "median"),
                                    share_beating_persistence=("beats_persistence_3d", "mean")).reset_index()

    # 5. robustness ------------------------------------------------------------------------------------------------
    import joblib
    model = joblib.load(a.model)
    sfile = res / "official_eval_stress.csv"
    stress_df = stress_slice(model, a.data, labels, static, thr)
    stress_df.to_csv(sfile, index=False)

    # 6. machine score -----------------------------------------------------------------------------------------------
    ours = machine_score(M, P, win, gap, stress_df, run, "BasinPulse")
    base = machine_score(P, P, [0.0] * 3, gap_p, stress_df, None, "persistence")
    pd.DataFrame({"component": list(WEIGHTS), "weight": list(WEIGHTS.values()),
                  "BasinPulse": list(ours.values()), "persistence": list(base.values())}).to_csv(
        res / "official_eval_score.csv", index=False)

    write_doc(run, problems, same, M, P, C, win, reg, gap, stress_df, ours, base, len(bm))
    print(f"machine score: BasinPulse {sum(ours.values()):.1f} / 100, persistence {sum(base.values()):.1f} / 100")
    return 0


def write_doc(run, problems, same, M, P, C, win, reg, gap, st, ours, base, n_unseen) -> None:
    g = lambda D, h, k, ts=HEADLINE: D.get((ts, h, k), np.nan)  # noqa: E731
    f2 = lambda v: "n/a" if not np.isfinite(v) else f"{v:.2f}"  # noqa: E731
    L = ["# Judge-style evaluation (BASINSHIFT specification)", "",
         "Generated by `scripts/official_eval.py`. It runs the test the way the specification's evaluator would "
         "(section 14.2): test inputs only → `predict.py` → 8.4 validation → labels joined → section 10 metrics. "
         "Model: the benchmark model `outputs/final_T0/model.joblib` (`src.tuning:T0`), trained without the 60 "
         "unseen basins and without 2008–2014. All data are real Caravan / CAMELS-US observations.", "",
         "## 1. Hidden run (14.2 steps 3–5)", "",
         "| | |", "|---|---|",
         f"| Command | `{run['command']}` |",
         f"| Forecast origins | {run['origins']:,} in {run['basins']} basins (temporal + geo + geo_temporal) |",
         f"| Wall time | {run['seconds'] / 60:.1f} min on a laptop CPU, no GPU, no network |",
         f"| Peak memory | {run['peak_memory_gb']:.1f} GB |",
         f"| Model artifact | {run['artifact_mb']:.1f} MB |",
         f"| 8.4 validation of the written file | {'passed, 0 problems' if not problems else '; '.join(problems)} |",
         f"| Same as the logged benchmark run | largest metric difference {same:.1e} |", "",
         f"## 2. Accuracy on {n_unseen} basins never seen in training, 2008–2014 (`{HEADLINE}`)", "",
         "Median over basins where marked; 95% basin-bootstrap intervals in brackets; event counts in the last rows.",
         "", "| Metric (10.1) | 1 day | 3 days | 7 days | Persistence 1 / 3 / 7 | Climatology 1 / 3 / 7 |",
         "|---|---|---|---|---|---|"]
    met = pd.read_csv(ROOT / "docs" / "results" / "official_eval_metrics.csv").set_index(
        ["test_set", "horizon", "metric"])
    rows = [("NSE, median", "nse_median"), ("KGE, median", "kge_median"), ("MAE, mm/day", "mae_basin_mean"),
            ("High-flow AUPRC", "auprc_high"), ("Extreme-flow AUPRC", "auprc_extreme"),
            ("Recall at 10 alert-days / basin-year (high)", "recall_at_budget10_high"),
            ("Precision at 10 alert-days / basin-year (high)", "precision_at_budget10_high"),
            ("Recall at p ≥ 0.5 (high)", "recall_high"), ("Precision at p ≥ 0.5 (high)", "precision_high"),
            ("False alarms / basin-year (high)", "false_alarms_per_basin_year_high"),
            ("Brier skill score (high)", "brier_skill_high"), ("Brier skill score (extreme)", "brier_skill_extreme"),
            ("10–90% range coverage (target 0.80)", "interval_coverage"),
            ("10–90% range width, mm/day", "interval_width"), ("Pinball loss", "pinball_loss")]
    for label, k in rows:
        cells = []
        for h in HS:
            r = met.loc[(HEADLINE, h, k)]
            ci = f" [{r.ci_low:.2f}, {r.ci_high:.2f}]" if np.isfinite(r.ci_low) else ""
            cells.append(f"**{r.value:.2f}**{ci}")
        L.append(f"| {label} | " + " | ".join(cells) + " | " + " / ".join(f2(g(P, h, k)) for h in HS)
                 + " | " + " / ".join(f2(g(C, h, k)) for h in HS) + " |")
    ev = [met.loc[(HEADLINE, 1, k)] for k in ("auprc_high", "auprc_extreme")]
    L += [f"| Events in the test period (high / extreme) | {int(ev[0].n_events):,} / {int(ev[1].n_events):,} "
          f"of {int(ev[0].n_rows):,} basin-days | | | | |", "",
          "Other test sets, median NSE 1 / 3 / 7 days: familiar basins in later years (`temporal`) "
          + " / ".join(f"{g(M, h, 'nse_median', 'temporal'):.2f}" for h in HS)
          + "; unseen basins in the training years (`geo`) "
          + " / ".join(f"{g(M, h, 'nse_median', 'geo'):.2f}" for h in HS) + ".", "",
          "## 3. Generalization to unseen basins (10.1, 10.4, H2)", "",
          "| | 1 day | 3 days | 7 days |", "|---|---|---|---|",
          "| Share of unseen basins where BasinPulse beats persistence (NSE) | "
          + " | ".join(f"{w:.0%}" for w in win) + " |",
          "| Unseen ÷ familiar median NSE (1.00 = no generalization gap) | "
          + " | ".join(f"{x:.2f}" for x in gap) + " |", "",
          "| Region (USGS) | Unseen basins | Median NSE, 3 days | Share beating persistence, 3 days |",
          "|---|---|---|---|"]
    L += [f"| {r.region} | {r.basins} | {r.nse_3d_median:.2f} | {r.share_beating_persistence:.0%} |"
          for r in reg.itertuples()]
    L += ["", "## 4. Robustness: fixed corruption scenarios (7.5), unseen basins 2008–2014", "",
          "Median NSE at 1 / 3 / 7 days; change vs clean inputs in brackets (1-day).", "",
          "| Scenario | BasinPulse | Persistence |", "|---|---|---|"]
    for sc in ["clean"] + list(stress.SCENARIOS):
        cell = []
        for who in ("BasinPulse", "persistence"):
            d = st[(st.model == who) & (st.scenario == sc)].set_index("horizon")
            c0 = st[(st.model == who) & (st.scenario == "clean") & (st.horizon == 1)]["nse_median"].iloc[0]
            delta = "" if sc == "clean" else f" ({d.loc[1, 'nse_median'] - c0:+.2f})"
            cell.append(" / ".join(f"{d.loc[h, 'nse_median']:.2f}" for h in HS) + delta)
        name = sc if sc == "clean" else f"{sc}: {stress.SCENARIOS[sc]}"
        L.append(f"| {name} | {cell[0]} | {cell[1]} |")
    L += ["", "## 5. Section 10.2 machine score (our pre-declared normalization)", "",
          "The specification gives the weights but says the normalization must still be frozen (10.3), so the "
          "official number can only be computed by the organizers. We fixed our own formula in "
          "`scripts/official_eval.py` before reading any result: every skill term is 0 at persistence level and "
          "full marks only for a perfect forecast, so this is a strict scale; persistence is scored with the "
          "same formula for reference.", "",
          "| Component | Weight | BasinPulse | Persistence |", "|---|---|---|---|"]
    L += [f"| {k} | {w} | {ours[k]:.1f} | {base[k]:.1f} |" for k, w in WEIGHTS.items()]
    L += [f"| **Total** | **100** | **{sum(ours.values()):.1f}** | **{sum(base.values()):.1f}** |", "",
          "## 6. Hypotheses (4.3) and deliverables (12)", "",
          "- H1 (catchment attributes help transfer), H3 (probabilistic outputs), H4 (robust training): validation "
          "ablations in [EXPERIMENTS.md](EXPERIMENTS.md) and [FEATURE_IMPORTANCE.md](FEATURE_IMPORTANCE.md).",
          "- H2 (geographic hold-outs reveal gaps): section 3 above.",
          "- Deliverables 1–10: model `models/basinpulse_final.joblib` + `predict.py`; source and pinned "
          "`requirements.txt` + `Dockerfile`; [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md); "
          "[DATA_PROVENANCE.md](DATA_PROVENANCE.md); this file and `docs/results/`; README reproduction guide; "
          "[MODEL_CARD.md](MODEL_CARD.md); website `web/`; [TEAM_DECLARATION.md](TEAM_DECLARATION.md).", "",
          "Responsible use (16.1): research benchmark output, never the sole basis for evacuation, emergency "
          "response, infrastructure operation or public warnings."]
    (ROOT / "docs" / "OFFICIAL_STYLE_EVALUATION.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
