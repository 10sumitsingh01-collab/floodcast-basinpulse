"""Does ShiftGuard keep the 10-90% range honest on unfamiliar basins? (validation + Europe; no US test labels)

    python scripts/shiftguard_eval.py --fold 0 --extra-train data/caravan_extra

Fits the same configuration with and without ShiftGuard on the training basins outside station fold `--fold`,
then forecasts (a) that fold's held-out basins (validation) and (b) the 579 Central European basins (independent
check). Reports range coverage and width per novelty group. Decision rule (fixed in advance): adopt ShiftGuard if,
on (a), the largest |coverage - 0.80| over the three novelty groups shrinks and the mean width grows by <= 10%.
Writes docs/SHIFTGUARD.md, docs/results/shiftguard.csv and docs/figures/shiftguard.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import evaluate  # noqa: E402
from src.tuning import ShiftGuardT0, T0  # noqa: E402

ID, DATE = "gauge_id", "date"
TARGETS = ["streamflow_h1", "streamflow_h3", "streamflow_h7"]
GROUPS = ["familiar", "new", "very new"]


def coverage_table(preds, labels, thr, nov, edges, name):
    j = evaluate.join(preds, labels, thr)
    j["group"] = np.searchsorted(edges, j[ID].map(nov).to_numpy(), side="right")
    j["covered"] = (j.y_true >= j["lower"]) & (j.y_true <= j["upper"])
    j["width"] = j["upper"] - j["lower"]
    t = j.groupby(["horizon", "group"]).agg(coverage=("covered", "mean"), width=("width", "mean"),
                                           basins=(ID, "nunique"), rows=(ID, "size")).reset_index()
    t["model"] = name
    t["group"] = t["group"].map(dict(enumerate(GROUPS)))
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "caravan")
    ap.add_argument("--extra-train", type=Path, default=ROOT / "data" / "caravan_extra")
    ap.add_argument("--transfer", type=Path, default=ROOT / "data" / "transfer_lamah")
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()
    train = pd.read_parquet(a.data / "train.parquet")
    static = pd.read_parquet(a.data / "static.parquet")
    thr = pd.read_csv(a.data / "thresholds.csv", dtype={ID: str})
    if a.extra_train and (a.extra_train / "train.parquet").exists():
        et = pd.read_parquet(a.extra_train / "train.parquet")
        train = pd.concat([train, et[train.columns]], ignore_index=True)
        static = pd.concat([static, pd.read_parquet(a.extra_train / "static.parquet")], ignore_index=True)
        thr = pd.concat([thr, pd.read_csv(a.extra_train / "thresholds.csv", dtype={ID: str})], ignore_index=True)
    folds = json.loads((a.data / "splits.json").read_text())["station_folds"]
    held = {g for g, f in folds.items() if f == a.fold}
    tr, va = train[~train[ID].isin(held)], train[train[ID].isin(held)].reset_index(drop=True)
    inputs = [c for c in train.columns if c not in ["streamflow"] + TARGETS]
    va_lab = va[[ID, DATE] + TARGETS].assign(test_set="val_unseen_basins")

    eu_test = pd.read_parquet(a.transfer / "test.parquet")
    eu_hist = pd.read_parquet(a.transfer / "history.parquet")
    eu_st = pd.read_parquet(a.transfer / "static.parquet")
    eu_thr = pd.read_csv(a.transfer / "thresholds.csv", dtype={ID: str})
    eu_lab = pd.read_parquet(a.transfer / "test_labels.parquet")
    st_all = pd.concat([static, eu_st], ignore_index=True)
    thr_all = pd.concat([thr, eu_thr], ignore_index=True)

    tables, nse = [], {}
    sg = ShiftGuardT0().fit(tr, static, thr)
    nov_va = sg.novelty(static, va[ID].unique())
    nov_eu = sg.novelty(st_all, eu_test[ID].unique())
    edges = sg._nov_edges
    for name, model in (("ShiftGuard", sg), ("standard", T0().fit(tr, static, thr))):
        pv = model.predict(model, va[inputs], tr, static, thr)
        tables.append(coverage_table(pv, va_lab, thr, nov_va, edges, name).assign(data="validation (US unseen)"))
        pe = model.predict(model, eu_test, eu_hist, st_all, thr_all)
        tables.append(coverage_table(pe, eu_lab, eu_thr, nov_eu, edges, name).assign(data="Europe (zero-shot)"))
        m, _ = evaluate.score(pv, va_lab, thr, n_boot=0)
        nse[name] = m[m.metric == "nse_median"].set_index("horizon")["value"].round(4).to_dict()
        print(f"{name} done", flush=True)
    T = pd.concat(tables, ignore_index=True)
    (ROOT / "docs" / "results").mkdir(exist_ok=True)
    T.to_csv(ROOT / "docs" / "results" / "shiftguard.csv", index=False)

    v = T[T.data.str.startswith("validation")]
    gap = v.assign(gap=(v.coverage - 0.8).abs()).groupby("model").gap.max()
    width = v.groupby("model").width.mean()
    adopt = bool(gap["ShiftGuard"] < gap["standard"] and width["ShiftGuard"] <= 1.10 * width["standard"])
    decision = {"adopt": adopt, "max_gap": gap.round(4).to_dict(), "mean_width": width.round(4).to_dict(),
                "nse_unseen": nse, "novelty_edges": [round(float(x), 3) for x in edges]}
    (ROOT / "docs" / "results" / "shiftguard_decision.json").write_text(json.dumps(decision, indent=2))

    lines = ["# ShiftGuard: forecasts that know when they are on unfamiliar ground", "",
             "Every basin gets a **novelty score**: the mean standardised distance from its catchment traits (climate "
             "indices, elevation, slope, land cover, soils, regulation, glaciers, area) to its 5 nearest training basins. "
             "ShiftGuard calibrates the 10–90% range separately for *familiar*, *new* and *very new* basins (terciles of "
             "the calibration basins' novelty), so ranges stay tight where the model knows the terrain and widen where it "
             "does not. Point forecasts and probabilities are unchanged. Generated by `scripts/shiftguard_eval.py`.", "",
             f"**Decision (rule fixed in advance): {'ADOPTED' if adopt else 'NOT adopted'}.** Largest coverage gap from "
             f"0.80 across novelty groups on unseen validation basins: standard {gap['standard']:.3f} → ShiftGuard "
             f"{gap['ShiftGuard']:.3f}; mean range width {width['standard']:.3f} → {width['ShiftGuard']:.3f} mm/day.", "",
             "| data | horizon | novelty group | basins | coverage standard | coverage ShiftGuard | width standard | width ShiftGuard |",
             "|---|---|---|---|---|---|---|---|"]
    piv = T.pivot_table(index=["data", "horizon", "group"], columns="model", values=["coverage", "width", "basins"],
                        aggfunc="first").reset_index()
    for _, r in piv.iterrows():
        lines.append(f"| {r[('data', '')]} | {r[('horizon', '')]} | {r[('group', '')]} | {int(r[('basins', 'standard')])} | "
                     f"{r[('coverage', 'standard')]:.3f} | {r[('coverage', 'ShiftGuard')]:.3f} | "
                     f"{r[('width', 'standard')]:.2f} | {r[('width', 'ShiftGuard')]:.2f} |")
    lines += ["", "Novelty group boundaries: " + ", ".join(f"{x:.2f}" for x in edges) +
              f". Median NSE on unseen validation basins (unchanged by design): {nse}.", ""]
    (ROOT / "docs" / "SHIFTGUARD.md").write_text("\n".join(lines), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150, sharey=True)
    for ax, data in zip(axes, ["validation (US unseen)", "Europe (zero-shot)"]):
        d = T[(T.data == data) & (T.horizon == 1)]
        x = np.arange(len(GROUPS))
        for k, (mname, col) in enumerate((("standard", "#9D9D9D"), ("ShiftGuard", "#4C78A8"))):
            s = d[d.model == mname].set_index("group").reindex(GROUPS)
            ax.bar(x + (k - 0.5) * 0.38, s.coverage, width=0.36, color=col, label=mname)
        ax.axhline(0.8, color="#E45756", ls="--", lw=1, label="target 0.80")
        ax.set_xticks(x, GROUPS)
        ax.set_ylim(0.5, 1.0)
        ax.set_title(f"{data}, 1 day ahead", fontsize=10)
    axes[0].set_ylabel("10–90% range coverage")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("ShiftGuard: range coverage by how unfamiliar the basin is")
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "figures" / "shiftguard.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
