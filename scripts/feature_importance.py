"""Which inputs matter for UNSEEN basins? Group permutation importance on validation (training data only).

    python scripts/feature_importance.py --fold 0

Fits the final model on the training basins outside one station fold (splits.json), then forecasts the held-out
fold's basins. For each input group, the group's columns are shuffled together across rows (breaking their link
to the outcome while keeping their distribution) and the drop in skill is measured. Repeated with 2 seeds.
Also reports LightGBM split-gain importance summed by group. Writes docs/FEATURE_IMPORTANCE.md and a figure.
Never reads test.parquet or test labels.
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
from src.bs_models import FLOW_FEATURES, RobustMeanModel, build_features  # noqa: E402

ID, DATE = "gauge_id", "date"
TARGETS = ["streamflow_h1", "streamflow_h3", "streamflow_h7"]


def groups(cols: list[str], static_cols: set[str]) -> dict[str, list[str]]:
    g = {
        "Flow history (observed)": [c for c in cols if c in FLOW_FEATURES],
        "Rain": [c for c in cols if c.startswith(("p0", "p_lag", "p_sum", "p_max", "api"))],
        "Snow": [c for c in cols if c.startswith("swe")],
        "Soil water": [c for c in cols if c.startswith("sm")],
        "Temperature, energy, evaporation": [c for c in cols if c.startswith(("t0", "tmax", "t_mean", "rad", "dew",
                                                                               "pet_sum", "p_minus_pet"))],
        "Season": ["doy_sin", "doy_cos"],
        "Catchment attributes": [c for c in cols if c in static_cols or c == "log_area"],
        "Flow thresholds (scale)": [c for c in cols if c == "log_q_high"],
    }
    used = {c for v in g.values() for c in v}
    missing = [c for c in cols if c not in used]
    assert not missing, f"features without a group: {missing}"
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "caravan")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    train = pd.read_parquet(a.data / "train.parquet")
    static = pd.read_parquet(a.data / "static.parquet")
    thr = pd.read_csv(a.data / "thresholds.csv", dtype={ID: str})
    folds = json.loads((a.data / "splits.json").read_text())["station_folds"]
    held = {g for g, f in folds.items() if f == a.fold}
    tr, va = train[~train[ID].isin(held)], train[train[ID].isin(held)].reset_index(drop=True)
    inputs = [c for c in train.columns if c not in ["streamflow"] + TARGETS]

    model = RobustMeanModel().fit(tr, static, thr)
    feats = build_features(va[inputs], static, thr)
    X = feats[model.feature_cols].to_numpy(np.float32)
    anchor = model._anchor(feats)
    ids, dates = va[ID].astype(str).to_numpy(), va[DATE].to_numpy()
    labels = va[[ID, DATE] + TARGETS].assign(test_set="val_unseen_basins")
    keys = ["nse_median", "auprc_high", "interval_coverage"]

    def score(Xm):
        m, _ = evaluate.score(model.predict_from_features(Xm, anchor, ids, dates), labels, thr, n_boot=0)
        return m[m.metric.isin(keys)].set_index(["metric", "horizon"])["value"]

    base = score(X)
    G = groups(model.feature_cols, set(static.columns) - {ID})
    col_idx = {c: i for i, c in enumerate(model.feature_cols)}
    rows = []
    for name, cols in G.items():
        drops = []
        for s in range(a.seeds):
            rng = np.random.default_rng(100 + s)
            Xp = X.copy()
            perm = rng.permutation(len(Xp))
            idx = [col_idx[c] for c in cols]
            Xp[:, idx] = X[perm][:, idx]          # shuffle the group's columns jointly across rows
            drops.append(base - score(Xp))
        d = pd.concat(drops, axis=1).mean(axis=1)
        rows.append({"group": name, "n_features": len(cols),
                     **{f"dNSE_h{h}": d[("nse_median", h)] for h in (1, 3, 7)},
                     **{f"dAUPRC_h{h}": d[("auprc_high", h)] for h in (1, 3, 7)}})
        print(f"{name}: dNSE {d[('nse_median', 1)]:.3f} / {d[('nse_median', 3)]:.3f} / {d[('nse_median', 7)]:.3f}",
              flush=True)
    res = pd.DataFrame(rows).sort_values("dNSE_h3", ascending=False)

    gain = {}
    for (h, q), reg in model.reg.items():
        if q != "mean":
            continue
        imp = pd.Series(reg.booster_.feature_importance("gain"), index=model.feature_cols)
        gain[h] = {g: float(imp[c].sum() / imp.sum()) for g, c in G.items()}
    res["gain_share_h3"] = res["group"].map(gain[3])
    out = ROOT / "docs"
    res.to_csv(out / "results" / "feature_importance.csv", index=False)

    lines = ["# Which inputs matter for unseen basins?", "",
             f"Group permutation importance on validation: final model fitted without station fold {a.fold}, "
             f"evaluated on its {len(held)} held-out basins (1981–2008, training data only). Each group's columns are "
             f"shuffled together across rows; the table shows the **drop** in skill (higher = more important), "
             f"mean of {a.seeds} shuffles. `gain share` = share of the 3-day mean model's split gain. "
             "Generated by `scripts/feature_importance.py`.", "",
             f"Unshuffled: NSE {base[('nse_median', 1)]:.3f} / {base[('nse_median', 3)]:.3f} / "
             f"{base[('nse_median', 7)]:.3f}, high-flow AUPRC {base[('auprc_high', 1)]:.3f} / "
             f"{base[('auprc_high', 3)]:.3f} / {base[('auprc_high', 7)]:.3f} (1 / 3 / 7 days).", "",
             "| input group | features | NSE drop 1d | 3d | 7d | AUPRC drop 1d | 3d | 7d | gain share 3d |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        lines.append(f"| {r.group} | {r.n_features} | {r.dNSE_h1:+.3f} | {r.dNSE_h3:+.3f} | {r.dNSE_h7:+.3f} | "
                     f"{r.dAUPRC_h1:+.3f} | {r.dAUPRC_h3:+.3f} | {r.dAUPRC_h7:+.3f} | {r.gain_share_h3:.0%} |")
    lines.append("")
    (out / "FEATURE_IMPORTANCE.md").write_text("\n".join(lines), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    r2 = res.iloc[::-1]
    y = np.arange(len(r2))
    for k, (h, c) in enumerate(((1, "#4C78A8"), (3, "#F58518"), (7, "#E45756"))):
        ax.barh(y + (k - 1) * 0.26, r2[f"dNSE_h{h}"], height=0.25, color=c, label=f"{h} day")
    ax.set_yticks(y, r2.group)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("drop in median NSE when the group is shuffled (unseen validation basins)")
    ax.set_title("Which inputs matter for rivers the model never saw?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "feature_importance.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
