"""Figures for the report and slides, from logged runs only (real outputs, nothing invented).

    python scripts/make_figures.py --run outputs/lgbm_final --baseline outputs/persistence_baseline

Writes docs/figures/: hydrographs.png, skill_by_horizon.png, reliability.png, stress.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ID, DATE = "gauge_id", "date"
BLUE, RED, GREY, ORANGE = "#4C78A8", "#E45756", "#9D9D9D", "#F58518"


def hydrographs(run: Path, data: Path, out: Path, base: Path | None = None) -> None:
    """Three unseen basins, 1-day forecast vs observed flow over one wet season of the test period."""
    p = pd.read_parquet(run / "predictions.parquet")
    bp = None
    if base is not None and (base / "predictions.parquet").exists() and base.resolve() != run.resolve():
        bp = pd.read_parquet(base / "predictions.parquet", columns=[ID, DATE, "horizon", "predicted_discharge"])
        bp = bp[bp.horizon == 1].rename(columns={"predicted_discharge": "persistence"}).drop(columns="horizon")
    lab = pd.read_parquet(data / "test_labels.parquet")
    thr = pd.read_csv(data / "thresholds.csv", dtype={ID: str}).set_index(ID)
    g = lab[lab.test_set == "geo_temporal"]
    # pick basins with the most high-flow days at h=1, so the figure shows floods (deterministic choice)
    ev = g.assign(e=g.streamflow_h1 > g[ID].map(thr.q_high)).groupby(ID).e.sum().sort_values(ascending=False)
    picks = list(ev.index[:3])
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), dpi=150, sharex=False)
    for ax, gid in zip(axes, picks):
        a = p[(p[ID] == gid) & (p.horizon == 1)].merge(g[[ID, DATE, "streamflow_h1"]], on=[ID, DATE])
        a = a.sort_values(DATE)
        peak = a.loc[a.streamflow_h1.idxmax(), DATE]
        w = a[(a[DATE] >= peak - pd.Timedelta(days=120)) & (a[DATE] <= peak + pd.Timedelta(days=60))]
        x = w[DATE] + pd.Timedelta(days=1)   # forecast for day t+1
        ax.fill_between(x, w["lower"], w["upper"], color=BLUE, alpha=0.2, label="10–90% range")
        ax.plot(x, w["streamflow_h1"], color="black", lw=1.2, label="observed")
        if bp is not None:
            wb = w[[ID, DATE]].merge(bp, on=[ID, DATE], how="left")
            ax.plot(x, wb["persistence"].to_numpy(), color=GREY, lw=1.0, ls="--", label="persistence")
        ax.plot(x, w["predicted_discharge"], color=BLUE, lw=1.2, label="forecast (1 day ahead)")
        ax.axhline(thr.loc[gid, "q_high"], color=RED, ls="--", lw=0.8, label="high-flow threshold")
        ax2 = ax.twinx()
        ax2.plot(x, w["p_high"], color=ORANGE, lw=0.8, alpha=0.8, label="P(high flow)")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("P(high)", color=ORANGE)
        ax.set_ylabel("flow (mm/day)")
        ax.set_title(f"{gid} (never seen in training)", fontsize=10)
    h1, l1 = axes[0].get_legend_handles_labels()
    axes[0].legend(h1, l1, loc="upper left", fontsize=8)
    fig.suptitle("1-day forecasts on unseen basins, test period (geo_temporal)")
    fig.tight_layout()
    fig.savefig(out / "hydrographs.png")
    plt.close(fig)


def skill(run: Path, base: Path, out: Path) -> None:
    m = pd.read_csv(run / "metrics.csv")
    b = pd.read_csv(base / "metrics.csv")
    panels = [("nse_median", "NSE (median over basins)"), ("auprc_high", "High-flow AUPRC"),
              ("auprc_extreme", "Extreme-flow AUPRC"), ("interval_width", "10–90% range width (mm/day)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), dpi=150)
    for ax, (k, title) in zip(axes, panels):
        for df, name, color, dx in ((b, "persistence", GREY, -0.18), (m, "BasinPulse", BLUE, 0.18)):
            s = df[(df.test_set == "geo_temporal") & (df.metric == k)].sort_values("horizon")
            x = np.arange(len(s)) + dx
            ax.bar(x, s.value, width=0.34, color=color, label=name)
            if s.ci_low.notna().all():
                ax.errorbar(x, s.value, yerr=[s.value - s.ci_low, s.ci_high - s.value], fmt="none",
                            ecolor="black", lw=0.8, capsize=2)
        ax.set_xticks(range(3), ["1 day", "3 days", "7 days"])
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color="black", lw=0.5)
    axes[0].legend(fontsize=8)
    fig.suptitle("Unseen basins, test period: BasinPulse vs persistence (95% basin-bootstrap CIs)")
    fig.tight_layout()
    fig.savefig(out / "skill_by_horizon.png")
    plt.close(fig)


def reliability(run: Path, out: Path) -> None:
    r = pd.read_csv(run / "reliability_high.csv")
    fig, ax = plt.subplots(figsize=(4.8, 4.6), dpi=150)
    for h, c in ((1, BLUE), (3, ORANGE), (7, RED)):
        s = r[r["slice"] == f"geo_temporal_h{h}"]
        ax.plot(s.mean_predicted, s.observed_frequency, "o-", color=c, ms=4, label=f"{h} day")
    ax.plot([0, 1], [0, 1], color="black", lw=0.7, ls="--", label="perfect")
    ax.set_xlabel("forecast probability of high flow")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability, unseen basins (test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "reliability.png")
    plt.close(fig)


def stress(run: Path, out: Path) -> None:
    s = pd.read_csv(run / "stress.csv")
    s = s[(s.test_set == "geo_temporal") & (s.horizon == 1) & (s.metric.isin(["nse_median", "auprc_high"]))]
    piv = s.pivot(index="scenario", columns="metric", values="relative_change") * 100
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    y = np.arange(len(piv))
    ax.barh(y - 0.18, piv["nse_median"], height=0.34, color=BLUE, label="NSE")
    ax.barh(y + 0.18, piv["auprc_high"], height=0.34, color=ORANGE, label="High-flow AUPRC")
    ax.set_yticks(y, piv.index)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("change vs clean inputs (%)")
    ax.set_title("Robustness: unseen basins, 1-day horizon")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "stress.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, default=Path("outputs/persistence_baseline"))
    ap.add_argument("--data", type=Path, default=Path("data/caravan"))
    ap.add_argument("--out", type=Path, default=Path("docs/figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    hydrographs(a.run, a.data, a.out, a.baseline)
    skill(a.run, a.baseline, a.out)
    reliability(a.run, a.out)
    if (a.run / "stress.csv").exists():
        stress(a.run, a.out)
    print(f"figures written to {a.out}")


if __name__ == "__main__":
    main()
