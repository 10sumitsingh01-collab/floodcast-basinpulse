"""Build the website's data from logged runs (no numbers are typed by hand).

    python web/build_data.py --run outputs/final_T0 --baseline outputs/persistence_baseline

Writes web/static/data/: summary.json (headline results), rivers.json (the 60 unseen test rivers with skill and
novelty), river/<id>.json (forecast vs observed around each river's largest flood, with ranges, flood probability,
persistence and the drivers of each alert). Uses the test-set predictions of a finished run and the model it saved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import evaluate  # noqa: E402
from src.bs_models import BasinshiftModel  # noqa: E402
from src.explain import explain  # noqa: E402

ID, DATE = "gauge_id", "date"
OUT = ROOT / "web" / "static" / "data"


def r(x, n=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else round(x, n)


def metric(m: pd.DataFrame, ts: str, h: int, k: str):
    s = m[(m.test_set == ts) & (m.horizon == h) & (m.metric == k)]
    return None if s.empty else {"value": r(s.value.iloc[0]), "lo": r(s.ci_low.iloc[0]), "hi": r(s.ci_high.iloc[0])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=ROOT / "outputs" / "final_T0")
    ap.add_argument("--baseline", type=Path, default=ROOT / "outputs" / "persistence_baseline")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "caravan")
    ap.add_argument("--extra", type=Path, default=ROOT / "data" / "caravan_extra")
    ap.add_argument("--caravan", type=Path, default=Path.home() / "BasinPulse-data" / "caravan")
    ap.add_argument("--days", type=int, default=150)
    a = ap.parse_args()
    (OUT / "river").mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(a.run / "metrics.csv")
    b = pd.read_csv(a.baseline / "metrics.csv")
    keys = ["nse_median", "kge_median", "mae_pooled", "auprc_high", "auprc_extreme", "brier_skill_high",
            "precision_high", "recall_high", "false_alarms_per_basin_year_high", "interval_coverage", "interval_width"]
    summary = {"model": {}, "persistence": {}}
    for name, df in (("model", m), ("persistence", b)):
        for ts in ("geo_temporal", "geo", "temporal"):
            summary[name][ts] = {k: {str(h): metric(df, ts, h, k) for h in (1, 3, 7)} for k in keys}
    eu_path = ROOT / "outputs" / "transfer_lamah" / "metrics.csv"
    if eu_path.exists():
        eu = pd.read_csv(eu_path)
        summary["europe"] = {mod: {k: {str(h): r(eu[(eu.model == mod) & (eu.horizon == h) & (eu.metric == k)].value.iloc[0])
                                       for h in (1, 3, 7)} for k in ("nse_median", "auprc_high", "auprc_extreme",
                                                                     "interval_coverage")}
                             for mod in eu.model.unique()}
        summary["europe_basins"] = int(eu.n_basins.iloc[0])
    lab = pd.read_parquet(a.data / "test_labels.parquet")
    summary["counts"] = {"unseen_rivers": int(lab.loc[lab.test_set.str.startswith("geo"), ID].nunique()),
                         "seen_rivers": int(lab.loc[lab.test_set == "temporal", ID].nunique()),
                         "forecasts": int(3 * len(pd.read_parquet(a.data / "test.parquet", columns=[ID])))}
    (OUT / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    # ---- the 60 unseen rivers --------------------------------------------------------------------------------
    thr = pd.read_csv(a.data / "thresholds.csv", dtype={ID: str})
    static = pd.read_parquet(a.data / "static.parquet")
    train = pd.read_parquet(a.data / "train.parquet")
    if (a.extra / "train.parquet").exists():
        static = pd.concat([static, pd.read_parquet(a.extra / "static.parquet")], ignore_index=True)
        train_ids = set(train[ID]) | set(pd.read_parquet(a.extra / "train.parquet", columns=[ID])[ID])
    else:
        train_ids = set(train[ID])
    nov_model = BasinshiftModel(shift_guard=True)
    nov_model._remember_basins(sorted(train_ids), set(), [], static)       # reference = all training basins
    g = lab[lab.test_set == "geo_temporal"]
    rivers = sorted(g[ID].unique())
    novelty = nov_model.novelty(static, rivers)
    names = pd.read_csv(a.caravan / "attributes" / "camels" / "attributes_other_camels.csv",
                        dtype={ID: str}).set_index(ID)["gauge_name"] if (a.caravan / "attributes").exists() else {}
    preds = pd.read_parquet(a.run / "predictions.parquet")
    preds = preds[preds[ID].isin(rivers)]
    base = pd.read_parquet(a.baseline / "predictions.parquet", columns=[ID, DATE, "horizon", "predicted_discharge"])
    base = base[base[ID].isin(rivers)]
    j = evaluate.join(preds, g, thr)
    per = {h: evaluate.basin_metrics(j[j.horizon == h]) for h in (1, 3, 7)}
    jb = evaluate.join(base.assign(lower=0.0, upper=0.0, p_high=0.0, p_extreme=0.0), g, thr)
    perb = {h: evaluate.basin_metrics(jb[jb.horizon == h]) for h in (1, 3, 7)}
    st = static.drop_duplicates(ID).set_index(ID)
    model = joblib.load(a.run / "model.joblib")
    test = pd.read_parquet(a.data / "test.parquet")
    test = test[test[ID].isin(rivers)]
    edges = np.quantile(novelty.to_numpy(), [1 / 3, 2 / 3])
    out = []
    for gid in rivers:
        row = {"id": gid, "name": str(names.get(gid, gid)), "lat": r(st.loc[gid, "gauge_lat"]),
               "lon": r(st.loc[gid, "gauge_lon"]), "area_km2": r(st.loc[gid, "area_km2"], 0),
               "aridity": r(st.loc[gid, "aridity"], 2), "snow": r(st.loc[gid, "frac_snow"], 2),
               "novelty": r(novelty[gid], 2),
               "novelty_group": ["familiar", "new", "very new"][int(np.searchsorted(edges, novelty[gid], side="right"))],
               "q_high": r(thr.set_index(ID).loc[gid, "q_high"]),
               "nse": {str(h): r(per[h]["nse"].get(gid)) for h in (1, 3, 7)},
               "nse_persistence": {str(h): r(perb[h]["nse"].get(gid)) for h in (1, 3, 7)}}
        out.append(row)
        # detail: window around the largest observed flow in the test period (flow on t+1 = h1 label at t)
        lg = g[g[ID] == gid].sort_values(DATE)
        peak = lg.loc[lg.streamflow_h1.idxmax(), DATE] + pd.Timedelta(days=1)
        days = pd.date_range(peak - pd.Timedelta(days=int(a.days * 0.7)), peak + pd.Timedelta(days=int(a.days * 0.3)))
        obs = lg.assign(day=lg[DATE] + pd.Timedelta(days=1)).set_index("day")["streamflow_h1"].reindex(days)
        detail = {"id": gid, "name": row["name"], "q_high": row["q_high"], "q_extreme": r(thr.set_index(ID).loc[gid, "q_extreme"]),
                  "dates": [d.strftime("%Y-%m-%d") for d in days], "observed": [r(v) for v in obs]}
        # only the plotted window plus a year of look-back is needed for features and explanations
        win = test[(test[ID] == gid) & (test[DATE] >= days[0] - pd.Timedelta(days=410))
                   & (test[DATE] <= days[-1])].reset_index(drop=True)
        hist = train[(train[ID] == gid) & (train[DATE] >= days[0] - pd.Timedelta(days=420))]
        for h in (1, 3, 7):
            ph = preds[(preds[ID] == gid) & (preds.horizon == h)].assign(day=lambda x: x[DATE] + pd.Timedelta(days=h))
            ph = ph.set_index("day").reindex(days)
            bh = base[(base[ID] == gid) & (base.horizon == h)].assign(day=lambda x: x[DATE] + pd.Timedelta(days=h))
            bh = bh.set_index("day").reindex(days)
            ex = explain(model, win, hist, static, thr, horizon=h)
            ex = ex.assign(day=ex[DATE] + pd.Timedelta(days=h)).set_index("day").reindex(days)
            alert = ph["p_high"].to_numpy(float) >= 0.5
            drivers = []
            for i in range(len(days)):
                ok = alert[i] and isinstance(ex["driver_1"].iloc[i], str)
                drivers.append([[ex[f"driver_{k}"].iloc[i], r(ex[f"share_{k}"].iloc[i], 2)] for k in (1, 2, 3)]
                               if ok else None)
            detail[f"h{h}"] = {"forecast": [r(v) for v in ph["predicted_discharge"]],
                               "lower": [r(v) for v in ph["lower"]], "upper": [r(v) for v in ph["upper"]],
                               "p_high": [r(v) for v in ph["p_high"]],
                               "persistence": [r(v) for v in bh["predicted_discharge"]], "drivers": drivers}
        (OUT / "river" / f"{gid}.json").write_text(json.dumps(detail), encoding="utf-8")
        print(f"{gid} done", flush=True)
    (OUT / "rivers.json").write_text(json.dumps(out), encoding="utf-8")
    # sample upload for "try the model": two unseen rivers, model inputs only (no targets), 2012-10 to 2013-12
    ss = sorted(out, key=lambda x: -(x["nse"]["3"] or -9))[:2]
    sample = test[test[ID].isin([x["id"] for x in ss]) & (test[DATE] >= "2012-10-01") & (test[DATE] <= "2013-12-31")]
    sample.to_csv(OUT / "sample_inputs.csv", index=False)
    print(f"wrote {OUT}: summary, {len(out)} rivers")


if __name__ == "__main__":
    main()
