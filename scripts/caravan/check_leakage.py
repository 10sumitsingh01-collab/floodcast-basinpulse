"""Leakage audit for the BASINSHIFT Caravan data package (BASINSHIFT 7.2, 7.6, 17 "Split integrity").

    python scripts/check_leakage.py --package ~/BasinPulse-data/package/caravan

Re-derives every rule from the files on disk instead of trusting the code that wrote them.
Writes LEAKAGE_REPORT.md into the package and exits 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_caravan import HORIZONS, ID, DATE, OBS, Y, climate_indices  # noqa: E402

BANNED_STATIC = ("dis_", "run_", "riv_tc", "ria_ha")
HCOLS = [f"{Y}_h{h}" for h in HORIZONS]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", type=Path, default=Path.home() / "BasinPulse-data" / "package" / "caravan")
    p = ap.parse_args().package
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    train = pd.read_parquet(p / "train.parquet")
    test = pd.read_parquet(p / "test.parquet")
    labels = pd.read_parquet(p / "test_labels.parquet")
    static = pd.read_parquet(p / "static.parquet")
    splits = json.loads((p / "splits.json").read_text())
    prep = json.loads((p / "prep_manifest.json").read_text())
    test_start = pd.Timestamp(splits["test_period"][0])
    train_end = pd.Timestamp(splits["train_period"][1])
    fit_end = pd.Timestamp(splits["temporal_validation"]["fit_end_inclusive"])

    # 1. basin separation
    geo = set(labels.loc[labels["test_set"].str.startswith("geo"), ID])
    overlap = geo & set(train[ID])
    check("held-out basins never appear in train", not overlap, f"{len(geo)} held-out, overlap {len(overlap)}")
    folds = set(splits["station_folds"])
    check("basin folds contain train basins only", folds == set(train[ID]),
          f"{len(folds)} in folds, {train[ID].nunique()} train basins")

    # 2. time separation
    check("train rows end before the test period", train[DATE].max() < test_start,
          f"train max {train[DATE].max().date()}, test start {test_start.date()}")
    late = labels[labels["test_set"].isin(["temporal", "geo_temporal"])]
    check("temporal test origins are in the test period", late[DATE].min() >= test_start, f"min {late[DATE].min().date()}")
    dup = train[[ID, DATE]].merge(test[[ID, DATE]], on=[ID, DATE])
    check("no (basin, date) in both train and test", dup.empty, f"{len(dup)} rows")

    # 3. train targets: correct values and never from the test period
    flow = train[[ID, DATE, Y]]
    worst_date, mism = pd.Timestamp.min, 0
    for h, col in zip(HORIZONS, HCOLS):
        have = train.dropna(subset=[col])
        worst_date = max(worst_date, (have[DATE] + pd.Timedelta(days=h)).max())
        fut = flow.assign(**{DATE: flow[DATE] - pd.Timedelta(days=h)}).rename(columns={Y: "chk"})
        m = have[[ID, DATE, col]].merge(fut, on=[ID, DATE], how="left")
        mism += int((~np.isclose(m[col], m["chk"], equal_nan=False)).sum())
    check("train targets equal the flow h days later", mism == 0, f"{mism} mismatches")
    check("no train target comes from the test period", worst_date <= train_end,
          f"latest target date {worst_date.date()}, train end {train_end.date()}")

    # 4. what models see in test
    check("test inputs contain no target columns", not ({Y, *HCOLS} & set(test.columns)))
    tl = test.merge(labels, on=[ID, DATE])
    equal = [f"{c}~{hc}" for c in test.columns if c not in (ID, DATE) for hc in HCOLS
             if np.allclose(tl[c].to_numpy(float), tl[hc].to_numpy(float), equal_nan=False)]
    check("no test input column equals a hidden target", not equal, ", ".join(equal) or "none")
    # streamflow_obs at t must be the flow ON day t: it equals the h1 label of the day before
    prev = labels[[ID, DATE, HCOLS[0]]].assign(**{DATE: labels[DATE] + pd.Timedelta(days=1)})
    m = test[[ID, DATE, OBS]].merge(prev, on=[ID, DATE]).dropna()
    check("streamflow_obs is same-day flow, not future flow", np.allclose(m[OBS], m[HCOLS[0]]),
          f"{len(m):,} consecutive pairs compared")

    # 5. static attributes
    banned = [c for c in static.columns if c.startswith(BANNED_STATIC)]
    check("no modelled discharge/runoff attributes", not banned, ", ".join(banned) or "none")
    src = Path(prep["args"]["caravan"]) / "timeseries" / "csv" / prep["args"]["source"]
    start = pd.Timestamp(prep["args"]["start"])
    worst = 0.0
    for gid in list(static[ID])[:25]:
        f = src / f"{gid}.csv"
        if f.exists():
            d = pd.read_csv(f, parse_dates=["date"]).set_index("date")
            exp = climate_indices(d[(d.index >= start) & (d.index < test_start)])
            got = static.set_index(ID).loc[gid]
            worst = max(worst, max(abs(float(got[k]) - v) / (abs(v) + 1e-9) for k, v in exp.items()))
    check("climate indices use train-period weather only", worst < 1e-4, f"max rel diff {worst:.1e}")

    # 6. thresholds
    tv =pd.read_csv(p / "thresholds_val.csv", dtype={ID: str}).set_index(ID)
    ref_v = train[train[DATE] <= fit_end].dropna(subset=[Y]).groupby(ID)[Y].quantile(0.95)
    check("validation thresholds use the fit period only",
          float((tv["q_high"] - ref_v.reindex(tv.index)).abs().max()) < 1e-5)
    t = pd.read_csv(p / "thresholds.csv", dtype={ID: str}).set_index(ID)
    ref_t = train.dropna(subset=[Y]).groupby(ID)[Y].quantile(0.95)
    check("test thresholds use the train period only (train basins)",
          float((t.loc[ref_t.index, "q_high"] - ref_t).abs().max()) < 1e-5)
    check("every scored basin has a threshold", set(labels[ID]) <= set(t.index))

    ok = all(r[1] for r in results)
    lines = [f"| {'PASS' if r[1] else '**FAIL**'} | {r[0]} | {r[2]} |" for r in results]
    report = ("# Leakage audit (generated by scripts/check_leakage.py)\n\n"
              f"Result: **{'ALL PASS' if ok else 'FAILURES'}** ({sum(r[1] for r in results)}/{len(results)})\n\n"
              "| Result | Check | Detail |\n|---|---|---|\n" + "\n".join(lines) + "\n")
    (p / "LEAKAGE_REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
