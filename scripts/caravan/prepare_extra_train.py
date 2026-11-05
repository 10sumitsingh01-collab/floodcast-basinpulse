"""Extra TRAINING basins: eligible CAMELS-US basins that were not sampled into the 299-basin benchmark.

    python scripts/caravan/prepare_extra_train.py --n 150 --out data/caravan_extra

These basins are in no test set (`basin_selection.csv` status "not sampled"), so adding them to training cannot
leak test information. Only their training-period rows (1981-10 to 2008-09) are written, prepared exactly like
data/caravan/train.parquet (same inputs, horizon targets inside the training period, attributes with climate
indices from training-period weather, 95th / 99th percentile thresholds from the training period).
Use with:  python basinshift_run.py --extra-train data/caravan_extra ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import prepare_caravan as pc

ID, DATE, Y, OBS = pc.ID, pc.DATE, pc.Y, pc.OBS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caravan", type=Path, default=pc.BASE / "caravan")
    ap.add_argument("--benchmark", type=Path, default=Path("data/caravan"))
    ap.add_argument("--out", type=Path, default=Path("data/caravan_extra"))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    split = json.loads((a.benchmark / "splits.json").read_text())
    start = pd.Timestamp(split["train_period"][0])
    end_train = pd.Timestamp(split["train_period"][1])
    sel = pd.read_csv(a.benchmark / "basin_selection.csv", dtype={ID: str})
    pool = sorted(sel.loc[sel.status == "not sampled", ID])
    used = set(sel.loc[sel.status.isin(["train", "geo_test"]), ID])
    assert not (set(pool) & used)
    rng = np.random.default_rng(a.seed)
    pick = pc.stratified_pick(pool, a.n, rng)

    other = pd.read_csv(a.caravan / "attributes" / "camels" / "attributes_other_camels.csv", dtype={ID: str}).set_index(ID)
    hyd = pd.read_csv(a.caravan / "attributes" / "camels" / "attributes_hydroatlas_camels.csv", dtype={ID: str}).set_index(ID)
    parts, static = [], []
    for g in pick:
        df = pd.read_csv(a.caravan / "timeseries" / "csv" / "camels" / f"{g}.csv",
                         usecols=[DATE, Y] + pc.DYNAMIC, parse_dates=[DATE])
        df = df[(df[DATE] >= start) & (df[DATE] <= end_train)].copy()
        df.loc[df[Y] < 0, Y] = np.nan
        rec = {ID: g, "area_km2": other.loc[g, "area"], "gauge_lat": other.loc[g, "gauge_lat"],
               "gauge_lon": other.loc[g, "gauge_lon"]}
        rec.update(pc.climate_indices(df.set_index(DATE)))
        rec.update({c: hyd.loc[g, c] for c in pc.HYDROATLAS if c in hyd.columns})
        static.append(rec)
        df.insert(0, ID, g)
        parts.append(df)
    tr = pd.concat(parts, ignore_index=True)
    tr[pc.DYNAMIC + [Y]] = tr[pc.DYNAMIC + [Y]].astype("float32")
    tr[OBS] = tr[Y]
    for h in pc.HORIZONS:                      # targets from training-period rows only
        fut = tr[[ID, DATE, Y]].assign(**{DATE: tr[DATE] - pd.Timedelta(days=h)})
        tr = tr.merge(fut.rename(columns={Y: f"{Y}_h{h}"}), on=[ID, DATE], how="left")
        tr[f"{Y}_h{h}"] = tr[f"{Y}_h{h}"].astype("float32")
    bench_cols = pd.read_parquet(a.benchmark / "train.parquet").columns
    tr = tr[list(bench_cols)]
    ref = tr.dropna(subset=[Y]).groupby(ID)[Y]
    thr = pd.DataFrame({"q_high": ref.quantile(0.95), "q_extreme": ref.quantile(0.99),
                        "n_obs": ref.size()}).reset_index()
    a.out.mkdir(parents=True, exist_ok=True)
    tr.to_parquet(a.out / "train.parquet", index=False)
    pd.DataFrame(static).to_parquet(a.out / "static.parquet", index=False)
    thr.to_csv(a.out / "thresholds.csv", index=False)
    print(f"{len(pick)} extra training basins, {len(tr):,} rows, max target date "
          f"{(tr.dropna(subset=[Y + '_h7'])[DATE] + pd.Timedelta(days=7)).max().date()} (train end {end_train.date()})")


if __name__ == "__main__":
    main()
