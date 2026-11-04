"""Training data for the SUBMITTED model: every eligible basin, every year (nothing held back).

    python scripts/caravan/prepare_final_train.py --source camels --out data/final_camels
    python scripts/caravan/prepare_final_train.py --source lamah  --out data/final_lamah --min-coverage 0.8

Model selection used the benchmark in data/caravan (held-out basins and years). Once the configuration is fixed,
the model the organisers run on their hidden data is retrained on all real observations we have, prepared with the
same recipe: same inputs, horizon targets by calendar date inside each basin's record, catchment attributes with
climate indices from the basin's own weather, and 95th / 99th percentile thresholds of its observed flow.
Writes train.parquet, static.parquet, thresholds.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import prepare_caravan as pc

ID, DATE, Y, OBS = pc.ID, pc.DATE, pc.Y, pc.OBS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caravan", type=Path, default=pc.BASE / "caravan")
    ap.add_argument("--source", default="camels")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", default="1981-10-01")
    ap.add_argument("--end", default="2014-09-30")
    ap.add_argument("--min-coverage", type=float, default=0.75, help="share of days with observed flow")
    a = ap.parse_args()
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    other = pd.read_csv(a.caravan / "attributes" / a.source / f"attributes_other_{a.source}.csv",
                        dtype={ID: str}).set_index(ID)
    hyd = pd.read_csv(a.caravan / "attributes" / a.source / f"attributes_hydroatlas_{a.source}.csv",
                      dtype={ID: str}).set_index(ID)
    parts, static, skipped = [], [], 0
    n_days = (end - start).days + 1
    for f in sorted((a.caravan / "timeseries" / "csv" / a.source).glob("*.csv")):
        g = f.stem
        df = pd.read_csv(f, usecols=[DATE, Y] + pc.DYNAMIC, parse_dates=[DATE])
        df = df[(df[DATE] >= start) & (df[DATE] <= end)].copy()
        df.loc[df[Y] < 0, Y] = np.nan
        if (g not in other.index or g not in hyd.index or df[Y].notna().sum() / n_days < a.min_coverage
                or df[pc.DYNAMIC].isna().mean().max() > 0.05 or (df[DATE].diff().dt.days.dropna() != 1).any()):
            skipped += 1
            continue
        rec = {ID: g, "area_km2": other.loc[g, "area"], "gauge_lat": other.loc[g, "gauge_lat"],
               "gauge_lon": other.loc[g, "gauge_lon"]}
        rec.update(pc.climate_indices(df.set_index(DATE)))
        rec.update({c: hyd.loc[g, c] for c in pc.HYDROATLAS if c in hyd.columns})
        static.append(rec)
        df.insert(0, ID, g)
        df[pc.DYNAMIC + [Y]] = df[pc.DYNAMIC + [Y]].astype("float32")
        df[OBS] = df[Y]
        for h in pc.HORIZONS:
            fut = df[[ID, DATE, Y]].assign(**{DATE: df[DATE] - pd.Timedelta(days=h)})
            df = df.merge(fut.rename(columns={Y: f"{Y}_h{h}"}), on=[ID, DATE], how="left")
            df[f"{Y}_h{h}"] = df[f"{Y}_h{h}"].astype("float32")
        parts.append(df[[ID, DATE, Y, OBS] + pc.DYNAMIC + [f"{Y}_h{h}" for h in pc.HORIZONS]])
    tr = pd.concat(parts, ignore_index=True)
    ref = tr.dropna(subset=[Y]).groupby(ID)[Y]
    thr = pd.DataFrame({"q_high": ref.quantile(0.95), "q_extreme": ref.quantile(0.99),
                        "n_obs": ref.size()}).reset_index()
    a.out.mkdir(parents=True, exist_ok=True)
    tr.to_parquet(a.out / "train.parquet", index=False)
    pd.DataFrame(static).to_parquet(a.out / "static.parquet", index=False)
    thr.to_csv(a.out / "thresholds.csv", index=False)
    print(f"{a.source}: {len(static)} basins used, {skipped} skipped, {len(tr):,} rows "
          f"({tr[DATE].min().date()} to {tr[DATE].max().date()})")


if __name__ == "__main__":
    main()
