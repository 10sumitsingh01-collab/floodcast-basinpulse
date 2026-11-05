"""Zero-shot transfer package: Caravan basins from ANOTHER continent, prepared exactly like the US benchmark.

    python scripts/caravan/prepare_transfer.py --source lamah --out data/transfer_lamah

No row of this package is ever used for training. It provides, for the same test period as the US benchmark
(2008-10-01 to 2014-09-30):
    test.parquet         gauge_id, date, streamflow_obs, 10 weather columns (forecast origins)
    history.parquet      inputs before the test period (look-back; climatology), no targets
    static.parquet       same attributes as the US package (climate indices from 1981-2008 weather only)
    thresholds.csv       95th / 99th percentile flow over 1981-10 to 2008-09 (event definition, as for US basins)
    test_labels.parquet  streamflow_h1/_h3/_h7 per origin. SCORING ONLY.
    basin_selection.csv, PREP_REPORT.md
LaMAH-CE is CC-BY-SA-4.0 (share-alike): this package is for local evaluation, not redistribution.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import prepare_caravan as pc

ID, DATE, Y, OBS = pc.ID, pc.DATE, pc.Y, pc.OBS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caravan", type=Path, default=pc.BASE / "caravan")
    ap.add_argument("--source", default="lamah")
    ap.add_argument("--out", type=Path, default=Path("data/transfer_lamah"))
    ap.add_argument("--start", default="1981-10-01")
    ap.add_argument("--test-start", default="2008-10-01")
    ap.add_argument("--end", default="2014-09-30")
    ap.add_argument("--min-ref-coverage", type=float, default=0.80, help="observed flow in 1981-2008 (thresholds)")
    ap.add_argument("--min-test-coverage", type=float, default=0.50)
    a = ap.parse_args()
    start, test_start, end = map(pd.Timestamp, (a.start, a.test_start, a.end))
    a.out.mkdir(parents=True, exist_ok=True)

    files = sorted((a.caravan / "timeseries" / "csv" / a.source).glob("*.csv"))
    other = pd.read_csv(a.caravan / "attributes" / a.source / f"attributes_other_{a.source}.csv",
                        dtype={ID: str}).set_index(ID)
    hyd = pd.read_csv(a.caravan / "attributes" / a.source / f"attributes_hydroatlas_{a.source}.csv",
                      dtype={ID: str}).set_index(ID)
    n_ref, n_test = (test_start - start).days, (end - test_start).days + 1
    rows, frames = [], {}
    for f in files:
        gid = f.stem
        df = pd.read_csv(f, usecols=[DATE, Y] + pc.DYNAMIC, parse_dates=[DATE])
        df = df[(df[DATE] >= start) & (df[DATE] <= end)].copy()
        df.loc[df[Y] < 0, Y] = np.nan
        cov_ref = df.loc[df[DATE] < test_start, Y].notna().sum() / n_ref
        cov_te = df.loc[df[DATE] >= test_start, Y].notna().sum() / n_test
        wx = float(df[pc.DYNAMIC].isna().mean().max())
        reason = ("no attributes" if gid not in other.index or gid not in hyd.index else
                  f"reference coverage {cov_ref:.0%}" if cov_ref < a.min_ref_coverage else
                  f"test coverage {cov_te:.0%}" if cov_te < a.min_test_coverage else
                  f"weather {wx:.0%} missing" if wx > 0.05 else
                  "dates not continuous" if (df[DATE].diff().dt.days.dropna() != 1).any() else "")
        rows.append({ID: gid, "country": other.loc[gid, "country"] if gid in other.index else "",
                     "ref_coverage": round(cov_ref, 4), "test_coverage": round(cov_te, 4),
                     "status": f"excluded: {reason}" if reason else "used"})
        if not reason:
            frames[gid] = df
    sel = pd.DataFrame(rows)
    used = sorted(frames)

    static, parts = [], []
    for g in used:
        df = frames[g].set_index(DATE)
        rec = {ID: g, "area_km2": other.loc[g, "area"], "gauge_lat": other.loc[g, "gauge_lat"],
               "gauge_lon": other.loc[g, "gauge_lon"]}
        rec.update(pc.climate_indices(df[df.index < test_start]))
        rec.update({c: hyd.loc[g, c] for c in pc.HYDROATLAS if c in hyd.columns})
        static.append(rec)
        d = frames[g].copy()
        d.insert(0, ID, g)
        parts.append(d)
    static = pd.DataFrame(static)
    allrows = pd.concat(parts, ignore_index=True)
    allrows[pc.DYNAMIC + [Y]] = allrows[pc.DYNAMIC + [Y]].astype("float32")
    allrows[OBS] = allrows[Y]

    ref = allrows[allrows[DATE] < test_start].dropna(subset=[Y]).groupby(ID)[Y]
    thr = pd.DataFrame({"q_high": ref.quantile(0.95), "q_extreme": ref.quantile(0.99),
                        "n_obs": ref.size()}).reset_index()

    te = allrows[allrows[DATE] >= test_start].reset_index(drop=True)
    labels = te[[ID, DATE]].assign(test_set="europe_zero_shot")
    for h in pc.HORIZONS:
        fut = allrows[[ID, DATE, Y]].assign(**{DATE: allrows[DATE] - pd.Timedelta(days=h)})
        labels = labels.merge(fut.rename(columns={Y: f"{Y}_h{h}"}), on=[ID, DATE], how="left")
    hist = allrows[allrows[DATE] < test_start]   # all earlier inputs (no targets); models take what they need
    cols = [ID, DATE, OBS] + pc.DYNAMIC
    te[cols].to_parquet(a.out / "test.parquet", index=False)
    hist[cols].to_parquet(a.out / "history.parquet", index=False)
    static.to_parquet(a.out / "static.parquet", index=False)
    labels.to_parquet(a.out / "test_labels.parquet", index=False)
    thr.to_csv(a.out / "thresholds.csv", index=False)
    sel.to_csv(a.out / "basin_selection.csv", index=False)

    countries = static[ID].map(other["country"]).value_counts().to_dict()
    n_ev = int((labels.merge(thr, on=ID).eval("streamflow_h1 > q_high")).sum())
    rep = f"""# Zero-shot transfer package: `{a.source}` (generated)

Built {datetime.now(timezone.utc).isoformat(timespec='seconds')} from Caravan v1.6. Never used for training.
- basins: {len(files)} extracted, {len(used)} used, {len(files) - len(used)} excluded (`basin_selection.csv`)
- countries: {countries}
- test origins: {len(te):,} ({test_start.date()} to {end.date()}); high-flow days at 1 day ahead: {n_ev:,}
- area: median {static.area_km2.median():,.0f} km² (10-90%: {static.area_km2.quantile(.1):,.0f} – {static.area_km2.quantile(.9):,.0f})
- same inputs, attribute recipe and threshold definition as the US benchmark
- license: LaMAH-CE data are CC-BY-SA-4.0 (Klingler et al. 2021); not redistributed
"""
    (a.out / "PREP_REPORT.md").write_text(rep, encoding="utf-8")
    (a.out / "prep_manifest.json").write_text(json.dumps({"source": a.source, "n_used": len(used),
                                                          "args": {k: str(v) for k, v in vars(a).items()}},
                                                         indent=2), encoding="utf-8")
    print(rep)


if __name__ == "__main__":
    main()
