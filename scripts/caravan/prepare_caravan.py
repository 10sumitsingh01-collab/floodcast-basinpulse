"""Turn REAL Caravan basins into the team's BASINSHIFT data package.

    python scripts/prepare_caravan.py                     # defaults below
    python scripts/prepare_caravan.py --n-basins 671      # every eligible CAMELS-US basin

Reads the untouched Caravan files from fetch_caravan.py and writes, to --out (unzip as data/caravan/):
    train.parquet        gauge_id, date, streamflow, streamflow_obs, streamflow_h1/_h3/_h7, weather
                         (train basins, train period)
    test.parquet         gauge_id, date, streamflow_obs, weather: one row per forecast origin, no targets
    static.parquet       gauge_id + catchment attributes
    test_labels.parquet  gauge_id, date, test_set, streamflow_h1/_h3/_h7. SCORING ONLY (Lane D).
                         No feature or model code may read this file.
    thresholds.csv       per-basin 95%/99% flow (q_high, q_extreme), reference period = train period, all basins
    thresholds_val.csv   same, fit period only, train basins (for temporal validation)
    splits.json          temporal validation cutoff + region-stratified basin folds inside train
    basin_selection.csv  every candidate basin and what happened to it (kept / held out / excluded + why)
    licenses/            Caravan + CAMELS license texts (CC-BY-4.0)
    PREP_REPORT.md, prep_manifest.json   counts, periods, rules, sha256 of every file

Task (BASINSHIFT 8.1): row (basin, t) is a forecast origin at the end of day t. streamflow_obs is the
flow observed on day t (an input); targets are the flows on days t+1, t+3, t+7.

Benchmark design (BASINSHIFT 7):
  - split by BASIN first, then by TIME, before any feature is built (7.2)
  - train      = train basins, train period
  - test_set "temporal"     = train basins, later period          (Test A analogue, 7.4)
  - test_set "geo"          = held-out basins, train period       (Test B analogue, 7.3)
  - test_set "geo_temporal" = held-out basins, later period       (unseen basin AND unseen time)
  - held-out basins are sampled per region (first 2 digits of the USGS id ~ HUC-02 region)
  - climate indices are recomputed from the TRAIN period forcing only; Caravan's own indices
    use the full record (incl. test years), so they are not used
  - HydroATLAS fields derived from modelled discharge/runoff are excluded
Missing streamflow is never filled. Negative streamflow is set to missing and counted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path.home() / "BasinPulse-data"
ID, DATE, Y = "gauge_id", "date", "streamflow"
OBS = "streamflow_obs"   # observed flow up to the forecast origin: an input, not a label
HORIZONS = (1, 3, 7)     # forecast lead times in days (BASINSHIFT 8.1)

DYNAMIC = [
    "total_precipitation_sum",
    "temperature_2m_mean",
    "temperature_2m_min",
    "temperature_2m_max",
    "potential_evaporation_sum_FAO_PENMAN_MONTEITH",
    "snow_depth_water_equivalent_mean",
    "surface_net_solar_radiation_mean",
    "dewpoint_temperature_2m_mean",
    "volumetric_soil_water_layer_1_mean",
    "volumetric_soil_water_layer_3_mean",
]
# Physical catchment descriptors from HydroATLAS. Excluded on purpose:
#   dis_* (modelled discharge), run_mm_syr (modelled runoff), riv_tc_usu / ria_ha_usu
#   (river volume / area derived from modelled discharge), long-term climate fields (recomputed instead).
HYDROATLAS = [
    "ele_mt_sav", "slp_dg_sav", "for_pc_sse", "crp_pc_sse", "urb_pc_sse", "lka_pc_sse",
    "dor_pc_pva", "gwt_cm_sav", "cly_pc_sav", "snd_pc_sav", "slt_pc_sav", "soc_th_sav",
    "kar_pc_sse", "prm_pc_sse", "gla_pc_sse", "ppd_pk_sav", "swc_pc_syr", "snw_pc_syr",
]


def region_of(gid: str) -> str:
    num = gid.split("_", 1)[-1]
    return num[:2] if num[:2].isdigit() else "other"


def climate_indices(df: pd.DataFrame) -> dict:
    """Climate indices from forcing only (no streamflow), on the period passed in."""
    p = df["total_precipitation_sum"]
    pet = df["potential_evaporation_sum_FAO_PENMAN_MONTEITH"]
    t = df["temperature_2m_mean"]
    p_mean = p.mean()
    wet = (p >= 1.0).astype(int)
    high = (p >= 5 * p_mean).astype(int)
    return {
        "p_mean": p_mean,
        "pet_mean": pet.mean(),
        "aridity": pet.mean() / p_mean if p_mean > 0 else np.nan,
        "frac_snow": p[t < 0].sum() / p.sum() if p.sum() > 0 else np.nan,
        "high_prec_freq": high.mean(),
        "low_prec_freq": 1 - wet.mean(),
        "p_seasonality": _seasonality(p),
    }


def _seasonality(p: pd.Series) -> float:
    m = p.groupby(p.index.month).mean()
    return float((m.max() - m.min()) / m.mean()) if m.mean() > 0 else np.nan


def stratified_pick(ids: list[str], k: int, rng: np.random.Generator) -> list[str]:
    """Pick k ids, proportional to region size, at least 1 per region where possible."""
    by_r: dict[str, list[str]] = {}
    for g in sorted(ids):
        by_r.setdefault(region_of(g), []).append(g)
    if k >= len(ids):
        return sorted(ids)
    quota = {r: max(1, round(k * len(v) / len(ids))) for r, v in by_r.items()}
    while sum(quota.values()) > k:  # trim largest quotas first
        r = max(quota, key=lambda x: (quota[x], x))
        quota[r] -= 1
    picked = []
    for r in sorted(by_r):
        members = list(by_r[r])
        rng.shuffle(members)
        picked += members[: quota[r]]
    return sorted(picked)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caravan", type=Path, default=BASE / "caravan")
    ap.add_argument("--source", default="camels")
    ap.add_argument("--out", type=Path, default=BASE / "package" / "caravan")
    ap.add_argument("--val-years", type=int, default=2, help="last N train years = temporal validation")
    ap.add_argument("--n-folds", type=int, default=5, help="region-stratified basin folds inside train")
    ap.add_argument("--n-basins", type=int, default=300, help="basins used in total (train + held out)")
    ap.add_argument("--geo-frac", type=float, default=0.2, help="share of basins held out entirely")
    ap.add_argument("--start", default="1981-10-01")
    ap.add_argument("--test-start", default="2008-10-01")
    ap.add_argument("--end", default="2014-09-30")
    ap.add_argument("--min-train-coverage", type=float, default=0.90)
    ap.add_argument("--min-test-coverage", type=float, default=0.50)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    start, test_start, end = map(pd.Timestamp, (a.start, a.test_start, a.end))
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)

    ts_dir = a.caravan / "timeseries" / "csv" / a.source
    files = sorted(ts_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No time series in {ts_dir}. Run scripts/fetch_caravan.py first.")
    other = pd.read_csv(a.caravan / "attributes" / a.source / f"attributes_other_{a.source}.csv",
                        dtype={ID: str}).set_index(ID)
    hyd = pd.read_csv(a.caravan / "attributes" / a.source / f"attributes_hydroatlas_{a.source}.csv",
                      dtype={ID: str}).set_index(ID)

    # ---- 1. QA every downloaded basin on real streamflow coverage -------------
    n_train_days = (test_start - start).days
    n_test_days = (end - test_start).days + 1
    rows, frames = [], {}
    for f in files:
        gid = f.stem
        df = pd.read_csv(f, usecols=[DATE, Y] + DYNAMIC, parse_dates=[DATE])
        df = df[(df[DATE] >= start) & (df[DATE] <= end)].copy()
        neg = int((df[Y] < 0).sum())
        df.loc[df[Y] < 0, Y] = np.nan
        tr = df[df[DATE] < test_start]
        te = df[df[DATE] >= test_start]
        cov_tr = tr[Y].notna().sum() / n_train_days
        cov_te = te[Y].notna().sum() / n_test_days
        wx_missing = float(df[DYNAMIC].isna().mean().max())
        reason = ""
        if gid not in other.index:
            reason = "no attributes"
        elif cov_tr < a.min_train_coverage:
            reason = f"train-period streamflow coverage {cov_tr:.0%} < {a.min_train_coverage:.0%}"
        elif cov_te < a.min_test_coverage:
            reason = f"test-period streamflow coverage {cov_te:.0%} < {a.min_test_coverage:.0%}"
        elif wx_missing > 0.05:
            reason = f"weather inputs {wx_missing:.0%} missing"
        rows.append({ID: gid, "region": region_of(gid), "train_coverage": round(cov_tr, 4),
                     "test_coverage": round(cov_te, 4), "negative_set_missing": neg,
                     "zero_flow_share": round(float((tr[Y] == 0).mean()), 4),
                     "status": f"excluded: {reason}" if reason else "eligible"})
        if not reason:
            frames[gid] = df
    sel = pd.DataFrame(rows).set_index(ID)

    # ---- 2. choose basins, then hold out whole basins per region ---------------
    eligible = sorted(frames)
    used = stratified_pick(eligible, a.n_basins, rng)
    geo = stratified_pick(used, int(round(a.geo_frac * len(used))), rng)
    for g in eligible:
        sel.loc[g, "status"] = "geo_test" if g in geo else ("train" if g in used else "not sampled")
    train_ids = sorted(set(used) - set(geo))

    # ---- 3. static attributes (climate indices from TRAIN period only) --------
    static = []
    for g in used:
        df = frames[g].set_index(DATE)
        rec = {ID: g, "area_km2": other.loc[g, "area"], "gauge_lat": other.loc[g, "gauge_lat"],
               "gauge_lon": other.loc[g, "gauge_lon"]}
        rec.update(climate_indices(df[df.index < test_start]))
        rec.update({c: hyd.loc[g, c] for c in HYDROATLAS if c in hyd.columns})
        static.append(rec)
    static = pd.DataFrame(static)

    # ---- 4. assemble files -----------------------------------------------------
    def tag(df: pd.DataFrame, g: str) -> pd.DataFrame:
        df = df.copy()
        df.insert(0, ID, g)
        later = df[DATE] >= test_start
        if g in geo:
            df["test_set"] = np.where(later, "geo_temporal", "geo")
        else:
            df["test_set"] = np.where(later, "temporal", "train")
        return df

    allrows = pd.concat([tag(frames[g], g) for g in used], ignore_index=True)
    allrows[DYNAMIC + [Y]] = allrows[DYNAMIC + [Y]].astype("float32")
    # streamflow_obs = flow observed on day t, known at the forecast origin (end of day t): an INPUT
    allrows[OBS] = allrows[Y]
    hcols = [f"{Y}_h{h}" for h in HORIZONS]
    is_train = allrows["test_set"] == "train"

    def add_targets(frame: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
        """Flow on day t+h in the same basin, matched by calendar date (gap-safe)."""
        for h in HORIZONS:
            fut = source[[ID, DATE, Y]].copy()
            fut[DATE] = fut[DATE] - pd.Timedelta(days=h)
            frame = frame.merge(fut.rename(columns={Y: f"{Y}_h{h}"}), on=[ID, DATE], how="left")
        return frame

    # train targets come from train rows only, so they stop at the end of the train period
    tr_rows = allrows.loc[is_train].reset_index(drop=True)
    train = add_targets(tr_rows[[ID, DATE, Y, OBS] + DYNAMIC], tr_rows)
    # test targets may come from any period of the same basin (geo origin in Sept 2008 -> Oct 2008)
    te_rows = allrows.loc[~is_train].reset_index(drop=True)
    labels = add_targets(te_rows[[ID, DATE, "test_set"]], allrows)
    test = te_rows[[ID, DATE, OBS] + DYNAMIC]
    train[hcols] = train[hcols].astype("float32")
    labels[hcols] = labels[hcols].astype("float32")

    # splits inside train: temporal validation = last `val_years`, plus region-stratified basin folds
    val_start = test_start - pd.DateOffset(years=a.val_years)
    fit_end = val_start - pd.Timedelta(days=1)
    folds = {}
    by_r: dict[str, list[str]] = {}
    for g in train_ids:
        by_r.setdefault(region_of(g), []).append(g)
    offset = 0
    for r in sorted(by_r):
        members = list(by_r[r])
        rng.shuffle(members)
        for i, g in enumerate(members):
            folds[g] = int((offset + i) % a.n_folds)
        offset += len(members)
    splits = {
        "train_period": [str(start.date()), str((test_start - pd.Timedelta(days=1)).date())],
        "temporal_validation": {"fit_end_inclusive": str(fit_end.date()),
                                "val_start_inclusive": str(val_start.date()),
                                "rule": "when fitting on dates <= fit_end, drop rows whose t+h is after fit_end"},
        "station_folds": folds, "n_folds": a.n_folds, "seed": a.seed,
        "horizons_days": list(HORIZONS),
        "test_period": [str(test_start.date()), str(end.date())],
    }

    def thresholds(frame: pd.DataFrame) -> pd.DataFrame:
        g = frame.dropna(subset=[Y]).groupby(ID)[Y]
        return pd.DataFrame({"q_high": g.quantile(0.95), "q_extreme": g.quantile(0.99),
                             "n_obs": g.size()}).reset_index()

    # thresholds.csv: all basins, reference period = train period (fixed before any modelling)
    thr = thresholds(allrows[allrows[DATE] < test_start])
    # thresholds_val.csv: train basins, fit period only (for scoring temporal validation)
    thr_val = thresholds(allrows[is_train & (allrows[DATE] <= fit_end)])

    outputs = {"train.parquet": train, "test.parquet": test, "static.parquet": static,
               "test_labels.parquet": labels}
    for name, df in outputs.items():
        df.to_parquet(a.out / name, index=False)
    thr.to_csv(a.out / "thresholds.csv", index=False)
    thr_val.to_csv(a.out / "thresholds_val.csv", index=False)
    (a.out / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    sel.reset_index().to_csv(a.out / "basin_selection.csv", index=False)
    lic = a.out / "licenses"
    lic.mkdir(exist_ok=True)
    for f in [a.caravan / "licenses" / "README.md", a.caravan / "licenses" / a.source / f"license_{a.source}.md"]:
        if f.exists():
            (lic / f.name).write_bytes(f.read_bytes())

    # ---- 5. report + manifest -----------------------------------------------------
    counts = labels["test_set"].value_counts().to_dict()
    events = (labels.merge(thr, on=ID).assign(high=lambda d: d[f"{Y}_h1"] > d["q_high"],
                                              ext=lambda d: d[f"{Y}_h1"] > d["q_extreme"])
              .groupby("test_set")[["high", "ext"]].sum().astype(int).to_dict("index"))
    excluded = sel[sel["status"].str.startswith("excluded")]
    hashes = {n: hashlib.sha256((a.out / n).read_bytes()).hexdigest() for n in
              list(outputs) + ["thresholds.csv", "thresholds_val.csv", "splits.json", "basin_selection.csv"]}
    fm = a.caravan / "fetch_manifest.json"
    meta = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": f"Caravan v1.6 CSV, source dataset '{a.source}'",
        "fetch_manifest_sha256": hashlib.sha256(fm.read_bytes()).hexdigest() if fm.exists() else None,
        "args": {k: str(v) for k, v in vars(a).items()},
        "n_downloaded": len(files), "n_eligible": len(eligible), "n_used": len(used),
        "n_train_basins": len(train_ids), "n_geo_basins": len(geo),
        "train_rows": len(train), "test_rows": counts, "test_events": events,
        "dynamic": DYNAMIC, "static": [c for c in static.columns if c != ID],
        "sha256": hashes,
    }
    (a.out / "prep_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    report = f"""# Caravan preparation report (generated, do not hand-edit)

Source: Caravan v1.6 (CSV), `{a.source}` (CAMELS-US, CC-BY-4.0). Real observed streamflow, ERA5-Land forcing.
Built: {meta['built_utc']}

## Periods (water years)
- train period: {start.date()} to {(test_start - pd.Timedelta(days=1)).date()}
- test period:  {test_start.date()} to {end.date()}

## Basins
- downloaded: {len(files)}; eligible after QA: {len(eligible)}; excluded: {len(excluded)}
- used: {len(used)} = {len(train_ids)} train + {len(geo)} held out entirely (geo test), stratified by region
- regions used: {static[ID].map(region_of).nunique()}

## Rows
- train: {len(train):,}
- test: {', '.join(f'{k} {v:,}' for k, v in sorted(counts.items()))}
- high / extreme flow days at horizon 1 (thresholds from reference period): {events}

## Forecast task
- each row (basin, t) is a forecast origin at the end of day t
- input `streamflow_obs` = flow on day t; targets `streamflow_h1/_h3/_h7` = flow on t+1/t+3/t+7
- train targets never reach past {(test_start - pd.Timedelta(days=1)).date()}
- held-out basins: no training rows, but their observed flow history is an input at prediction time
  (gauged-but-untrained basins, BASINSHIFT 7.4). A model ignoring `streamflow_obs` simulates ungauged basins.
- score each origin from its own row only: a later row's `streamflow_obs` is an earlier row's target

## Exclusions
{chr(10).join(f'- `{g}`: {s}' for g, s in excluded['status'].items()) or '- none'}

## Rules
- streamflow (mm/day, already per unit area) is never filled; negatives set missing ({int(sel['negative_set_missing'].sum())} values)
- climate indices recomputed from train-period forcing only
- HydroATLAS modelled discharge/runoff fields excluded
- `test_labels.parquet` is for scoring only (Lane D); no feature or model code reads it
"""
    (a.out / "PREP_REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
