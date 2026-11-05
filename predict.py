"""Run the trained BasinPulse model on NEW / hidden data and write the BASINSHIFT 8.3 submission.

    python predict.py --inputs hidden_test.parquet --out submission.jsonl.gz
    python predict.py --inputs test.csv --history history.csv --static static.csv --thresholds thresholds.csv \\
                      --model models/basinpulse_final.joblib --out submission.jsonl.gz

--inputs      one row per forecast origin (basin, day t): gauge_id (or catchment_id / station_id), date
              (or forecast_origin), streamflow_obs (observed flow up to day t; optional) and any of the weather
              columns below (missing ones are treated as unavailable, never invented).
--history     optional earlier rows for the same basins (same columns) so lags and rolling windows have a
              look-back; without it the first ~60 days of each basin have shorter windows.
--static      optional catchment attributes per basin (Caravan / HydroATLAS names); anything missing is NaN,
              and climate indices are computed from the supplied weather when absent.
--thresholds  optional per-basin q_high / q_extreme (95th / 99th percentile flow). If absent they are computed
              from the observed flow in --history + --inputs, which is the BASINSHIFT definition when those
              rows cover the reference period.
Output: one JSON record per (basin, origin) with 1/3/7-day forecasts, 10-90% range and P(high), P(extreme),
validated against the 8.4 rules before it is written. Units: mm/day (Caravan convention).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import submission  # noqa: E402

ID, DATE, OBS = "gauge_id", "date", "streamflow_obs"
WEATHER = [
    "total_precipitation_sum", "temperature_2m_mean", "temperature_2m_min", "temperature_2m_max",
    "potential_evaporation_sum_FAO_PENMAN_MONTEITH", "snow_depth_water_equivalent_mean",
    "surface_net_solar_radiation_mean", "dewpoint_temperature_2m_mean",
    "volumetric_soil_water_layer_1_mean", "volumetric_soil_water_layer_3_mean",
]
ALIASES = {ID: ["catchment_id", "station_id", "basin_id", "gauge"], DATE: ["forecast_origin", "time", "day"],
           OBS: ["streamflow", "discharge", "observed_flow", "q_obs"]}
STATIC_COLS = ["area_km2", "gauge_lat", "gauge_lon", "p_mean", "pet_mean", "aridity", "frac_snow",
               "high_prec_freq", "low_prec_freq", "p_seasonality", "ele_mt_sav", "slp_dg_sav", "for_pc_sse",
               "crp_pc_sse", "urb_pc_sse", "lka_pc_sse", "dor_pc_pva", "gwt_cm_sav", "cly_pc_sav", "snd_pc_sav",
               "slt_pc_sav", "soc_th_sav", "kar_pc_sse", "prm_pc_sse", "gla_pc_sse", "ppd_pk_sav", "swc_pc_syr",
               "snw_pc_syr"]


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={"gauge_id": str, "catchment_id": str, "station_id": str})


def normalise(df: pd.DataFrame, need_obs: bool = True) -> pd.DataFrame:
    """Map common column names to ours, coerce types, add missing input columns as NaN, keep daily continuity."""
    df = df.copy()
    for ours, alts in ALIASES.items():
        if ours not in df.columns:
            for a in alts:
                if a in df.columns and not (ours == OBS and a.startswith("streamflow_h")):
                    df = df.rename(columns={a: ours})
                    break
    if area := "area" if "area" in df.columns and "area_km2" not in df.columns else None:
        df = df.rename(columns={area: "area_km2"})
    df = df.drop(columns=[c for c in df.columns if c.startswith("streamflow_h")], errors="ignore")
    df[ID] = df[ID].astype(str)
    df[DATE] = pd.to_datetime(df[DATE], utc=True).dt.tz_localize(None).dt.normalize()
    if need_obs and OBS not in df.columns:
        df[OBS] = np.nan
    for c in WEATHER:
        if c not in df.columns:
            df[c] = np.nan
    return df


def continuous(df: pd.DataFrame) -> pd.DataFrame:
    """Insert missing calendar days per basin (as NaN rows) so day-based lags stay exact."""
    parts = []
    for g, d in df.groupby(ID, sort=False):
        d = d.drop_duplicates(DATE).set_index(DATE).sort_index()
        d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))
        d[ID] = g
        parts.append(d.rename_axis(DATE).reset_index())
    return pd.concat(parts, ignore_index=True)


def derive_static(rows: pd.DataFrame, static: pd.DataFrame | None) -> pd.DataFrame:
    from scripts.caravan.prepare_caravan import climate_indices  # same recipe as the training data
    ids = sorted(rows[ID].unique())
    st = pd.DataFrame({ID: ids})
    if static is not None:
        static = normalise(static, need_obs=False) if DATE in static.columns else static.copy()
        if ID not in static.columns:
            for a in ALIASES[ID]:
                if a in static.columns:
                    static = static.rename(columns={a: ID})
        if "area" in static.columns and "area_km2" not in static.columns:
            static = static.rename(columns={"area": "area_km2"})
        static[ID] = static[ID].astype(str)
        st = st.merge(static, on=ID, how="left")
    need_clim = [c for c in ("p_mean", "pet_mean", "aridity", "frac_snow", "high_prec_freq", "low_prec_freq",
                             "p_seasonality") if c not in st.columns or st[c].isna().any()]
    if need_clim:
        clim = {}
        for g, d in rows.groupby(ID):
            d = d.set_index(DATE)
            if d["total_precipitation_sum"].notna().sum() > 365:
                clim[g] = climate_indices(d)
        cdf = pd.DataFrame.from_dict(clim, orient="index").rename_axis(ID).reset_index()
        for c in need_clim:
            if c in cdf.columns:
                st[c] = st[c].fillna(st[ID].map(cdf.set_index(ID)[c])) if c in st.columns else st[ID].map(
                    cdf.set_index(ID)[c])
    for c in STATIC_COLS:
        if c not in st.columns:
            st[c] = np.nan
    return st[[ID] + STATIC_COLS]


def derive_thresholds(rows: pd.DataFrame, thr: pd.DataFrame | None) -> pd.DataFrame:
    ids = sorted(rows[ID].unique())
    out = pd.DataFrame({ID: ids})
    if thr is not None:
        thr = thr.copy()
        if ID not in thr.columns:
            for a in ALIASES[ID]:
                if a in thr.columns:
                    thr = thr.rename(columns={a: ID})
        thr[ID] = thr[ID].astype(str)
        out = out.merge(thr[[ID, "q_high", "q_extreme"]], on=ID, how="left")
    else:
        out["q_high"], out["q_extreme"] = np.nan, np.nan
    g = rows.dropna(subset=[OBS]).groupby(ID)[OBS]
    out["q_high"] = out["q_high"].fillna(out[ID].map(g.quantile(0.95)))
    out["q_extreme"] = out["q_extreme"].fillna(out[ID].map(g.quantile(0.99)))
    # basins with no observed flow at all: no basin-specific threshold can be defined
    return out


def write_explanations(model, allrows, is_hist, static, thr, keys, path: Path) -> None:
    """ShiftGuard extras: novelty per basin and top drivers per forecast (first member of an ensemble)."""
    from src.explain import explain
    base = model.members[0] if hasattr(model, "members") else model
    test, hist = allrows[~is_hist].reset_index(drop=True), allrows[is_hist].reset_index(drop=True)
    parts = [explain(base, test, hist, static, thr, horizon=h) for h in (1, 3, 7)]
    ex = keys.merge(pd.concat(parts, ignore_index=True), on=[ID, DATE], how="left")
    if hasattr(base, "novelty") and hasattr(base, "_nov_ref"):
        ex["basin_novelty"] = ex[ID].map(base.novelty(static, keys[ID].unique()))
    ex.to_parquet(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--history", type=Path, default=None)
    ap.add_argument("--static", type=Path, default=None)
    ap.add_argument("--thresholds", type=Path, default=None)
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "basinpulse_final.joblib")
    ap.add_argument("--out", type=Path, default=Path("submission.jsonl.gz"))
    ap.add_argument("--long-out", type=Path, default=None, help="also write long-format predictions (parquet)")
    ap.add_argument("--explain", type=Path, default=None,
                    help="also write ShiftGuard extras (parquet): each basin's novelty score and, per forecast, the "
                         "top drivers of P(high flow) at 1/3/7 days")
    a = ap.parse_args()
    t0 = time.time()
    import joblib
    model = joblib.load(a.model)

    test = normalise(read_any(a.inputs))
    keys = test[[ID, DATE]].drop_duplicates().reset_index(drop=True)
    hist = normalise(read_any(a.history)) if a.history else test.iloc[:0]
    allrows = pd.concat([hist, test], ignore_index=True).drop_duplicates([ID, DATE], keep="last")
    allrows = continuous(allrows[[ID, DATE, OBS] + WEATHER + ([c for c in ["obs_age_days"] if c in allrows] )])
    static = derive_static(allrows, read_any(a.static) if a.static else None)
    thr = derive_thresholds(allrows, read_any(a.thresholds) if a.thresholds else None)

    # Every origin is predicted from rows dated <= its own date: the model's features are trailing-only.
    first = keys.groupby(ID)[DATE].min()
    is_hist = allrows[DATE] < allrows[ID].map(first)
    preds = model.predict(model, allrows[~is_hist].reset_index(drop=True), allrows[is_hist].reset_index(drop=True),
                          static, thr)
    preds = keys.merge(preds, on=[ID, DATE], how="left")   # only the requested origins
    for c in ("predicted_discharge", "lower", "upper"):
        preds[c] = preds[c].fillna(0.0)
    preds["p_high"] = preds["p_high"].fillna(0.05)
    preds["p_extreme"] = np.minimum(preds["p_extreme"].fillna(0.01), preds["p_high"])
    submission.assert_valid(preds, keys)
    submission.write_jsonl(preds, a.out)
    back = submission.read_jsonl(a.out)
    submission.assert_valid(back, keys)
    if a.long_out:
        preds.to_parquet(a.long_out, index=False)
    if a.explain:
        write_explanations(model, allrows, is_hist, static, thr, keys, a.explain)
    print(f"{len(keys):,} forecast origins, {keys[ID].nunique()} basins -> {a.out} "
          f"({a.out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f} s; validated against BASINSHIFT 8.4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
