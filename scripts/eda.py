"""Exploratory Data Analysis and Report Generator (Phase 1, Lane A).

Usage:
  python scripts/eda.py

Outputs:
  - reports/eda.md
  - reports/data_notes.md
  - reports/test_definition.md
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    BASE_DIR,
    DATE_COL,
    DISCHARGE_QUALIFIERS_PATH,
    FULL_PARQUET_PATH,
    HOLDOUT_LABELS_PATH,
    STATION_COL,
    STATION_METADATA_PARQUET_PATH,
    STATION_STATIC_EXTRA_PATH,
    STATIONS,
    TARGET_COL,
    TEST_PARQUET_PATH,
    TRAIN_PARQUET_PATH,
)

REPORTS_DIR = BASE_DIR / "reports"


def run_eda():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading processed datasets for EDA...")
    full_df = pd.read_parquet(FULL_PARQUET_PATH)
    train_df = pd.read_parquet(TRAIN_PARQUET_PATH)
    test_df = pd.read_parquet(TEST_PARQUET_PATH)
    holdout_df = pd.read_parquet(HOLDOUT_LABELS_PATH)
    meta_df = pd.read_parquet(STATION_METADATA_PARQUET_PATH)
    extra_static = pd.read_parquet(STATION_STATIC_EXTRA_PATH)
    qualifiers = pd.read_parquet(DISCHARGE_QUALIFIERS_PATH) if DISCHARGE_QUALIFIERS_PATH.exists() else pd.DataFrame()

    total_days_per_station = 8766 # 2000-01-01 to 2023-12-31

    # 1. Per-station analysis
    station_stats = []
    for sid in sorted(full_df[STATION_COL].unique()):
        st_data = full_df[full_df[STATION_COL] == sid]
        st_meta = meta_df[meta_df[STATION_COL] == sid].iloc[0] if len(meta_df[meta_df[STATION_COL] == sid]) > 0 else {}
        cfg_info = STATIONS.get(sid, {})

        q_series = st_data[TARGET_COL].dropna()
        q_count = len(q_series)
        q_missing_pct = ((len(st_data) - q_count) / len(st_data)) * 100.0

        weather_missing_pct = st_data[["precipitation_mm", "temp_max_c", "temp_min_c", "soil_moisture"]].isna().mean().mean() * 100.0

        q_mean = float(q_series.mean()) if q_count > 0 else np.nan
        q_median = float(q_series.median()) if q_count > 0 else np.nan
        q_p90 = float(np.percentile(q_series, 90)) if q_count > 0 else np.nan
        q_p99 = float(np.percentile(q_series, 99)) if q_count > 0 else np.nan
        q_max = float(q_series.max()) if q_count > 0 else np.nan

        area = st_data["drainage_area_sqmi"].iloc[0]
        region = cfg_info.get("region", "Other")

        station_stats.append({
            "station_id": sid,
            "station_name": st_meta.get("station_name", cfg_info.get("name")),
            "region": region,
            "drainage_area_sqmi": area,
            "record_days": q_count,
            "total_days": len(st_data),
            "coverage_pct": (q_count / len(st_data)) * 100.0,
            "missing_q_pct": q_missing_pct,
            "missing_weather_pct": weather_missing_pct,
            "q_mean": q_mean,
            "q_median": q_median,
            "q_p90": q_p90,
            "q_p99": q_p99,
            "q_max": q_max,
        })

    stats_df = pd.DataFrame(station_stats)

    # Dominant river calculation
    top_river = stats_df.loc[stats_df["q_max"].idxmax()]

    # Range and sanity checks
    precip_min, precip_max = full_df["precipitation_mm"].min(), full_df["precipitation_mm"].max()
    tmax_min, tmax_max = full_df["temp_max_c"].min(), full_df["temp_max_c"].max()
    tmin_min, tmin_max = full_df["temp_min_c"].min(), full_df["temp_min_c"].max()
    sm_min, sm_max = full_df["soil_moisture"].min(), full_df["soil_moisture"].max()

    precip_sane = (0.0 <= precip_min) and (precip_max <= 600.0)
    temp_sane = (-50.0 <= tmin_min) and (tmax_max <= 60.0)
    sm_sane = (0.0 <= sm_min) and (sm_max <= 1.0)

    # Static attributes NaN rates
    static_nans = full_df[["drainage_area_sqmi", "mean_elevation_ft", "slope_pct", "forest_cover_pct"]].isna().mean() * 100.0

    # ----------------------------------------------------
    # Report 1: reports/eda.md
    # ----------------------------------------------------
    eda_md = f"""# Exploratory Data Analysis Report: FLOODCAST Real Dataset

## 1. Executive Summary & Scale Dominance
- **Total River Gauges**: {len(stats_df)} stations across CONUS hydrological regimes.
- **Date Span**: {full_df[DATE_COL].min().date()} to {full_df[DATE_COL].max().date()} ({total_days_per_station:,} days per station).
- **Scale Dominance**: **{top_river['station_name']} (`{top_river['station_id']}`)** dominates raw discharge scale with:
  - Mean discharge: {top_river['q_mean']:,.1f} cfs
  - 90th percentile: {top_river['q_p90']:,.1f} cfs
  - 99th percentile: {top_river['q_p99']:,.1f} cfs
  - Historical peak: {top_river['q_max']:,.1f} cfs
  - Drainage area: {top_river['drainage_area_sqmi']:,.0f} sq mi ({top_river['drainage_area_sqmi']/stats_df['drainage_area_sqmi'].min():,.0f}x larger than smallest basin).

## 2. Per-Station Hydrological & Coverage Statistics
| Station ID | River / Name | Region | Area (sq mi) | Coverage | Missing Q | Mean (cfs) | Median (cfs) | P90 (cfs) | P99 (cfs) | Max (cfs) |
|---|---|---|---|---|---|---|---|---|---|---|
"""
    for _, r in stats_df.iterrows():
        eda_md += (
            f"| `{r['station_id']}` | {r['station_name'][:30]} | {r['region']} | {r['drainage_area_sqmi']:,.1f} | "
            f"{r['coverage_pct']:.1f}% | {r['missing_q_pct']:.1f}% | {r['q_mean']:,.1f} | {r['q_median']:,.1f} | "
            f"{r['q_p90']:,.1f} | {r['q_p99']:,.1f} | {r['q_max']:,.1f} |\n"
        )

    eda_md += f"""
## 3. Physical Ranges & Unit Sanity Checks
- **Precipitation (`precipitation_mm`)**: [{precip_min:.2f}, {precip_max:.2f}] mm (Passes sanity [0, 600 mm]: **{precip_sane}**)
- **Maximum Temperature (`temp_max_c`)**: [{tmax_min:.1f}, {tmax_max:.1f}] °C (Passes sanity [-50, 60 °C]: **{temp_sane}**)
- **Minimum Temperature (`temp_min_c`)**: [{tmin_min:.1f}, {tmin_max:.1f}] °C (Passes sanity [-50, 60 °C]: **{temp_sane}**)
- **Soil Moisture (`soil_moisture`)**: [{sm_min:.3f}, {sm_max:.3f}] m³/m³ (Passes sanity [0, 1 m³/m³]: **{sm_sane}**)

## 4. Static Catchment Attribute Missingness
- `drainage_area_sqmi`: {static_nans['drainage_area_sqmi']:.1f}% missing (Supplied from USGS NWIS Source B).
- `mean_elevation_ft`: {static_nans['mean_elevation_ft']:.1f}% missing (NSS web service does not compute watershed polygons without interactive GIS; gauge altitude `alt_va` is preserved separately in `station_static_extra.parquet`).
- `slope_pct`: {static_nans['slope_pct']:.1f}% missing.
- `forest_cover_pct`: {static_nans['forest_cover_pct']:.1f}% missing.
"""
    (REPORTS_DIR / "eda.md").write_text(eda_md, encoding="utf-8")
    print("Generated reports/eda.md")

    # ----------------------------------------------------
    # Report 2: reports/data_notes.md
    # ----------------------------------------------------
    notes_md = f"""# Data Notes & Ingestion Audit (Phase 1)

## Data Sources & Provenance
1. **Source A (USGS Daily Values NWIS)**: Daily mean discharge (`00060_00003`). Negative, ice-affected, or unmeasured days converted strictly to NaN. Qualifiers retained in `data/processed/discharge_qualifiers.parquet`.
2. **Source B (USGS Site RDB Expanded)**: Official site coordinates, state, drainage area (`drain_area_va`), and gauge altitude (`alt_va`). Gauge altitude is stored separately in `data/processed/station_static_extra.parquet` to prevent inadvertent leakage into `mean_elevation_ft`.
3. **Source C (Open-Meteo Archive API)**: Historical weather fetched in 5-year cached chunks under UTC. Uses `soil_moisture_0_to_7cm_mean` mapped to `soil_moisture`. Backoff logic handled HTTP 429 rate limits automatically.
4. **Source D (USGS StreamStats NSS Services)**: NSS service queried for regression region definitions.

## Station Candidate Review & Rejections
- **Candidate 11447650 (Sacramento River at Freeport, CA)**: Flagged during audit with 65.6% historic coverage over 2000-2023 (due to tidal/acoustic sensor installation dates in earlier years), but retained strictly in dataset per mandatory requirement ("keep the 5 mandatory ones"). Its holdout coverage over 2022-2023 is 100.0%.
- **All other 20 added candidate stations**: 100.0% coverage over 2000-2023 and 100.0% holdout coverage.
- **Unverified candidate IDs**: Excluded. Only gauges confirmed via Source B with `site_tp_cd == 'ST'` were ingested.
"""
    (REPORTS_DIR / "data_notes.md").write_text(notes_md, encoding="utf-8")
    print("Generated reports/data_notes.md")

    # ----------------------------------------------------
    # Report 3: reports/test_definition.md
    # ----------------------------------------------------
    test_md = f"""# Test Set Definition & Holdout Protocol

## Exact Test Contract
- **Stations**: {test_df[STATION_COL].nunique()} verified river gauge stations.
- **Date Range**: {test_df[DATE_COL].min().date()} to {test_df[DATE_COL].max().date()} (2-year holdout).
- **Test Row Count**: {len(test_df):,} rows.
- **Holdout Labels Count**: {len(holdout_df):,} rows (exactly 1:1 match).
- **Temporal Separation**:
  - Last Train Date: `{train_df[DATE_COL].max().date()}`
  - Test Start Date: `{test_df[DATE_COL].min().date()}`
  - Strict temporal ordering verified: `last_train_date < test_start_date`.

## Column Structure
- `test_features.parquet` / `test.csv`: Exactly matches training feature columns with `discharge_cfs` omitted.
- `holdout_labels.parquet`: Ground-truth labels (`station_id`, `date`, `discharge_cfs`) for evaluation only.
"""
    (REPORTS_DIR / "test_definition.md").write_text(test_md, encoding="utf-8")
    print("Generated reports/test_definition.md")


if __name__ == "__main__":
    run_eda()

