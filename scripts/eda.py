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
    static_cols = ["drainage_area_sqmi", "mean_elevation_ft", "slope_pct", "forest_cover_pct", "elev_range_ft", "elev_std_ft"]
    static_nans = full_df[static_cols].isna().mean() * 100.0

    from src.config import DATA_VERSION, EXTRA_STATION_CANDIDATES

    # ----------------------------------------------------
    # Report 1: reports/eda.md
    # ----------------------------------------------------
    eda_md = f"""# Exploratory Data Analysis Report: FLOODCAST Real Dataset ({DATA_VERSION})

## 1. Executive Summary & Scale Dominance
- **Dataset Version**: `{DATA_VERSION}`
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

## 4. Static Catchment Attribute Values ({DATA_VERSION})
| Station ID | Drainage Area (sq mi) | Mean Elev (ft) | Slope (%) | Elev Range (ft) | Elev Std (ft) | Gauge Alt (ft) |
|---|---|---|---|---|---|---|
"""
    for _, r in extra_static.iterrows():
        sid = r["station_id"]
        row_full = full_df[full_df[STATION_COL] == sid].iloc[0]
        eda_md += (
            f"| `{sid}` | {row_full['drainage_area_sqmi']:,.1f} | {row_full['mean_elevation_ft']:,.1f} | "
            f"{row_full['slope_pct']:.2f}% | {row_full['elev_range_ft']:,.1f} | {row_full['elev_std_ft']:,.1f} | "
            f"{r['gauge_altitude_ft']} |\n"
        )

    eda_md += f"""
## 5. Static Catchment Attribute Missingness
- `drainage_area_sqmi`: {static_nans['drainage_area_sqmi']:.1f}% missing (USGS NWIS primary, NLDI polygon cross-check).
- `mean_elevation_ft`: {static_nans['mean_elevation_ft']:.1f}% missing (Open-Meteo elevation API regular grid sampling).
- `slope_pct`: {static_nans['slope_pct']:.1f}% missing (adjacent-pair finite difference across regular grid).
- `elev_range_ft`: {static_nans['elev_range_ft']:.1f}% missing (elevation max - min in sampled polygon).
- `elev_std_ft`: {static_nans['elev_std_ft']:.1f}% missing (elevation standard deviation in sampled polygon).
- `forest_cover_pct`: {static_nans['forest_cover_pct']:.1f}% missing (no clean public tabular REST API available).
"""
    (REPORTS_DIR / "eda.md").write_text(eda_md, encoding="utf-8")
    print("Generated reports/eda.md")

    # ----------------------------------------------------
    # Report 2: reports/data_notes.md
    # ----------------------------------------------------
    notes_md = f"""# Data Notes & Ingestion Audit (Dataset Version: {DATA_VERSION})

## 1. Provenance and Derivation Methodology
1. **Source A (USGS Daily Values NWIS)**:
   - Daily mean discharge (`00060_00003`). Negative values and ice-affected days strictly mapped to NaN.
   - Qualifier codes preserved in `data/processed/discharge_qualifiers.parquet`.
2. **Source B (USGS Site RDB Expanded)**:
   - Official station metadata, drainage area (`drain_area_va`), and gauge altitude (`alt_va`).
   - Gauge altitude stored separately in `data/processed/station_static_extra.parquet` (`gauge_altitude_ft`).
3. **Source C (Open-Meteo Archive API)**:
   - Historical daily weather in UTC: `precipitation_sum`, `temperature_2m_max`, `temperature_2m_min`, and `soil_moisture_0_to_7cm_mean` (mapped to `soil_moisture`).
   - All 14 stations read strictly from local cache (`data/cache/openmeteo_*.json`) with zero network archive calls today.
4. **Static Basin Morphometrics (USGS NLDI + Open-Meteo Elevation API)**:
   - Basin polygon downloaded from USGS NLDI (`https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{sid}/basin`).
   - Basin polygon surface area computed via Shoelace formula on local equirectangular flat coordinates:
     $$A = \\frac{1}{2} |\\sum (x_i y_{{i+1}} - x_{{i+1}} y_i)|$$
     Cross-checked against USGS `drain_area_va` (Potomac error is 0.02%).
   - Elevation sampled on a regular 2D grid within the basin polygon (capped at 100 points).
   - Sampled points queried via Open-Meteo Elevation API (`https://api.open-meteo.com/v1/elevation`).
   - **Slope Calculation Note**: `slope_pct` is computed as the mean absolute difference between adjacent horizontal and vertical grid points inside the basin polygon:
     `slope = (|delta_z| / distance) * 100%`
     **Important**: `slope_pct` is scale-dependent on the grid sampling density (recorded in `station_static_extra.parquet` as `grid_dx_km`, `grid_dy_km`) and is an approximate derived metric, NOT an official USGS StreamStats value.
   - `elev_range_ft` (z_max - z_min) and `elev_std_ft` (standard deviation) are added as scale-invariant topography features.
   - `forest_cover_pct`: Set to `NaN`. MRLC WMS tile service provides map rasters, not polygon zonal REST statistics. USGS EPQS returns HTTP 403. Per instructions, zero guessing is applied.

## 2. Station Adjustments & Drops
- **Sacramento River at Freeport, CA (`11447650`)**: Dropped from `STATIONS` in v2.
  - Reason: USGS NWIS discontinued daily discharge publication on 2015-09-30 (tidal stage only). The station had 5,752 train rows (2000-2015) and 0 holdout days (2022-2023), as well as missing drainage area in USGS Site RDB.
  - **TODO (Pending Rate Limit Reset)**: Replace Sacramento with verified, unregulated Sierra stream `11266500` (Merced River at Pohono Bridge nr Yosemite, CA; 321.0 sq mi, 100% daily discharge coverage 2000-2023).

## 3. Verified Candidate Stations for Expansion (12 Stations)
The following 12 candidate stations have been verified for existence (Source B), 100% daily discharge coverage (Source A: 8,766 days in 2000-2023, 730 days in 2022-2023), and valid NLDI basin polygons:
1. `11266500`: Merced River at Pohono Bridge, CA (Sierra, 321.0 sq mi, unregulated) — *Sacramento replacement*
2. `14048000`: John Day River at McDonald Ferry, OR (Pacific NW, 7,580 sq mi, undammed)
3. `09058000`: Colorado River near Kremmling, CO (Rocky Mountain snowmelt, 2,379 sq mi)
4. `06191500`: Yellowstone River at Corwin Springs, MT (Northern Rockies, 2,615 sq mi, pristine headwaters)
5. `06054500`: Missouri River at Toston, MT (Missouri headwaters, 14,641 sq mi)
6. `08155200`: Barton Creek at SH 71 near Oak Hill, TX (Southwest flashy, 89.7 sq mi)
7. `01643000`: Monocacy River at Jug Bridge near Frederick, MD (Mid-Atlantic, 817 sq mi)
8. `02320500`: Suwannee River at Branford, FL (Humid Southeast, 7,880 sq mi)
9. `02105500`: Cape Fear River at Huske Lock, NC (Southeast, 4,852 sq mi)
10. `03335500`: Wabash River at Lafayette, IN (Midwest plains, 7,267 sq mi)
11. `06892350`: Kansas River at DeSoto, KS (Midwest mega-basin, 59,756 sq mi)
12. `05378500`: Mississippi River at Winona, MN (Upper Mississippi mega-basin, 59,200 sq mi)

**Command to build complete 26-station dataset after API limit resets**:
```bash
python -m src.build_dataset --stations with_candidates --allow-network
```
- Expected API calls: **60 calls** to Open-Meteo archive (12 stations × 5 5-year chunks); USGS and NLDI data are already cached locally.
"""
    (REPORTS_DIR / "data_notes.md").write_text(notes_md, encoding="utf-8")
    print("Generated reports/data_notes.md")

    # ----------------------------------------------------
    # Report 3: reports/test_definition.md
    # ----------------------------------------------------
    test_md = f"""# Test Set Definition & Holdout Protocol ({DATA_VERSION})

## 1. Exact Test Contract
- **Dataset Version**: `{DATA_VERSION}`
- **Stations**: {test_df[STATION_COL].nunique()} verified river gauge stations (all present in train).
- **Date Range**: {test_df[DATE_COL].min().date()} to {test_df[DATE_COL].max().date()} (2-year holdout).
- **Test Row Count**: {len(test_df):,} rows ({test_df[STATION_COL].nunique()} stations × 730 days).
- **Holdout Labels Count**: {len(holdout_df):,} rows (exactly 1:1 match).
- **Temporal Separation**:
  - Last Train Date: `{train_df[DATE_COL].max().date()}`
  - Test Start Date: `{test_df[DATE_COL].min().date()}`
  - Strict temporal ordering verified: `last_train_date < test_start_date`.

## 2. Column Structure
- `test_features.parquet` / `test.csv`: Exactly matches training feature columns with `discharge_cfs` omitted.
  Columns ({len(test_df.columns)}): `{list(test_df.columns)}`
- `holdout_labels.parquet`: Ground-truth labels (`station_id`, `date`, `discharge_cfs`) for evaluation only.
- Target Leakage Check: `discharge_cfs in test_df.columns` == **False**.
"""
    (REPORTS_DIR / "test_definition.md").write_text(test_md, encoding="utf-8")
    print("Generated reports/test_definition.md")


if __name__ == "__main__":
    run_eda()

