# Data Notes & Ingestion Audit (Dataset Version: v2)

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
   - Basin polygon downloaded from USGS NLDI (`https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-12119000/basin`).
   - Basin polygon surface area computed via Shoelace formula on local equirectangular flat coordinates:
     $$A = \frac12 |\sum (x_i y_{i+1} - x_{i+1} y_i)|$$
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
