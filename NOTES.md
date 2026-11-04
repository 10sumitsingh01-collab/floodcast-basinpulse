# Team Notes & Coordination Log

## Phase 0 Real Data Status & Zero-Simulation Rule
- **CRITICAL RULE**: NEVER invent or fabricate data. All simulator code and synthetic files have been completely purged from the repository.
- **Lane A (`src/data.py`)**: `generate_starter_dataset` deleted. `load_train()`, `load_test()`, and `load_station_metadata()` strictly load real CSV files; if missing, they raise `FileNotFoundError` stating real data is pending.
- **Lane B (`src/features.py`)**, **Lane C (`src/models.py`)**, **Lane D (`src/validate.py`, `solution.py`)**: Preserved as built, but no model training or scoring will be performed until real data is downloaded and verified.
- **Metrics**: All metrics in `model_card.md` and `README.md` have been reset to `TBD: real data pending`. `experiments/results.csv` cleared.

## Phase 2 Dataset Build (v2) Hand-off Notes (Lane A -> Lanes B, C, D)
- **Dataset Version**: `v2`
- **Stations**: 14 active stations (Sacramento 11447650 dropped due to discontinued USGS NWIS record and lack of holdout observations).
  - Train: 112,503 rows (2000-01-01 to 2021-12-31). Zero missing weather; discharge present on all rows.
  - Test: 10,220 rows (2022-01-01 to 2023-12-31; 14 stations × 730 days). Strictly no discharge column.
  - Holdout: 10,220 rows (1:1 ground truth).
- **Static Attributes Update for Lane B**:
  - `drainage_area_sqmi`: 100% complete (0 NaNs across train and test).
  - `mean_elevation_ft`: 100% complete (0 NaNs; derived via USGS NLDI polygon + Open-Meteo elevation API).
  - `slope_pct`: 100% complete (0 NaNs; finite difference across adjacent grid points; note: scale-dependent).
  - `elev_range_ft` and `elev_std_ft`: 100% complete (0 NaNs; scale-invariant topographical metrics added after forest_cover_pct).
  - `forest_cover_pct`: 100% NaN (no tabular REST zonal stats API exists).
  - Extra features in `data/processed/station_static_extra.parquet`: `gauge_altitude_ft`, `latitude`, `longitude`, `polygon_area_sqmi`, `grid_dx_km`, `grid_dy_km`, `n_points`.
- **Expansion Candidates**:
  - 12 candidates verified for USGS DV and NLDI. Rebuilding with candidates requires 60 Open-Meteo archive calls after rate-limit reset.

