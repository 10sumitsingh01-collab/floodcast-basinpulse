# Team Notes & Coordination Log

## Phase 0 Real Data Status & Zero-Simulation Rule
- **CRITICAL RULE**: NEVER invent or fabricate data. All simulator code and synthetic files have been completely purged from the repository.
- **Lane A (`src/data.py`)**: `generate_starter_dataset` deleted. `load_train()`, `load_test()`, and `load_station_metadata()` strictly load real CSV files; if missing, they raise `FileNotFoundError` stating real data is pending.
- **Lane B (`src/features.py`)**, **Lane C (`src/models.py`)**, **Lane D (`src/validate.py`, `solution.py`)**: Preserved as built, but no model training or scoring will be performed until real data is downloaded and verified.
- **Metrics**: All metrics in `model_card.md` and `README.md` have been reset to `TBD: real data pending`. `experiments/results.csv` cleared.

## Phase 1 Dataset Build & Hand-off Notes (Lane A -> Lanes B, C, D)
- **Real Dataset Built**: 15 USGS stream stations across CONUS spanning 2000-01-01 to 2023-12-31.
  - Train: 118,255 rows (2000-01-01 to 2021-12-31). Zero missing weather values; discharge present for all train rows.
  - Test: 10,220 rows (2022-01-01 to 2023-12-31). Strictly no `discharge_cfs` column.
  - Holdout Ground Truth: `data/holdout/holdout_labels.parquet` (10,220 rows) for offline evaluation by Lane D.
- **Station Static Attributes Note for Lane B**:
  - `drainage_area_sqmi` is 100% complete (sourced from USGS NWIS site info `drain_area_va`).
  - `mean_elevation_ft`, `slope_pct`, `forest_cover_pct` are `NaN` for all stations because StreamStats NSS scenario endpoint only provides regional regression equations rather than delineated basin morphometrics.
  - Extra feature available: `data/processed/station_static_extra.parquet` provides `gauge_altitude_ft` (`alt_va` from USGS NWIS). Lane B can join on `station_id` if desired.
- **Discharge Scale Warning for Lane C**:
  - Station `07010000` (Mississippi River at St. Louis, MO; 697,000 sq mi) has discharge up to 500,000+ cfs, whereas flashy creek `08158000` has median ~16 cfs. Target transformation (e.g. `log1p(discharge_cfs)` or basin-area normalized specific discharge `cfs / sqmi`) is strongly advised before regression modeling.
- **Discharge Quality Codes**:
  - Available in `data/processed/discharge_qualifiers.parquet` for any fold filtering or weight boosting.
- **Station Review**:
  - Station `11447650` (Sacramento River at Freeport, CA) has 5,200 valid discharge days in train (record discontinued by USGS on 2015-09-30) and 0 days in holdout period (2022-2023). Retained in train per mandatory requirement; test set has 0 rows for this station per holdout existence rule.
