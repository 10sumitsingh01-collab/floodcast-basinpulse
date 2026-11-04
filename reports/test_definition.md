# Test Set Definition & Holdout Protocol (v2)

## 1. Exact Test Contract
- **Dataset Version**: `v2`
- **Stations**: 14 verified river gauge stations (all present in train).
- **Date Range**: 2022-01-01 to 2023-12-31 (2-year holdout).
- **Test Row Count**: 10,220 rows (14 stations × 730 days).
- **Holdout Labels Count**: 10,220 rows (exactly 1:1 match).
- **Temporal Separation**:
  - Last Train Date: `2021-12-31`
  - Test Start Date: `2022-01-01`
  - Strict temporal ordering verified: `last_train_date < test_start_date`.

## 2. Column Structure
- `test_features.parquet` / `test.csv`: Exactly matches training feature columns with `discharge_cfs` omitted.
  Columns (12): `['station_id', 'date', 'precipitation_mm', 'temp_max_c', 'temp_min_c', 'soil_moisture', 'drainage_area_sqmi', 'mean_elevation_ft', 'slope_pct', 'forest_cover_pct', 'elev_range_ft', 'elev_std_ft']`
- `holdout_labels.parquet`: Ground-truth labels (`station_id`, `date`, `discharge_cfs`) for evaluation only.
- Target Leakage Check: `discharge_cfs in test_df.columns` == **False**.
