# Test Set Definition & Holdout Protocol

## Exact Test Contract
- **Stations**: 14 stations with verified holdout labels (all 15 ingested stations except mandatory Sacramento `11447650`, whose USGS NWIS daily discharge was discontinued in 2015).
- **Date Range**: 2022-01-01 to 2023-12-31 (2-year holdout).
- **Test Row Count**: 10,220 rows (14 stations × 730 days).
- **Holdout Labels Count**: 10,220 rows (exactly 1:1 match).
- **Temporal Separation**:
  - Last Train Date: `2021-12-31`
  - Test Start Date: `2022-01-01`
  - Strict temporal ordering verified: `last_train_date < test_start_date`.

## Column Structure
- `test_features.parquet` / `test.csv`: Exactly matches training feature columns with `discharge_cfs` omitted.
- `holdout_labels.parquet`: Ground-truth labels (`station_id`, `date`, `discharge_cfs`) for evaluation only.
