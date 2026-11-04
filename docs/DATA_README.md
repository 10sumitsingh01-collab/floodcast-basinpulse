# BASINSHIFT data package v1 (Caravan v1.6, CAMELS-US): real observed data

Unzip into the repo as `data/caravan/` (already gitignored). Never commit these files.

## Files
| File | Rows | What |
|---|---|---|
| `train.parquet` | 2,357,018 | 239 train basins, 1981-10-01 to 2008-09-30. Inputs + targets |
| `test.parquet` | 1,246,829 | Forecast origins to predict: 239 train basins 2008-10 to 2014-09, plus 60 held-out basins 1981-2014. Inputs only |
| `static.parquet` | 299 | Catchment attributes per basin |
| `thresholds.csv` | 299 | `q_high` (95th pct) / `q_extreme` (99th pct) flow per basin, reference period 1981-10 to 2008-09. High-flow event = flow > q_high |
| `thresholds_val.csv` | 239 | Same, fit period only (up to 2006-09-30), for temporal validation |
| `splits.json` | | Temporal validation cutoff + 5 region-stratified basin folds (inside train) |
| `basin_selection.csv` | 671 | Every CAMELS basin: train / geo_test / not sampled / excluded + reason |
| `PREP_REPORT.md`, `LEAKAGE_REPORT.md` | | Counts, rules, 15 leakage checks (all pass) |
| `licenses/` | | CC-BY-4.0 texts; cite Caravan (Kratzert et al. 2023) and CAMELS (Newman et al. 2015) |

The true test answers (`test_labels.parquet`) are NOT in this zip. Lane D holds them for scoring only.

## Columns
- `gauge_id`: string, e.g. `camels_01013500`. Keep it a string.
- `date`: day t. **Each row is a forecast origin at the end of day t.**
- `streamflow`: flow on day t, mm/day, **already divided by catchment area. Do not divide by area again.** (train only)
- `streamflow_obs`: the same flow on day t, as an INPUT known at the origin (train and test)
- `streamflow_h1`, `_h3`, `_h7`: TARGETS = flow on day t+1, t+3, t+7 (train only; NaN where unobserved)
- Weather (ERA5-Land, basin-averaged, day t): `total_precipitation_sum` (mm), `temperature_2m_mean/min/max` (C),
  `potential_evaporation_sum_FAO_PENMAN_MONTEITH` (mm), `snow_depth_water_equivalent_mean` (mm),
  `surface_net_solar_radiation_mean` (W/m2), `dewpoint_temperature_2m_mean` (C),
  `volumetric_soil_water_layer_1_mean`, `volumetric_soil_water_layer_3_mean` (m3/m3)
- Static: `area_km2`, `gauge_lat`, `gauge_lon`; climate indices from 1981-2008 weather only (`p_mean`, `pet_mean`,
  `aridity`, `frac_snow`, `high_prec_freq`, `low_prec_freq`, `p_seasonality`); HydroATLAS: elevation `ele_mt_sav`,
  slope `slp_dg_sav`, forest `for_pc_sse`, cropland `crp_pc_sse`, urban `urb_pc_sse`, lakes `lka_pc_sse`,
  regulation `dor_pc_pva`, groundwater depth `gwt_cm_sav`, clay/sand/silt `cly/snd/slt_pc_sav`, soil carbon `soc_th_sav`,
  karst `kar_pc_sse`, permafrost `prm_pc_sse`, glacier `gla_pc_sse`, population `ppd_pk_sav`, soil water `swc_pc_syr`, snow `snw_pc_syr`

## Rules
1. A feature for row t may only use data from dates <= t (calendar date, per basin). Never shift(-k) or centred windows.
2. Never use `streamflow_h*` as a feature. In train, `streamflow` (day t) is fine as an input; it equals `streamflow_obs`.
3. Temporal validation: when fitting on dates <= 2006-09-30, drop rows whose t+h is after that date.
4. Held-out basins have NO training rows; their observed history is given as input at prediction time.
5. Predict each origin from its own row's features. A later row's `streamflow_obs` is an earlier row's target.
6. Predictions go to Lane D in long format: `gauge_id, date, horizon, predicted_discharge, lower, upper, p_high, p_extreme`
   (one row per origin x horizon 1/3/7; lower/upper = 10th/90th percentile; p_extreme <= p_high).
