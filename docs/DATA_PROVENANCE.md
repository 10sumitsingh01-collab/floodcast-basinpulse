# Data provenance (BASINSHIFT 6.3, 11.1, deliverable 5)

All training and evaluation data is **real**: observed river discharge and reanalysis weather.
No synthetic data is used anywhere (training, validation, tests or reported results).

## Source
| Item | Value |
|---|---|
| Dataset | Caravan, a global community dataset for large-sample hydrology, **v1.6, CSV version** |
| Archive | https://zenodo.org/records/15530022 (`Caravan-csv.tar.gz`, 28,951,089,034 bytes) |
| Paper | https://doi.org/10.1038/s41597-023-01975-w |
| License | CC-BY-4.0 (Caravan and CAMELS-US); license texts shipped in the package `licenses/` folder |
| Subset | `camels` source: 671 CAMELS-US basins (USGS gauges) |
| Retrieved | 2026-11-04 by `scripts/caravan/fetch_caravan.py` (first 1.5 GB of the archive, which contains all attributes, licenses and every CAMELS-US time series; per-file sha256 in `fetch_manifest.json`) |
| Variables used | daily `streamflow` (mm/day, observed, USGS via CAMELS); 10 ERA5-Land basin-averaged weather variables; HydroATLAS catchment attributes; gauge coordinates and area |
| Period | water years 1981-10-01 to 2014-09-30 (CAMELS-US streamflow ends in 2014) |

## Citations (required by CC-BY; also in the technical report)
- Kratzert, F., et al. (2023). Caravan - A global community dataset for large-sample hydrology. *Scientific Data*, 10, 61.
- Newman, A. J., et al. (2015). Development of a large-sample watershed-scale hydrometeorological dataset for the contiguous USA. *Hydrology and Earth System Sciences*, 19, 209-223.
- Addor, N., et al. (2017). The CAMELS data set: catchment attributes and meteorology for large-sample studies. *HESS*, 21, 5293-5313.
- Linke, S., et al. (2019). Global hydro-environmental sub-basin and river reach characteristics at high spatial resolution (HydroATLAS). *Scientific Data*, 6, 283.
- Muñoz-Sabater, J., et al. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. *Earth System Science Data*, 13, 4349-4383.

## Transformations (all version-controlled, `scripts/caravan/prepare_caravan.py`)
1. Window 1981-10-01 to 2014-09-30. Negative streamflow set to missing (0 values in this subset). Missing flow never filled.
2. Basin QA: keep if observed flow covers >= 90% of the train period and >= 50% of the test period, and weather is <= 5% missing.
   671 → **611 eligible, 60 excluded** (reason per basin in `basin_selection.csv`).
3. Sampling: 299 basins in proportion to 14 USGS regions (seed 42); 60 of them held out entirely, again per region.
4. Climate indices (`p_mean`, `pet_mean`, `aridity`, `frac_snow`, `high_prec_freq`, `low_prec_freq`, `p_seasonality`)
   **recomputed from 1981-2008 weather only**. Caravan's own indices span the full record including our test years.
5. HydroATLAS attributes derived from **modelled discharge or runoff** (`dis_*`, `run_mm_syr`, `riv_tc_usu`, `ria_ha_usu`) removed.
6. Targets `streamflow_h1/_h3/_h7` = observed flow 1/3/7 days after each origin, same basin, by calendar date.
   Training targets never reach into the test period.
7. Thresholds `q_high`/`q_extreme` = each basin's 95th/99th percentile flow over 1981-10 to 2008-09, fixed before modelling.
8. Leakage audit: 15 checks in `scripts/caravan/check_leakage.py`, all pass (`LEAKAGE_REPORT.md`).

## Shared package
`basinshift_caravan_v1.zip`, 65 MB, sha256 `45a4bba1a2593e82ba43f718b05db365d0f605ece6b2e530f400abd61f27e849`.
Contains everything except `test_labels.parquet`, which only the evaluation (Lane D) holds.

## External data, pretrained models, synthetic data (11.1)
| Type | Used? |
|---|---|
| Other external datasets | No |
| Pretrained weights / foundation models | No |
| Synthetic data | No |
| Data v2 (earlier team dataset, USGS NWIS + Open-Meteo at gauge points) | Superseded; not used for any reported result |
