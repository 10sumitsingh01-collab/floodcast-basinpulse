# Model card: BasinPulse (BASINSHIFT entry)

## Model details
- **What:** daily river-flow forecaster for 1, 3 and 7 days ahead, with a 10–90% range and probabilities of
  high flow (above the basin's 95th percentile) and extreme flow (above its 99th percentile).
- **Type:** gradient-boosted trees (LightGBM 4.7). Per horizon: three quantile regressors (10/50/90%),
  conformal widening of the range, two isotonic-calibrated event classifiers.
- **Target:** log change from the latest observed flow, `log1p(flow[t+h]) − log1p(flow[t])`.
- **Code:** `src/bs_models.py` (model and features), `basinshift_run.py` (training, validation, scoring).
- **Version / commit:** see `experiments/basinshift_results.csv` for the final run's commit hash.
- **Team:** BasinPulse. Licence of code: as chosen by the team. Data licence: CC-BY-4.0 (Caravan, CAMELS-US).

## Intended use
- Research benchmark entry for BASINSHIFT: studying how well flow forecasts transfer to basins not seen in
  training, and how well their uncertainty is calibrated.
- **Not intended for:** flood warnings, evacuation, emergency response, reservoir or infrastructure operation,
  or any decision affecting people's safety (BASINSHIFT 16.1). Not a replacement for official forecasts.

## Inputs
At forecast origin t (end of day t), for one basin: observed flow up to day t; ERA5-Land basin-averaged weather
up to day t (precipitation, temperature, potential evaporation, snow water equivalent, soil water, radiation,
dewpoint); 28 static catchment attributes (area, location, climate indices from 1981–2008 weather, HydroATLAS
terrain, soil, land cover, regulation); the basin's flow thresholds. No input uses data after day t.

## Training data
Caravan v1.6, CAMELS-US subset: 239 basins, water years 1981-10 to 2008-09, 2,357,018 daily origins
(300,000 sampled per model; 20% of basins kept out of the trees for calibration). Real observations only.
See `docs/DATA_PROVENANCE.md`.

## Evaluation data
| Set | Basins | Period |
|---|---|---|
| `temporal` | 239 training basins | 2008-10 to 2014-09 |
| `geo` | 60 basins never used in training | 1981-10 to 2008-09 |
| `geo_temporal` | the same 60 | 2008-10 to 2014-09 |
All model choices were made on validation splits inside the training data. The test set was scored once.

## Metrics
NSE and KGE (median over basins), MAE, AUPRC for high and extreme flow (with event counts), Brier skill score,
recall / precision / false alarms per basin-year at probability 0.5, 10–90% range coverage and width,
pinball loss; 95% confidence intervals from resampling basins. Definitions: `docs/BASINSHIFT_EVALUATION.md`.

## Results
FINAL_RESULTS_PLACEHOLDER

## Robustness
STRESS_PLACEHOLDER

## Limitations and risks
- **One country:** CAMELS-US basins only, mostly small-to-medium natural catchments (median 330 km²). No evidence
  for very large, heavily regulated, tropical or arid-zone rivers outside the US.
- **Snowmelt basins** are under-represented in the held-out set (90th percentile snow fraction 0.26 vs 0.51 in
  training), so transfer to snow-dominated regimes is tested less.
- **Gauged at prediction time:** held-out basins have no training data, but their past flow is an input.
  For truly ungauged basins see the no-flow-history variant, which is weaker.
- **Reanalysis weather:** ERA5-Land arrives with a few days' delay; an operational version would use forecast
  weather, which has larger errors.
- **Thresholds** are statistical (95th / 99th percentile), not official flood stages, and say nothing about
  flood depth, extent or damage.
- **Late or missing observations** degrade the short-horizon forecast most (see robustness).

## Ethical considerations
Wrong confidence in a flood forecast can cause harm in both directions (missed floods, alarm fatigue).
We report calibration, false alarms and failure cases openly, and state that the system is not certified for
operational use.
