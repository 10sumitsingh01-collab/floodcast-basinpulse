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
Final model `src.bs_models:RobustMeanModel`, scored once on the test set (`outputs/lgbm_final/REPORT.md`).
Headline set `geo_temporal` (60 basins never used in training, 2008-10 to 2014-09); persistence in brackets.

| | 1 day | 3 days | 7 days |
|---|---|---|---|
| NSE, median over basins | 0.63 (0.49) | 0.30 (−0.17) | 0.15 (−0.43) |
| KGE, median over basins | 0.70 (0.74) | 0.33 (0.42) | 0.15 (0.28) |
| MAE, mm/day | 0.37 (0.50) | 0.74 (0.90) | 0.90 (1.14) |
| High-flow AUPRC | 0.81 (0.61) | 0.45 (0.27) | 0.28 (0.16) |
| Extreme-flow AUPRC | 0.49 (0.19) | 0.14 (0.06) | 0.07 (0.03) |
| Brier skill, high flow | 0.65 (0.51) | 0.31 (0.22) | 0.20 (0.13) |
| 10–90% range coverage (target 0.80) | 0.81 (0.80) | 0.80 (0.80) | 0.79 (0.80) |

Seen basins in later years (`temporal`): NSE 0.68 / 0.26 / 0.15. Unseen basins, training years (`geo`):
0.66 / 0.28 / 0.17. Confidence intervals and event counts: `docs/TECHNICAL_REPORT.md` sections 7–8.
Ungauged variant (no flow input), validation only: NSE 0.33 / 0.13 / 0.07, still above persistence beyond 1 day.

## Robustness
Test set, unseen basins, 1-day horizon, change in median NSE vs clean inputs (persistence in brackets):
10% of inputs missing −3% (−6%); 30% missing −9% (−33%); 3–7 day gaps −9% (−14%); weather noise −2% (0%);
latest flow 1 day late −26% (−94%). At 3 and 7 days no scenario costs more than 13%. Weak point: with late flow
data the 1-day range covers only 60% instead of 80%. Details: `docs/TECHNICAL_REPORT.md` section 9.

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
- **Late or missing observations** degrade the short-horizon forecast most; a stale flow reading makes the
  1-day range over-confident (see robustness).
- **Sharp peaks** are underestimated in height by the point forecast; use the high-flow probability for alerts.
- **KGE** is below persistence: forecasts hedge toward the expected value and are smoother than real flow.

## Ethical considerations
Wrong confidence in a flood forecast can cause harm in both directions (missed floods, alarm fatigue).
We report calibration, false alarms and failure cases openly, and state that the system is not certified for
operational use.
