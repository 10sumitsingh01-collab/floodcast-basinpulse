# BasinPulse: river-flow forecasting that transfers to unseen basins

**Team:** BasinPulse | **Challenge:** BASINSHIFT, the Global Flood Intelligence Challenge

For every river basin and every day *t*, BasinPulse forecasts the river's flow **1, 3 and 7 days ahead**,
gives a **10–90% range**, and estimates the **probability of high and extreme flow**. The headline test is on
**60 basins the model never saw during training**, in years after the training period.

> **Responsible use:** this is an experimental research benchmark entry. Its outputs must not be used as the
> sole basis for evacuation, emergency response, infrastructure operation or public warnings (BASINSHIFT 16.1).

## Data: real observations only
[Caravan v1.6](https://zenodo.org/records/15530022) (CC-BY-4.0), CAMELS-US subset: observed USGS daily
streamflow, ERA5-Land basin-averaged weather, HydroATLAS catchment attributes. No synthetic data anywhere.

| | Basins | Period | Rows |
|---|---|---|---|
| Train | 239 | 1981-10 to 2008-09 | 2,357,018 |
| Test `temporal` | same 239 | 2008-10 to 2014-09 | 523,649 |
| Test `geo` | **60 never in training** | 1981-10 to 2008-09 | 591,720 |
| Test `geo_temporal` (headline) | the same 60 | 2008-10 to 2014-09 | 131,460 |

Held-out basins are spread across all 14 USGS regions and match the training basins on area, aridity,
elevation, slope and forest cover ([split summary](docs/SPLIT_SUMMARY.md), [map](docs/basin_map.png)).
15 automated leakage checks pass. Details: [data README](docs/DATA_README.md), [provenance](docs/DATA_PROVENANCE.md).

## Method
- **Target:** the log change from today's observed flow to the flow in *h* days. Predicting the change instead of
  the level lets one model serve basins whose flows differ by orders of magnitude.
- **Forecast and range:** LightGBM quantile regression (10th / 50th / 90th percentile) per horizon, widened by
  conformal calibration on basins held out from the trees, so the range covers ~80% on unseen basins.
- **Flood probabilities:** LightGBM classifiers for flow above each basin's 95th / 99th percentile,
  isotonic-calibrated on the same held-out basins.
- **Inputs:** past observed flow, rain (lags, sums, antecedent index), temperature, snow and soil-water dynamics,
  evaporation, season, catchment attributes. Every input uses data up to day *t* only.
- **Reference:** calibrated persistence (tomorrow's flow = today's), the baseline every model must beat.
- **Research question (H1):** do catchment attributes improve transfer to unseen basins? Tested by an ablation.

## Results
All numbers come from logged runs (`experiments/`) with their commit hash. Full tables with 95% bootstrap
confidence intervals: `outputs/<run>/REPORT.md`.

Persistence baseline on `geo_temporal` (60 unseen basins, later years):

| | h=1 | h=3 | h=7 |
|---|---|---|---|
| NSE (median over basins) | 0.49 | −0.17 | −0.43 |
| High-flow AUPRC | 0.61 | 0.27 | 0.16 |
| 10–90% range coverage (target 0.80) | 0.80 | 0.80 | 0.80 |

LightGBM model: **TBD**. Filled in from the final run, scored once on the test set after all model choices
were made on validation.

## How to run

### 1. Environment
```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data
Either unzip the shared package into `data/caravan/` and check its hash:
```bash
sha256sum basinshift_caravan_v1.zip   # 45a4bba1a2593e82ba43f718b05db365d0f605ece6b2e530f400abd61f27e849
```
or rebuild it from Caravan (about 1.5 GB download, resumable):
```bash
python scripts/caravan/fetch_caravan.py download --gb 1.5
python scripts/caravan/fetch_caravan.py extract --sources camels
python scripts/caravan/prepare_caravan.py --out data/caravan
python scripts/caravan/check_leakage.py --package data/caravan
```

### 3. Validate a model (training data only, no test labels)
```bash
python basinshift_run.py --validate --folds 0,1 --model src.bs_models:BasinshiftModel --name val_lgbm
```
Scores unseen-basin folds and the last two training years: `outputs/val_lgbm/VALIDATION.md`.

### 4. Final run on the test set
```bash
python basinshift_run.py --model src.bs_models:BasinshiftModel --name lgbm_final
```
Trains on all training basins, writes the BASINSHIFT 8.3 submission (`submission.jsonl.gz`), validates it against
the 8.4 rules, scores it, runs 5 stress tests and writes `outputs/lgbm_final/REPORT.md`.
Test labels are read only after predictions are written and validated, and only the evaluation team holds them.

### 5. Tests
```bash
pytest tests/test_basinshift_eval.py tests/test_bs_models.py
```
Run on a slice of the real data: metric identities, submission rule rejections, stress scenarios, and checks that
no feature or prediction uses data from after its forecast date.

## Repository layout
```
basinshift_run.py            one command: data → model → submission → validation → scores → stress → report
src/
  bs_models.py               LightGBM model + features (Lane C), ablation variants
  baselines.py               calibrated persistence reference
  evaluate.py                BASINSHIFT section 10 metrics, basin-bootstrap CIs, reliability tables
  submission.py              section 8.3 JSON writer, 8.4 validator
  stress.py                  5 input-corruption scenarios (7.5)
scripts/caravan/             fetch, prepare, leakage audit, split summary (Lane A)
docs/
  DATA_README.md             columns, units, rules
  DATA_PROVENANCE.md         sources, licenses, citations, transformations
  SPLIT_SUMMARY.md, basin_map.png   are the held-out basins representative?
  BASINSHIFT_EVALUATION.md   metrics, model interface, stress scenarios
  TECHNICAL_REPORT.md        report following BASINSHIFT 12.1, with draft defense answers
tests/                       test_basinshift_eval.py, test_bs_models.py (+ legacy FLOODCAST tests)
data/caravan/                the data package (gitignored)
outputs/                     run outputs (gitignored)
experiments/                 run logs with commit hashes
```
The earlier FLOODCAST pipeline (`solution.py`, `src/config.py`, `src/data.py`, `src/features.py`,
`src/models.py`, `src/validate.py`) is kept unchanged for reference. It is not part of the BASINSHIFT entry.

## Citations
Kratzert et al. (2023) Caravan, *Scientific Data* 10, 61 · Newman et al. (2015) CAMELS, *HESS* 19, 209–223 ·
Linke et al. (2019) HydroATLAS, *Scientific Data* 6, 283 · Muñoz-Sabater et al. (2021) ERA5-Land, *ESSD* 13, 4349–4383.
