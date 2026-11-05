# BasinPulse: river-flow forecasting that transfers to unseen basins

**Team:** BasinPulse | **Challenge:** BASINSHIFT, the Global Flood Intelligence Challenge

**Live site:** https://basinpulse.onrender.com (results, the 60 unseen rivers, and the submitted model running live on
your CSV; free hosting, so the first visit after a quiet spell takes about a minute to wake up)

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
- **Forecast and range:** LightGBM quantile regression (10th / 50th / 90th percentile) and a mean regressor (Duan
  smearing) per horizon; the range is widened by conformal calibration on basins held out from the trees, so it
  covers ~80% on unseen basins.
- **Flood probabilities:** LightGBM classifiers for flow above each basin's 95th / 99th percentile,
  isotonic-calibrated on the same held-out basins.
- **Robust by training:** 15% of inputs hidden at random, and 20% of examples seen with a one-day-old flow reading
  plus its age, so gaps and late data do not break the forecast.
- **Inputs:** past observed flow, rain (lags, sums, antecedent index), temperature, snow and soil-water dynamics,
  evaporation, season, catchment attributes, age of the latest reading. Every input uses data up to day *t* only.
- **Reference:** calibrated persistence (tomorrow's flow = today's) and seasonal climatology.

## Results
All numbers come from logged runs (`experiments/`, `docs/results/`) with their commit hash.

**Headline: `geo_temporal`, 60 basins never seen in training, years after the training period.**
Final configuration (`src.tuning:T0` = `DelayAwareModel`, 600k-row sample, 500 trees, trained on the 239 benchmark
training basins + 149 extra basins) vs persistence, scored once on the test set after every choice was made on
validation ([experiments](docs/EXPERIMENTS.md)). 95% confidence intervals from resampling basins.

| | 1 day | 3 days | 7 days |
|---|---|---|---|
| NSE, median over basins | 0.48 → **0.63** [0.57, 0.76] | −0.17 → **0.30** [0.20, 0.36] | −0.43 → **0.15** [0.11, 0.21] |
| High-flow AUPRC | 0.61 → **0.82** | 0.27 → **0.45** | 0.16 → **0.29** |
| Extreme-flow AUPRC | 0.19 → **0.50** | 0.06 → **0.15** | 0.03 → **0.10** |
| Brier skill (high flow) | 0.51 → **0.65** | 0.22 → **0.32** | 0.13 → **0.20** |
| Alert precision at p ≥ 0.5 | 0.67 → **0.85** | n/a → **0.76** | n/a → **0.67** |
| False alarms per basin-year | 5.6 → **2.0** | 0 → 1.3 | 0 → 1.1 |
| 10–90% range coverage (target 0.80) | 0.80 → **0.82** | 0.80 → **0.81** | 0.80 → **0.79** |
| 10–90% range width (mm/day) | 0.67 → 0.88 | 1.78 → **1.60** | 2.87 → **2.02** |
| MAE (mm/day) | 0.50 → **0.36** | 0.90 → **0.74** | 1.14 → **0.91** |

Same picture on the other test sets (`temporal`: NSE 0.68 / 0.27 / 0.15; `geo`: 0.67 / 0.27 / 0.19).
Full tables: [docs/results/FINAL_T0_TEST_REPORT.md](docs/results/FINAL_T0_TEST_REPORT.md); figures in
`docs/figures/`; failure analysis in [docs/FAILURE_ANALYSIS.md](docs/FAILURE_ANALYSIS.md).
The **submitted model** (`models/basinpulse_final.joblib`) is the same configuration refitted on all 671
CAMELS-US basins and all years 1981–2014.

**Judge-style test.** `scripts/official_eval.py` runs the test the way the BASINSHIFT evaluator would: inputs only → `predict.py` (1.25M forecasts, 299 rivers, 12 min on a laptop CPU, 1.6 GB) → 8.4 validation → every section 10 metric, robustness under 6 corruption scenarios, and the 10.2 weighted score with a pre-declared normalization (53.9 / 100 vs 23.7 for persistence; 0 = persistence, 100 = perfect). BasinPulse beats persistence on 83% / 88% / 92% of unseen rivers at 1 / 3 / 7 days. [docs/OFFICIAL_STYLE_EVALUATION.md](docs/OFFICIAL_STYLE_EVALUATION.md)

**Zero-shot on another continent.** Trained on US basins only and applied without retraining to **579 Central
European basins** (LaMAH-CE: Austria, Germany, Switzerland), 2008–2014: NSE **0.74 / 0.36** at 1 / 3 days
(persistence 0.68 / 0.26, climatology 0.06 / 0.05), high-flow AUPRC 0.79 vs 0.64, extreme-flow AUPRC 0.39 vs 0.17.
[docs/TRANSFER_EUROPE.md](docs/TRANSFER_EUROPE.md)

**What we learned** (validation, [details](docs/EXPERIMENTS.md)):
- Predicting the *change* from the river's own current flow transfers well: unseen-basin skill is close to
  seen-basin skill, and holds on another continent.
- **More rivers + more capacity** helps unseen basins most: adding 149 training basins and doubling the sample
  raised every unseen-basin metric (extreme-flow AUPRC at 7 days +28%).
- **H1:** catchment attributes barely move the median forecast but matter for **flood detection at 3–7 days**
  (shuffling them costs 0.12 high-flow AUPRC; [feature importance](docs/FEATURE_IMPORTANCE.md)).
- **H4:** training with 15% of inputs hidden makes the model much more robust to missing data
  (1-day NSE with 30% of inputs missing: 0.52 → 0.59) at almost no clean cost.
- **Late data:** with a data-age input and delay training, the 1-day range stays calibrated when the latest flow
  reading is a day late (coverage 0.57 → 0.81).
- **Ungauged basins:** with no flow data at all, the model still beats persistence at 3 and 7 days.

**ShiftGuard** (built in, opt-in): every basin gets a **novelty score** (how unlike the training basins its
catchment is); ranges can be calibrated per novelty level; and every forecast comes with its **drivers**
("recent rain", "snowmelt", "soil wetness", ...) from exact tree contributions. `predict.py --explain` writes both.

## Run on new (hidden) data
```bash
python predict.py --inputs hidden_test.parquet --out submission.jsonl.gz
python predict.py --inputs test.csv --history history.csv --static static.csv --thresholds thresholds.csv \
                  --explain drivers.parquet --out submission.jsonl.gz
```
Loads `models/basinpulse_final.joblib` (trained on all 671 CAMELS-US basins, 1981–2014, by `train_final.py`).
Accepts parquet or CSV, common column names (`gauge_id`/`catchment_id`/`station_id`, `date`/`forecast_origin`),
missing weather columns and missing history / attribute / threshold files; the output is validated against the
BASINSHIFT 8.4 rules before it is written.

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
python basinshift_run.py --validate --folds 0,1 --model src.tuning:T0 --name val_T0
```
Scores unseen-basin folds and the last two training years: `outputs/val_T0/VALIDATION.md`.

### 4. Final run on the test set
```bash
python basinshift_run.py --model src.tuning:T0 --extra-train data/caravan_extra --name final_T0
```
Trains on all training basins, writes the BASINSHIFT 8.3 submission (`submission.jsonl.gz`), validates it against
the 8.4 rules, scores it, runs 5 stress tests and writes `outputs/final_T0/REPORT.md`.
Test labels are read only after predictions are written and validated, and only the evaluation team holds them.

### 5. Tests
```bash
pytest
```
Runs on a slice of the real data: metric identities, submission-rule rejections, the JSON Schema, stress
scenarios, and checks that no feature or prediction uses data from after its forecast date.
The tests need the data package in `data/caravan/` (step 2); on a clone without it they are skipped, not failed.

### 6. Docker (optional, isolated environment)
```bash
docker build -t basinpulse .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/outputs:/app/outputs" basinpulse
```

### 7. Website
```bash
python web/server.py          # http://localhost:8000
python web/build_data.py      # only to rebuild web/static/data from outputs/final_T0 (already committed)
```
A scroll-driven story, the headline numbers, a map of the 60 unseen rivers with each river's forecast around its
largest flood, ShiftGuard drivers, and a live "try the model" upload that runs the submitted model on a CSV.

### 8. Public website (Render, free)
`render.yaml` deploys the site from this repository: on render.com, **New → Blueprint**, pick the repository, and
Render builds `deploy/Dockerfile.web` and serves it at https://basinpulse.onrender.com,
redeploying on every push to `main`. The image holds only what the site needs (web/, predict.py, src/, the submitted
model; no training or test data). On the free 512 MB plan live uploads are limited to 20,000 rows, one at a time
(the server peaks below 300 MB). Free services sleep after 15 minutes idle and wake in about a minute.
`deploy/make_space.py` builds the same site as a Hugging Face Docker Space (needs a Hugging Face PRO account).

### 9. Offline demo
Open `docs/demo/index.html` in any browser (offline, single file): a map of the 60 unseen rivers; click one to see
its forecasts, range and flood probability around its largest flood, against persistence.

## Repository layout
```
basinshift_run.py            one command: data → model → submission → validation → scores → stress → report
src/
  bs_features.py             feature engineering (Lane B): 76 leakage-safe features
  bs_models.py               LightGBM model (Lane C); final model class; ablation variants
  baselines.py               persistence and seasonal-climatology references
  evaluate.py                BASINSHIFT section 10 metrics, alert budgets, basin-bootstrap CIs, reliability
  submission.py              section 8.3 JSON Lines writer, 8.4 validator
  stress.py                  input-corruption scenarios (7.5)
scripts/
  caravan/                   fetch, prepare, leakage audit, split summary, extra training basins, transfer package
  summarize_validation.py    docs/EXPERIMENTS.md from all validation runs
  failure_analysis.py        per-basin skill vs catchment characteristics
  warning_and_regions.py     early-warning lead time, skill by region
  feature_importance.py      which input groups matter for unseen basins
  transfer_test.py           zero-shot test on Central European basins
  make_figures.py, build_demo.py   figures and the demo page
docs/
  TECHNICAL_REPORT.md        report following BASINSHIFT 12.1, with defense answers
  MODEL_CARD.md, TEAM_DECLARATION.md, DATA_PROVENANCE.md, DATA_README.md
  EXPERIMENTS.md, FAILURE_ANALYSIS.md, WARNING_AND_REGIONS.md, FEATURE_IMPORTANCE.md, TRANSFER_EUROPE.md
  PRESENTATION.md            slide outline, talking points, defense one-liners
  submission.schema.json     machine-readable output contract (8.3)
  figures/, demo/, results/  figures, offline demo, metric tables of every reported run
web/                         website: FastAPI server (server.py), static site, data builder
tests/                       evaluation, model, schema tests (real data)
legacy/                      earlier FLOODCAST pipeline, not part of this entry (see legacy/README.md)
Dockerfile                   reproducible environment; requirements.txt pins every version
data/, outputs/              data packages and run outputs (gitignored)
experiments/                 run logs with commit hashes
```

## Citations
Kratzert et al. (2023) Caravan, *Scientific Data* 10, 61 · Newman et al. (2015) CAMELS, *HESS* 19, 209–223 ·
Linke et al. (2019) HydroATLAS, *Scientific Data* 6, 283 · Muñoz-Sabater et al. (2021) ERA5-Land, *ESSD* 13, 4349–4383.
