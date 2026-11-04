# BASINSHIFT evaluation (Lane D)

How we score every model, and how to plug a new model in. Numbers in reports come only from
`basinshift_run.py` runs, logged in `experiments/basinshift_results.csv` with their commit hash.

## Rebuild the data (Lane A, real data only)
```bash
python scripts/caravan/fetch_caravan.py download --gb 1.5     # head of Caravan v1.6 CSV archive, resumable
python scripts/caravan/fetch_caravan.py extract --sources camels
python scripts/caravan/prepare_caravan.py --out data/caravan  # 299 basins, splits, targets, thresholds
python scripts/caravan/check_leakage.py --package data/caravan  # 15 checks, writes LEAKAGE_REPORT.md
```
Or unzip the shared `basinshift_caravan_v1.zip` into `data/caravan/` and check its sha256.
Column meanings and rules: [DATA_README.md](DATA_README.md).

## Run the evaluation
```bash
python basinshift_run.py                                   # reference persistence baseline
python basinshift_run.py --model src.models:BasinshiftModel --name lgbm_v1   # a Lane C model
```
Outputs in `outputs/<name>/`: `REPORT.md`, `metrics.csv`, `stress.csv`, `reliability_high.csv`,
`predictions.parquet`, `submission.jsonl.gz` (the 8.3 format), `run.json` (commit, timings).

## Model interface
```python
model = fit(train, static, thresholds)
preds = predict(model, test, train, static, thresholds)
```
`preds` is long format, one row per (gauge_id, date, horizon 1/3/7):
`gauge_id, date, horizon, predicted_discharge, lower, upper, p_high, p_extreme`
(mm/day; lower/upper = 10th/90th percentile; probabilities in [0,1]; p_extreme <= p_high).
Either module-level functions (`--model src.baselines`) or a class (`--model module:Class`).
`predict` receives the test inputs only; labels are loaded after predictions are written.

## Test sets
| test_set | Basins | Period | What it measures |
|---|---|---|---|
| `temporal` | 239 train basins | 2008-10 to 2014-09 | Test A: later years |
| `geo` | 60 basins never in training | 1981-10 to 2008-09 | Test B: unseen basins |
| `geo_temporal` | the same 60 | 2008-10 to 2014-09 | unseen basins and unseen years (headline) |

Held-out basins contribute no training rows; their observed flow history is an input at prediction
time (gauged but never trained on, BASINSHIFT 7.4).

## Metrics (BASINSHIFT 10.1), per test set and horizon
| Dimension | Metric | Definition |
|---|---|---|
| Forecast | `nse_median`, `kge_median` | Nash-Sutcliffe / Kling-Gupta efficiency per basin (>= 30 obs), median over basins |
| | `mae_pooled`, `mae_basin_mean` | mean absolute error (mm/day), pooled / averaged over basins |
| Events | `auprc_high`, `auprc_extreme` | area under precision-recall for flow > q_high / q_extreme |
| | `recall_*`, `precision_*`, `false_alarms_per_basin_year_*` | alert when probability >= 0.5 |
| Probabilities | `brier_*`, `brier_skill_*` | Brier score; skill vs. always predicting the base rate (5% / 1%) |
| | `reliability_high.csv` | 10-bin reliability table |
| Intervals | `interval_coverage`, `interval_width`, `pinball_loss` | share of observations inside [lower, upper] (target 0.80), mean width, mean pinball loss at 0.1/0.5/0.9 |
| Robustness | `stress.csv` | relative change of key metrics under 5 input corruptions (7.5) |
| Efficiency | `run.json` | fit / predict / total seconds |

Every metric has a 95% bootstrap CI from resampling basins (200 resamples, seed 42, 10.4), and every
event metric reports its event count. Missing observations are never filled or scored.

## Stress scenarios (fixed seed 42)
`mask_10pct`, `mask_30pct` (input values removed at random), `gaps_3to7d` (contiguous gaps on ~10% of days),
`noise_weather` (Gaussian noise, 10% of each basin's sd), `obs_delay_1d` (latest flow observation 1 day late).

## Submission checks (BASINSHIFT 8.4), `src/submission.py`
Rejects: missing columns, non-string ids, unknown ids, unexpected horizons, NaN/inf, probabilities
outside [0,1], negative flow or bounds, lower > upper, p_extreme > p_high, duplicate keys, missing
or extra (gauge_id, date, horizon) rows. The written JSON file is read back and re-validated.
`tests/test_basinshift_eval.py` proves each rejection on real data.
