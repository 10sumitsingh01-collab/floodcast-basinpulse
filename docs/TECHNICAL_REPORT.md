# BasinPulse: technical report (BASINSHIFT 12.1)

Every number comes from a logged `basinshift_run.py` run (`outputs/<run>/`, commit hashes in
`experiments/basinshift_results.csv` and `experiments/basinshift_validation.csv`). All model choices were made
on validation splits inside the training data ([EXPERIMENTS.md](EXPERIMENTS.md)); the test set was scored once.

## 1. Abstract and executive summary
We forecast daily river flow 1, 3 and 7 days ahead, with a 10–90% range and the probability of high and
extreme flow, and test the forecasts on river basins never used in training. Data: 299 real CAMELS-US basins
from Caravan v1.6 (observed USGS flow, ERA5-Land weather, HydroATLAS attributes); 60 basins are held out
entirely, and test years (2008–2014) follow all training years. Method: per horizon, LightGBM quantile models
predict the log change from the latest observed flow, with conformal calibration of the range and isotonic
calibration of event probabilities, trained with 15% of inputs randomly hidden for robustness. On the 60 unseen
basins in unseen years, median NSE is 0.63 / 0.30 / 0.15 at 1 / 3 / 7 days vs. 0.49 / −0.17 / −0.43 for
persistence; high-flow AUPRC rises from 0.61 / 0.27 / 0.16 to 0.81 / 0.45 / 0.28; the range covers 81 / 80 /
79% of observations (target 80%) while being 31% narrower than persistence's at 7 days; false alarms fall from
5.6 to 2.0 per basin-year at the 1-day horizon. Skill on unseen basins is close to skill on seen basins.
Main limitations: CAMELS-US only, held-out basins are gauged at prediction time (an ungauged variant is weaker
but still beats persistence beyond 1 day), and reanalysis rather than operational weather inputs.

## 2. Problem formulation and research hypothesis
At the end of day t, for basin c, predict flow on t+1, t+3 and t+7 (mm/day), an 80% interval, and
P(flow > q_high(c)) and P(flow > q_extreme(c)), where q_high/q_extreme are the basin's 95th/99th
percentile flow over 1981-2008. Main hypothesis (H1): static catchment attributes improve transfer to
basins never seen in training. Secondary (H4): training with missing inputs improves robustness.

## 3. Dataset and feature provenance
**Source.** Caravan v1.6 (CSV, CC-BY-4.0), CAMELS-US subset: observed USGS daily streamflow (mm/day,
already normalised by catchment area), 10 ERA5-Land basin-averaged daily weather variables, and HydroATLAS
catchment attributes. Full source register and citations: `docs/DATA_PROVENANCE.md`.

**Selection.** 671 basins downloaded; 611 passed QA (observed flow covering >= 90% of 1981-2008 and
>= 50% of 2008-2014, weather <= 5% missing); 60 excluded with the reason recorded per basin. 299 basins
were sampled in proportion to the 14 USGS regions; 60 of them are held out entirely.

**Benchmark size.** 2,357,018 training origins (239 basins, median 27.0 years of observed flow each);
1,246,829 test origins: 523,649 `temporal`, 591,720 `geo`, 131,460 `geo_temporal`. At the 1-day horizon
the test sets contain 25,573 / 29,271 / 5,976 high-flow days and 5,329 / 5,874 / 1,243 extreme days.

**Representativeness (7.3).** Held-out and training basins have similar distributions of area, aridity,
precipitation, elevation, slope, forest cover and regulation (Kolmogorov-Smirnov distance <= 0.17 for all
ten characteristics; `docs/SPLIT_SUMMARY.md`, map in `docs/basin_map.png`). Every region contributes
held-out basins. One gap: the held-out set has fewer strongly snow-dominated basins (90th percentile snow
fraction 0.26 vs. 0.51), so transfer to snowmelt regimes is tested less than other regimes.

**Inputs at prediction time.** Weather up to day t, static attributes, and the basin's observed flow up to
day t (`streamflow_obs`). Held-out basins have no training rows but their past flow is available at
inference (gauged but never trained on, BASINSHIFT 7.4); a variant without any flow input covers the
ungauged case.

Features: TBD (Lane B, `src/bs_features.py`; the current model builds its own, see section 5).

## 4. Preprocessing and leakage controls
| Risk (7.2) | Control | Evidence |
|---|---|---|
| Basin identity leakage | Basins assigned to train / held out before any window or feature is built | check: held-out basins never in train (0 overlap) |
| Temporal leakage | Test origins start 2008-10-01; training targets (t+h) never pass 2008-09-30 | check: latest training target date 2008-09-30 |
| Validation leakage | Temporal validation drops fit rows whose t+h falls after the fit end | `basinshift_run.py --validate` |
| Future information in inputs | `streamflow_obs` is same-day flow; features use dates <= t only | check: obs equals the previous origin's h=1 label on 1,236,888 consecutive pairs |
| Target-derived statistics | Thresholds and climate indices from 1981-2008 only; validation thresholds from the fit period only | checks: recomputation matches to 1e-5 |
| Modelled-discharge attributes | HydroATLAS `dis_*`, `run_mm_syr`, `riv_tc_usu`, `ria_ha_usu` removed | check: none present |
| Label access | `test_labels.parquet` not shipped to modelling lanes; loaded only after predictions are written and validated | package contents, `basinshift_run.py` |
| Missing data | Missing flow never filled or scored; negatives set missing | `PREP_REPORT.md` |

All 15 checks are automated in `scripts/caravan/check_leakage.py` and pass (`LEAKAGE_REPORT.md`).

## 5. Model architecture and training procedure
**Features** (`src/bs_models.py: build_features`, 76 per origin, all from dates <= t): latest observed log-flow
(forward-filled at most 30 days) and its lags (1/2/3/7 d), rolling means (3/7/30 d) and changes; days with
missing flow; precipitation (today, lags 1–3, sums over 3/7/14/30/60 d, 7-day max, antecedent precipitation index
with k = 0.9); temperature (today, max, 3/7/30-day means); snow water equivalent and its 1/3/7-day change
(snowmelt); soil water layers 1 and 3, 7-day mean and 3-day change; evaporation sums (7/30 d) and the 30-day
water balance P − PET; radiation, dewpoint; day-of-year sin/cos; 28 static attributes plus log area; the current
flow relative to the basin's mean rainfall and its 95th / 99th percentile thresholds.

**Target.** r_h = log1p(flow[t+h]) − log1p(flow_obs[t]). The model predicts how much the river will change, in
relative terms, which is comparable across basins whose flows differ by orders of magnitude.

**Models per horizon h ∈ {1, 3, 7}** (LightGBM 4.7, 300 trees, 63 leaves, learning rate 0.06, row and column
subsampling 0.8, seed 42):
1. Quantile regressors at 0.1, 0.5 and 0.9; quantiles sorted per row.
2. An L2 regressor for the point forecast, back-transformed with Duan's smearing factor (mean forecast).
3. Conformal widening (CQR): the 10–90% range is widened by the 80% quantile of conformity scores on
   calibration basins (the fitted widening was ≈ 0, i.e. the raw quantiles were already calibrated).
4. Two binary classifiers (flow > q_high, flow > q_extreme), isotonic-calibrated on calibration basins;
   p_extreme ≤ p_high enforced.

**Training.** 20% of the training basins (seeded) are kept out of the trees as calibration basins, so the range
and probabilities are calibrated on basins the trees never saw, mimicking the unseen-basin test. 300,000 origins
are sampled from the remaining basins (laptop memory). **Robustness training (H4):** 15% of input values are
hidden at random before features are built, so the model learns to cope with gaps. Final training on all 239
training basins: ~5 minutes on an 8-thread, 7 GB laptop.

## 6. Baseline comparison and ablation study
All on validation (`val_unseen_basins`: 96 training basins held out in two folds), from [EXPERIMENTS.md](EXPERIMENTS.md):

| Model | NSE 1d | NSE 3d | NSE 7d | AUPRC high 3d | Coverage 3d |
|---|---|---|---|---|---|
| Persistence | 0.473 | −0.219 | −0.461 | 0.282 | 0.785 |
| LightGBM, median forecast (v1) | 0.609 | 0.223 | 0.118 | 0.481 | 0.795 |
| + mean forecast (v2) | 0.670 | 0.278 | 0.192 | 0.481 | 0.795 |
| v1 without catchment attributes (H1) | 0.595 | 0.224 | 0.115 | 0.467 | 0.789 |
| v1 with robustness training (H4) | 0.598 | 0.214 | 0.114 | 0.470 | 0.805 |
| v1 without any flow input (ungauged) | 0.327 | 0.128 | 0.074 | 0.354 | 0.808 |
| **Final: H4 + mean** | **0.666** | **0.280** | **0.193** | 0.470 | 0.805 |

- **Biggest effect:** using observed flow at all (ungauged → gauged: 1-day NSE 0.33 → 0.61), then the mean
  forecast (+0.06 NSE at 1 day, +0.15 KGE at 7 days).
- **H1:** catchment attributes give small, consistent gains in most metrics, inside the confidence intervals.
  Part of the basin's scale information also reaches the model through its flow thresholds.
- **H4:** costs ~0.01 NSE on clean inputs, but raises mean stressed 1-day NSE by 0.025 (30% missing inputs:
  0.518 → 0.588). Adopted by a rule fixed before looking (≥ 0.01 stressed gain, < 0.02 clean loss).

## 7. Temporal and unseen-basin results
Test set, final model, median NSE over basins [95% basin-bootstrap CI]:

| Test set | Basins | 1 day | 3 days | 7 days |
|---|---|---|---|---|
| `temporal` (seen basins, later years) | 239 | 0.678 [0.649, 0.727] | 0.261 [0.224, 0.310] | 0.150 [0.126, 0.180] |
| `geo` (unseen basins) | 60 | 0.659 [0.547, 0.739] | 0.276 [0.204, 0.357] | 0.171 [0.156, 0.243] |
| `geo_temporal` (unseen basins, later years) | 60 | 0.634 [0.577, 0.753] | 0.295 [0.195, 0.371] | 0.149 [0.116, 0.205] |
| persistence, `geo_temporal` | 60 | 0.485 | −0.171 | −0.429 |

Unseen-basin skill is within the confidence intervals of seen-basin skill at every horizon: the model transfers.
KGE is 0.70 / 0.33 / 0.15 on `geo_temporal`, below persistence (0.74 / 0.42 / 0.28): persistence reproduces the
observed variability exactly by construction, while any forecast that hedges toward the expected value is smoother.
MAE falls from 0.50 / 0.90 / 1.14 to 0.37 / 0.74 / 0.90 mm/day.

## 8. Extreme-event and calibration results
`geo_temporal` (5,976 high-flow and 1,243 extreme days at 1 day ahead):

| | 1 day | 3 days | 7 days |
|---|---|---|---|
| AUPRC high (persistence) | 0.813 (0.609) | 0.446 (0.274) | 0.279 (0.160) |
| AUPRC extreme (persistence) | 0.493 (0.192) | 0.140 (0.063) | 0.068 (0.034) |
| Brier skill, high | 0.645 | 0.311 | 0.198 |
| Recall / precision at p ≥ 0.5 | 0.65 / 0.85 | 0.23 / 0.76 | 0.11 / 0.71 |
| False alarms per basin-year | 2.0 (5.6) | 1.2 | 0.7 |
| Range coverage (target 0.80) | 0.811 | 0.796 | 0.788 |
| Range width, mm/day (persistence) | 0.89 (0.67) | 1.59 (1.78) | 1.99 (2.87) |

Probabilities are calibrated (reliability diagram: `docs/figures/reliability.png`). At 3 and 7 days, alerts at a
fixed 0.5 threshold are precise but miss most events: the probabilities are honest about how uncertain multi-day
peaks are. Users who prefer fewer missed floods would alert at a lower probability (higher recall, more false alarms).

## 9. Robustness and efficiency analysis
Test set, `geo_temporal`, same fitted model with corrupted inputs (fixed seed 42). Relative change of median NSE
vs clean inputs, persistence → final model (final model's stressed NSE in brackets):

| Scenario | 1 day | 3 days | 7 days |
|---|---|---|---|
| 10% of inputs removed | −6% → **−3%** (0.61) | −5% → **−1%** (0.29) | −2% → −1% (0.15) |
| 30% of inputs removed | −33% → **−9%** (0.58) | +7% → −13% (0.26) | +4% → −6% (0.14) |
| 3–7 day gaps on ~10% of days | −14% → **−9%** (0.58) | +11% → −7% (0.28) | −3% → +0% (0.15) |
| Weather noise (10% of sd) | 0% → −2% (0.62) | 0% → −1% (0.29) | 0% → +1% (0.15) |
| Latest flow observation 1 day late | −94% → **−26%** (0.47) | −32% → **−11%** (0.26) | −12% → **−3%** (0.15) |

Even under stress the final model stays well above clean persistence at 3 and 7 days. (Persistence "improves"
in a few 3–7 day cells only because its clean NSE is negative.) Ranges stay calibrated under missing data and
gaps (coverage 0.79–0.81), **except when the latest flow is a day late: 1-day coverage falls to 0.60**, because the
model does not know the observation is stale. A deployed version should pass the data age as an input.
Robustness training (H4) is what keeps these losses small (section 6). Figure: `figures/stress.png`.

**Efficiency** (8-thread, 7 GB RAM Windows laptop, CPU only, `outputs/lgbm_final/run.json`): features for 2.36 M
origins in ~65 s; training 291 s (per horizon: 3 quantile + 1 mean regressor and 2 classifiers, 240–300 trees
each, plus calibration); prediction of 1.25 M origins × 3 horizons in 243 s. CPU only; no GPU needed.

**Code version.** The final run reports commit `72500f8-dirty`; the "dirty" part was documentation only
(`git diff 72500f8 -- src/ basinshift_run.py` is empty).

**Hydrographs** (`figures/hydrographs.png`): on the sharpest peaks the 1-day point forecast underestimates the
height, but the high-flow probability rises at the onset of each event, which is what an alert needs.

## 10. Limitations, failure cases and responsible use
**Failure cases** ([FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md), per-basin NSE on `geo_temporal`): the model beats
persistence on 52 of 60 unseen basins at 3 days; no unseen basin has negative NSE at 1 day (2 at 3 days, 6 at
7 days). Skill is lowest in flat, low-elevation basins with frequent intense rain (3-day NSE correlates with
high-precipitation frequency −0.39, slope +0.36, elevation +0.32, snow fraction +0.38). Five of the eight worst
basins are in the south-central US (USGS parts 07–08), where short, intense storms drive flow and daily data
cannot resolve their timing.
Snowmelt and mountain basins are the easiest. The single worst basin at 7 days (camels_09306242: 87 km², steep,
snow fraction 0.36, NSE −4.2) is an outlier worth inspecting; one such basin moves the mean but not the median.

**Limitations**
- CAMELS-US only: one country, mostly natural catchments; global transfer is not shown.
- Held-out basins are gauged at inference (their past flow is an input). Without any flow input (ungauged
  variant, validation) skill drops to NSE 0.33 / 0.13 / 0.07, still above persistence beyond 1 day.
- KGE is below persistence: forecasts that hedge toward the expected value are smoother than the observed flow.
- Fewer snow-dominated basins in the held-out set than in training (see `SPLIT_SUMMARY.md`).
- Thresholds are statistical labels (95th/99th percentile), not official flood stages (16.2).
- Discharge is basin-level; this says nothing about inundation depth or exposure.
- ERA5-Land is reanalysis; a real-time system would use operational forecasts with different errors.
- Research benchmark only: not for evacuation, emergency response or public warnings (16.1).

## 11. Reproduction instructions and artifact references
See `docs/BASINSHIFT_EVALUATION.md`: rebuild data, check hashes, run `python basinshift_run.py --model ...`.
Artifacts: data package sha256 (in `basinshift_caravan_v1.sha256.txt`), commit hash per run.

---

## Draft answers to the defense questions (12.2)
1. **Evidence of generalization to unseen basins?** 60 basins never in training, stratified across 14
   regions; reported separately as `geo` and `geo_temporal`, with basin-bootstrap CIs. On `geo_temporal`,
   median NSE 0.63 / 0.30 / 0.15 vs 0.68 / 0.26 / 0.15 on seen basins (`temporal`): inside each other's CIs.
   The model beats persistence on 52 of 60 unseen basins at 3 days.
2. **Inputs unavailable in deployment?** ERA5-Land is reanalysis, available with ~5 days delay; same-day
   observed flow may also arrive late (tested by `obs_delay_1d`).
3. **How are thresholds defined and frozen?** 95th/99th percentile of each basin's observed flow,
   1981-10-01 to 2008-09-30, computed once in `prepare_caravan.py` before any modelling.
4. **Behaviour when discharge history is missing?** Stress tests `mask_*`, `gaps_3to7d`, `obs_delay_1d`;
   the model uses the last observation within 30 days, flags missing days as a feature, and was trained with
   15% of inputs hidden (H4). Test-set degradation per scenario: section 9. With no flow input at all
   (ungauged variant), validation NSE is 0.33 / 0.13 / 0.07: weaker, but still above persistence beyond 1 day.
5. **Which ablation changed the result most?** Removing observed flow (1-day NSE 0.61 → 0.33), then the mean
   vs median forecast (+0.06 NSE at 1 day, +0.15 KGE at 7 days). Catchment attributes (H1): small consistent
   gains inside the CIs. Robustness training (H4): +0.025 stressed NSE for −0.01 clean NSE.
6. **How is uncertainty calibrated?** Ranges: quantile regression plus conformal widening fitted on basins the
   trees never saw; test coverage 0.81 / 0.80 / 0.79 (target 0.80). Probabilities: isotonic calibration on the same
   basins; Brier skill 0.65 / 0.31 / 0.20 and the reliability diagram follows the diagonal.
7. **Largest failure modes?** Flat, low-elevation basins with frequent intense rain, mostly south-central US
   (short storms, daily resolution); one small steep basin at 7 days (NSE −4.2); smoothed peaks (KGE below persistence).
   Details in `FAILURE_ANALYSIS.md`.
8. **What before real deployment?** Local validation, operational weather feeds, expert review, monitoring,
   fail-safes, coordination with agencies (16.1).
