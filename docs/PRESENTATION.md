# BasinPulse: presentation outline (8 slides, ~6 minutes + defense)

Each slide lists what to show, what to say, and which judging category it earns points in (BASINSHIFT 13).
All numbers are from the final run (`outputs/lgbm_final/REPORT.md`) or validation (`docs/EXPERIMENTS.md`).

---

### 1. The problem (20 s)
**Show:** `docs/figures/hydrographs.png`, top panel: a flood on a basin the model never saw.
**Say:** "Flood forecasts usually work where the model was trained. BASINSHIFT asks: do they still work on rivers
they've never seen? We built a forecaster and tested it exactly there: 60 rivers it never saw, in years after
its training data ends."

### 2. Data: real, and split so we can't cheat (Real-world applicability, Reproducibility)
**Show:** `docs/basin_map.png`.
**Say:** 299 real US river basins from Caravan: observed USGS flow, ERA5-Land weather averaged over each basin.
60 basins held out completely, spread over all 14 regions and statistically similar to the training basins
(KS ≤ 0.17 on every characteristic). Test years 2008–2014 come after all training years. 15 automated leakage
checks pass. The test answers never left the evaluator, and every model choice was made on validation.

### 3. Method (Technical architecture)
**Show:** a simple diagram: inputs (flow, rain, snow, soil, basin traits; dates ≤ t only) → per horizon:
quantile models + mean model + 2 calibrated classifiers → forecast, 10–90% range, P(high), P(extreme).
**Say:** We predict the *change* from today's flow, in log space, so one model fits rivers whose flows differ a
thousand-fold. Quantile LightGBM gives the range, conformal calibration on held-out basins keeps it honest,
isotonic calibration makes the probabilities mean what they say, and training with 15% of inputs hidden makes
it robust to missing data. Trains in 5 minutes on a laptop CPU.

### 4. Headline result (ML performance and generalization, 20 pts)
**Show:** `docs/figures/skill_by_horizon.png`.
**Say:** On unseen rivers in unseen years, against persistence (tomorrow = today):
- NSE 0.63 / 0.30 / 0.15 at 1 / 3 / 7 days, vs 0.49 / −0.17 / −0.43. At 3 and 7 days persistence is worse
  than guessing the average; we stay positive, and the confidence intervals don't overlap.
- High-flow AUPRC 0.81 / 0.45 / 0.28 vs 0.61 / 0.27 / 0.16. Extreme floods: 0.49 vs 0.19 at 1 day, 2.6×.
- Skill on unseen rivers matches skill on the training rivers (NSE 0.68 / 0.26 / 0.15): the model transfers.
- Beats persistence on 52 of 60 unseen rivers at 3 days.

### 5. Uncertainty you can trust (Uncertainty and responsible AI, 10 pts)
**Show:** `docs/figures/reliability.png`.
**Say:** The 10–90% range covers 81 / 80 / 79% of real values (target 80%) on unseen rivers, and at 7 days
it's 31% narrower than persistence's range. The probability curve follows the diagonal: when we say 30%,
it happens about 30% of the time. Flood alerts: precision 0.85, and false alarms drop from 5.6 to 2.0 per
river per year.

### 6. Robustness (Robustness and reliability, 15 pts)
**Show:** `docs/figures/stress.png`.
**Say:** Five stress tests on the test set. With 30% of inputs missing, 1-day NSE drops 9% (persistence: 33%).
With the latest flow reading a day late, −26% (persistence: −94%). At 3 and 7 days nothing costs more than 13%.
H4 holds: training with hidden inputs raised stressed NSE by 0.025 on validation for a 0.01 clean cost.
Honest weak spot: with late flow data, the 1-day range is too confident (60% coverage), so a deployed
system should pass the data age as an input.

### 7. What we learned (Research novelty, 15 pts)
**Say:**
- Predicting *change relative to the river's own current state* is what makes transfer work.
- H1: catchment attributes help a little and consistently, but within the confidence intervals: most of the
  transferable signal is in the flow and weather dynamics.
- Ungauged rivers (no flow data at all): NSE 0.33 / 0.13 / 0.07, still better than persistence beyond 1 day,
  so weather + basin traits alone carry real multi-day skill.
- Hard cases: flat basins with frequent intense rain (south-central US); easy: snowmelt and mountain rivers.

### 8. Limits and responsible use (Real-world applicability, Responsible AI)
**Say:** US only; fewer snowmelt rivers in the held-out set; reanalysis weather; sharp peaks are
under-estimated in height (the probability still rises at the onset); thresholds are statistical, not official
flood stages. A research benchmark, not a warning system. Deployment would need operational weather feeds,
local validation, expert review and agency coordination.

---

## Defense questions (12.2): one-line answers
1. *Generalization evidence?* 60 never-seen basins across 14 regions: NSE 0.63 / 0.30 / 0.15 vs 0.68 / 0.26 / 0.15
   on seen basins (inside each other's CIs); beats persistence on 52 of 60 at 3 days.
2. *Inputs unavailable in deployment?* ERA5-Land arrives a few days late; flow data can be late (tested: −26% NSE
   at 1 day, 60% range coverage).
3. *Thresholds?* Each basin's 95th / 99th percentile flow over 1981–2008, fixed before any modelling.
4. *Missing discharge history?* Last observation up to 30 days, missing-day counts as a feature, robustness
   training; 30% missing inputs cost 9% NSE. No flow at all: NSE 0.33 / 0.13 / 0.07 (validation).
5. *Biggest ablation effect?* Removing observed flow (1-day NSE 0.61 → 0.33); then mean vs median forecast
   (+0.06 NSE); attributes (H1) small; robustness training (H4) +0.025 stressed NSE.
6. *Uncertainty calibration?* Conformal widening (ranges) and isotonic regression (probabilities), both fitted
   on basins held out from the trees; test coverage 0.81 / 0.80 / 0.79, Brier skill 0.65 / 0.31 / 0.20.
7. *Failure modes?* Flat, low-lying basins with frequent intense rain (5 of the 8 worst in the south-central
   US); sharp peaks under-estimated in height; over-confident range when flow data is late.
8. *Before deployment?* Operational weather forecasts, data-age input, local validation, monitoring, fail-safes,
   agency coordination.

## Demo (if asked)
`python basinshift_run.py --model src.bs_models:RobustMeanModel --name demo` runs data → model → validated
submission → scores → stress tests → report. Then open `outputs/demo/REPORT.md` and `docs/figures/`.
