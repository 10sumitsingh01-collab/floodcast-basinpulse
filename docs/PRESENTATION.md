# BasinPulse: presentation outline (8 slides, ~6 minutes + defense)

Each slide lists what to show, what to say, and which judging category it earns points in (BASINSHIFT 13).
Numbers marked [R] come from `outputs/lgbm_final/REPORT.md`; fill them from the final run only.

---

### 1. The problem (10 s)
**Show:** one hydrograph of a flood on a basin the model never saw (`docs/figures/hydrographs.png`, top panel).
**Say:** "Flood forecasts work where models were trained. BASINSHIFT asks: do they still work on rivers they've
never seen? We built a forecaster and tested it exactly there."

### 2. Data: real, and split so we can't cheat (Real-world applicability, Reproducibility)
**Show:** `docs/basin_map.png`.
**Say:** 299 real CAMELS-US basins from Caravan (observed USGS flow, ERA5-Land weather). 60 held out
completely, in all 14 regions, statistically similar to training basins (KS ≤ 0.17). Test years 2008–2014
come after all training years. 15 automated leakage checks, all pass. Test answers never left the evaluator.

### 3. Method (Technical architecture)
**Show:** a simple diagram: inputs → features (dates ≤ t only) → per horizon: 3 quantile models + 2 calibrated
classifiers → forecast, range, P(high), P(extreme).
**Say:** We predict the *change* from today's flow in log space, so one model fits rivers of every size.
Quantile LightGBM gives the range; conformal calibration on held-out basins makes it honest; isotonic
calibration makes probabilities mean what they say.

### 4. Headline result (ML performance and generalization, 20 pts)
**Show:** `docs/figures/skill_by_horizon.png`.
**Say:** On unseen basins in unseen years vs persistence: NSE [R], high-flow AUPRC [R], extreme AUPRC [R],
each with 95% CIs over basins. Biggest gain at 3 and 7 days, where persistence collapses.

### 5. Uncertainty you can trust (Uncertainty and responsible AI, 10 pts)
**Show:** `docs/figures/reliability.png` + coverage numbers.
**Say:** The 10–90% range covers [R] of real values (target 80%) on unseen basins, while being [R]% narrower
than the persistence range at 7 days. When we say 30%, it floods about 30% of the time.

### 6. Robustness (Robustness and reliability, 15 pts)
**Show:** `docs/figures/stress.png`.
**Say:** Five stress tests: 10% / 30% missing inputs, 3–7 day gaps, noisy weather, late flow data. [R]
H4: training with hidden inputs [helps / does not help] (validation result [R]).

### 7. What we learned (Research novelty, 15 pts)
**Say:** H1: catchment attributes [improve / do not improve] transfer to unseen basins: [R] (ablation on
validation folds). Ungauged case: without any flow history, skill drops to [R], which shows how much
the model relies on knowing the river's current state.

### 8. Limits and responsible use (Real-world applicability, Responsible AI)
**Say:** US only; fewer snowmelt basins in the held-out set; reanalysis weather; statistical thresholds, not
official flood stages. Research benchmark, not a warning system. Deployment would need operational weather
feeds, local validation, expert review and agency coordination.

---

## Defense questions (12.2): one-line answers
1. *Generalization evidence?* 60 never-seen basins across 14 regions, scored separately with basin-bootstrap CIs.
2. *Inputs unavailable in deployment?* ERA5-Land delay; late flow data (tested as `obs_delay_1d`).
3. *Thresholds?* Each basin's 95th / 99th percentile flow over 1981–2008, fixed before modelling.
4. *Missing discharge history?* Last observation up to 30 days, then the basin's running mean; stress-tested.
5. *Biggest ablation effect?* [R]
6. *Uncertainty calibration?* Conformal widening (quantiles) and isotonic regression (probabilities), both
   fitted on basins held out from the trees; checked by coverage, pinball loss, Brier skill, reliability.
7. *Failure modes?* [R]: per-basin worst cases from `metrics` (flashy arid basins and snowmelt onsets expected).
8. *Before deployment?* Operational weather, local validation, monitoring, fail-safes, agency coordination.

## Demo (if asked)
`python basinshift_run.py --model src.bs_models:BasinshiftModel` end to end, then open
`outputs/<run>/REPORT.md` and `docs/figures/`.
