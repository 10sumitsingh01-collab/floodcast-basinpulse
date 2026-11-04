# BasinPulse: technical report (BASINSHIFT 12.1)

Status: skeleton. Every number must come from a logged `basinshift_run.py` run (commit hash in
`experiments/basinshift_results.csv`). Anything not yet measured says TBD.

## 1. Abstract and executive summary
TBD after final model. One paragraph: task, data (299 real CAMELS-US basins), method, headline
`geo_temporal` results vs. persistence, main limitation.

## 2. Problem formulation and research hypothesis
At the end of day t, for basin c, predict flow on t+1, t+3 and t+7 (mm/day), an 80% interval, and
P(flow > q_high(c)) and P(flow > q_extreme(c)), where q_high/q_extreme are the basin's 95th/99th
percentile flow over 1981-2008. Main hypothesis (H1): static catchment attributes improve transfer to
basins never seen in training. Secondary (H4): training with missing inputs improves robustness.

## 3. Dataset and feature provenance
Caravan v1.6 (CSV, CC-BY-4.0), CAMELS-US subset: observed USGS daily streamflow, ERA5-Land
basin-averaged weather, HydroATLAS attributes. 671 basins downloaded, 611 passed QA (coverage >= 90% of
the train period, >= 50% of the test period, weather <= 5% missing), 299 sampled in proportion to 14
USGS regions. See `docs/DATA_README.md`, `basin_selection.csv`, `PREP_REPORT.md`.
Features: TBD (Lane B, `src/features.py`).

## 4. Preprocessing and leakage controls
- Basins split first (60 held out entirely, stratified by region), then time (test from 2008-10-01).
- Targets in train never reach into the test period; thresholds and climate indices from 1981-2008 only.
- HydroATLAS attributes derived from modelled discharge/runoff removed.
- Missing streamflow never filled; negatives set missing.
- 15 automated checks in `scripts/caravan/check_leakage.py`, all passing (`LEAKAGE_REPORT.md`).
- Labels are loaded only after predictions are written and validated.

## 5. Model architecture and training procedure
TBD (Lane C). Reference baseline: calibrated persistence (`src/baselines.py`).

## 6. Baseline comparison and ablation study
TBD. Table: persistence vs. final model per test set and horizon; H1 with vs. without static attributes.

## 7. Temporal and unseen-basin results
TBD. From `outputs/<run>/REPORT.md`: temporal, geo, geo_temporal; NSE/KGE medians with basin-bootstrap CIs.

## 8. Extreme-event and calibration results
TBD. AUPRC, Brier skill, recall/precision/false alarms with event counts; reliability tables; interval coverage.

## 9. Robustness and efficiency analysis
TBD. `stress.csv`: relative change under the 5 scenarios; runtime and memory from `run.json`.

## 10. Limitations, failure cases and responsible use
- CAMELS-US only: one country, mostly natural catchments; global transfer is not shown.
- Held-out basins are gauged at inference (their past flow is an input); a no-flow-history run is needed
  to speak about ungauged basins.
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
   regions; reported separately as `geo` and `geo_temporal`, with basin-bootstrap CIs. Numbers: TBD.
2. **Inputs unavailable in deployment?** ERA5-Land is reanalysis, available with ~5 days delay; same-day
   observed flow may also arrive late (tested by `obs_delay_1d`).
3. **How are thresholds defined and frozen?** 95th/99th percentile of each basin's observed flow,
   1981-10-01 to 2008-09-30, computed once in `prepare_caravan.py` before any modelling.
4. **Behaviour when discharge history is missing?** Stress tests `mask_*`, `gaps_3to7d`, `obs_delay_1d`;
   persistence falls back to the last observation within 30 days. Model behaviour: TBD.
5. **Which ablation changed the result most?** TBD.
6. **How is uncertainty calibrated?** Quantile intervals checked by coverage (target 80%) and pinball loss;
   probabilities by Brier skill and reliability tables. Baseline coverage is 0.80 on every test set.
7. **Largest failure modes?** TBD from per-basin results (expect arid/flashy and snow-dominated basins).
8. **What before real deployment?** Local validation, operational weather feeds, expert review, monitoring,
   fail-safes, coordination with agencies (16.1).
