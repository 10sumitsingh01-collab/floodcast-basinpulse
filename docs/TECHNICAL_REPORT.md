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
