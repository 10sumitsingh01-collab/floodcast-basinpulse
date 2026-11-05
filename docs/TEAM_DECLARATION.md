# Team declaration (BASINSHIFT deliverable 10)

**Team:** BasinPulse
**Repository:** https://github.com/10sumitsingh01-collab/floodcast-basinpulse

> Fill in the names and confirm every row before submission. Everything here must be true.

## Authors and roles
| Name | GitHub | Lane | Main contributions |
|---|---|---|---|
| _fill in_ | 10sumitsingh01-collab | Team lead, integration | Repository, merges, early FLOODCAST pipeline and Data v2 (superseded, kept in `legacy/`) |
| Aarav Vatsh | AaravVatsh27 | A: data; also C and D work | Caravan data pipeline and leakage audit; BASINSHIFT model, evaluation, stress tests, submission format, documentation |
| _fill in_ | _fill in_ | B: features | _fill in_ |
| _fill in_ | _fill in_ | C: models | _fill in_ |
| _fill in_ | _fill in_ | D: validation, delivery | _fill in_ |

## Use of AI tools
AI coding assistants were used during development (including Claude Code by Anthropic), for writing and reviewing
code and documentation under the team's direction. Every reported number comes from code in this repository run on
real data; results were checked by automated tests (`tests/`), a 15-check leakage audit
(`scripts/caravan/check_leakage.py`) and a fresh-clone reproduction that matched the final predictions bit for bit.
_Team: list any other AI tools you used (e.g. other coding agents) here._

## Data and third-party components (11.1)
| Component | Use | License |
|---|---|---|
| Caravan v1.6 (Kratzert et al. 2023), CAMELS-US subset (Newman et al. 2015) | training and US test data | CC-BY-4.0 |
| ERA5-Land (Muñoz-Sabater et al. 2021), via Caravan | weather inputs | Copernicus licence, as distributed in Caravan |
| HydroATLAS (Linke et al. 2019), via Caravan | catchment attributes | CC-BY-4.0 |
| LaMAH-CE (Klingler et al. 2021), via Caravan | zero-shot evaluation only (Central Europe); not redistributed | CC-BY-SA-4.0 |
| LightGBM, scikit-learn, pandas, NumPy, SciPy, PyArrow, Matplotlib, pytest | software | MIT / BSD / Apache-2.0 / PSF (open source) |

No other external datasets, no pretrained weights or foundation models, and no synthetic data were used for
training, validation or reported results. No proprietary API is needed to run the pipeline.

## Integrity statements
- The test labels were never used for training, tuning or model selection; all choices were made on validation
  splits inside the training data, with selection rules written down before results were seen
  (`docs/EXPERIMENTS.md`, `outputs/queue*.sh`).
- No predictions were edited by hand.
- The submitted model is the one in this repository at the tagged commit; its predictions are reproducible from a
  fresh clone (`docs/TECHNICAL_REPORT.md`, section 11).

## Conflicts of interest
_None known. Team: confirm or declare._
