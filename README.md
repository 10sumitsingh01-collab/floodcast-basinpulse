# BasinPulse: FLOODCAST Early Warning River Discharge Model

**Team Name**: BasinPulse  
**Challenge**: FLOODCAST — Predict the Surge Before It Strikes (24h Hackathon)

## Approach Summary
BasinPulse predicts daily river discharge (`discharge_cfs`) across diverse US river basins using an ensemble of gradient-boosted trees specifically tuned for hydrological flood peaks. Our feature engineering captures physical basin mechanics including precipitation volume scaling ($\text{rain} \times \text{area}$), exponential recession hydrographs, antecedent soil moisture saturation, and temperature-driven snowmelt. To address the competition metric (40% overall RMSE, 30% top-10% flood peak RMSE), we deploy sample-weighted LightGBM and XGBoost models combined with Duan's smearing-calibrated log models. Strict temporal forward-chaining validation ensures zero target leakage and accurate generalization to future test periods.

## Repository Layout
```
├── src/
│   ├── config.py         # Shared constants, paths, and competition parameters
│   ├── data.py           # Lane A: Data loading, schema enforcement (real data only)
│   ├── features.py       # Lane B: Hydrological feature pipeline (zero target leakage)
│   ├── models.py         # Lane C: Peak-boosted sample weighting & ensemble models
│   └── validate.py       # Lane D: Forward CV, LOSO CV, scoring, finalize & verification
├── solution.py           # Lane D: Thin pipeline orchestrator
├── tests/
│   ├── smoke.py          # Lane D: Comprehensive end-to-end smoke test suite
│   ├── test_data.py      # Unit tests for Lane A
│   ├── test_features.py  # Unit tests for Lane B
│   ├── test_models.py    # Unit tests for Lane C
│   └── test_validate.py  # Unit tests for Lane D
├── experiments/
│   └── results.csv       # Append-only experiment tracking log (real data only)
├── requirements.txt      # Core project dependencies
├── model_card.md         # Comprehensive model card (benchmarks: TBD: real data pending)
└── NOTES.md              # Cross-lane architectural log
```

## How to Run

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Smoke Tests
Verify all lane interfaces, dtypes, schemas, and pipeline integrity:
```bash
python tests/smoke.py
```
Or run the unit test suite with pytest:
```bash
pytest tests/
```

### 3. Run Solution Pipeline (Once real data is in place)
```bash
python solution.py
```
*(Requires real `data/train.csv` and `data/test.csv`)*
