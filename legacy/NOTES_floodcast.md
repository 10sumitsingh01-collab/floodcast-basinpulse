# Team Notes & Coordination Log

## Phase 0 Real Data Status & Zero-Simulation Rule
- **CRITICAL RULE**: NEVER invent or fabricate data. All simulator code and synthetic files have been completely purged from the repository.
- **Lane A (`src/data.py`)**: `generate_starter_dataset` deleted. `load_train()`, `load_test()`, and `load_station_metadata()` strictly load real CSV files; if missing, they raise `FileNotFoundError` stating real data is pending.
- **Lane B (`src/features.py`)**, **Lane C (`src/models.py`)**, **Lane D (`src/validate.py`, `solution.py`)**: Preserved as built, but no model training or scoring will be performed until real data is downloaded and verified.
- **Metrics**: All metrics in `model_card.md` and `README.md` have been reset to `TBD: real data pending`. `experiments/results.csv` cleared.
