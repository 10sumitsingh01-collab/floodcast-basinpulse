# Team Notes & Coordination Log

## Architecture & Lane Boundaries
- **Lane A (`src/data.py`)**: Strict typing, contract compliance, date normalization to midnight `datetime64[ns]`, string `station_id`, and synthetic starter generator with realistic hydrological physics.
- **Lane B (`src/features.py`)**: Catchment morphology, lagged precipitation, exponential moving averages (unit hydrographs), snowmelt proxies, seasonal cycles. Completely isolated from target variable `discharge_cfs` with test-verified zero leakage.
- **Lane C (`src/models.py`)**: Extreme-event sample weighting (`make_weights`, default top-10% boost factor 5.0), dual-objective ensemble (LightGBM on raw cfs + XGBoost on raw cfs + LightGBM on log1p with Duan's smearing factor calibration), and non-negative finite post-processing.
- **Lane D (`src/validate.py`, `solution.py`, `tests/smoke.py`)**: Forward-chaining temporal cross-validation with 30-day purging buffer, Leave-One-Station-Out (LOSO) spatial validation, exact metric calculation (40% overall RMSE, 30% top-10% RMSE, 20% MAE, 10% R2), automated submission finalization, and strict verification.

## Cross-Lane Requests & Decisions
1. **Purity of `build_features`**: Ensured that `build_features` accepts arbitrary combinations of train and test rows without mutating row indices or column order, and guarantees that any presence or absence of `discharge_cfs` produces identical feature matrices.
2. **Extreme Flood Weighting**: Since top-10% flood days per station carry 30% of the competition score and dominate raw RMSE, `make_weights` was calibrated to compute 90th percentile thresholds per station strictly on the current fold's training rows (complying with Rule 8).
3. **Smearing Factor**: The log-space model uses Duan's smearing estimate calculated exclusively from the training fold residuals to prevent exponential transformation downward bias.
