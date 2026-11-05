# Lane C model (yash)

`models.py`, `config.py` and `test_models_lane_c.py` are Lane C's BASINSHIFT model as pushed on branch `c/models`
(commit a94bb71), placed here so they do not collide with the legacy FLOODCAST files. Only these three files were
taken from that branch: its other changes (the Data v2 data files and `data/holdout/holdout_labels.parquet`)
are deliberately not included. `src/lane_c_adapter.py` wraps the model in the standard
`fit(train, static, thresholds)` / `predict(...)` interface so `basinshift_run.py` can validate and score it
exactly like every other model.
