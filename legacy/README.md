# Legacy: earlier FLOODCAST pipeline (not part of the BASINSHIFT entry)

Before the team moved to the BASINSHIFT specification, the repository held a FLOODCAST pipeline: same-day
discharge regression for 14 USGS stations ("Data v2", USGS NWIS + Open-Meteo weather at gauge points), scored by an
RMSE composite. It is kept here unchanged for transparency. It is **not** used by any BASINSHIFT result, its model
card describes a design that was never evaluated on real data, and its imports (`src.config`, `src.data`, ...) refer
to its original layout. Its dependencies are in `legacy/requirements.txt`.

The BASINSHIFT entry starts at the repository root: `README.md`, `basinshift_run.py`, `src/`, `docs/`.
