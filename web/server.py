"""BasinPulse website + API.

    python web/server.py            # http://localhost:8000
    HOST=0.0.0.0 PORT=7860 python web/server.py      # public hosting (Hugging Face Space, see deploy/)

GET  /                      the site (web/static/index.html)
GET  /api/summary           headline results (from logged runs, built by web/build_data.py)
GET  /api/rivers            the 60 unseen test rivers: skill, novelty, location
GET  /api/river/{id}        forecast vs observed around the river's largest flood, with ranges, P(high), drivers
POST /api/forecast          upload a CSV/parquet of river inputs (same columns as predict.py --inputs); the final
                            model (models/basinpulse_final.joblib) forecasts it live. Returns per-basin series.
GET  /api/sample            a small real input file (2 unseen rivers, 2013) to try /api/forecast with
GET  /api/health            model and data status
"""
from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import predict as P  # noqa: E402  (same input handling as the command-line predictor)
from src import submission  # noqa: E402

STATIC = ROOT / "web" / "static"
DATA = STATIC / "data"
MODEL_PATH = ROOT / "models" / "basinpulse_final.joblib"
# Limits for live forecasts; a small public host sets lower values through the environment (see render.yaml).
MAX_ROWS = int(os.environ.get("MAX_ROWS", "60000"))       # 60k = about 40 basin-years, fast on a laptop
MAX_BYTES = int(os.environ.get("MAX_BYTES", "15000000"))  # largest accepted upload
BUSY = threading.BoundedSemaphore(int(os.environ.get("MAX_LIVE", "2")))  # live forecasts running at once
app = FastAPI(title="BasinPulse", docs_url="/api/docs")
_model = None
_model_lock = threading.Lock()


def model():
    global _model
    with _model_lock:   # the startup warm-up and a first request never load it twice
        if _model is None:
            import joblib
            _model = joblib.load(MODEL_PATH)
    return _model


def _json(name: str):
    p = DATA / name
    if not p.exists():
        raise HTTPException(503, f"{name} not built yet: run python web/build_data.py")
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.get("/api/health")
def health():
    info = json.loads(MODEL_PATH.with_suffix(".json").read_text()) if MODEL_PATH.with_suffix(".json").exists() else {}
    return {"model": MODEL_PATH.exists(), "model_info": info, "data_built": (DATA / "summary.json").exists()}


@app.get("/api/summary")
def summary():
    return _json("summary.json")


@app.get("/api/rivers")
def rivers():
    return _json("rivers.json")


@app.get("/api/river/{gid}")
def river(gid: str):
    if not gid.replace("_", "").isalnum():
        raise HTTPException(400, "bad id")
    return _json(f"river/{gid}.json")


@app.get("/api/sample")
def sample():
    p = DATA / "sample_inputs.csv"
    if not p.exists():
        raise HTTPException(503, "sample not built yet")
    return Response(p.read_bytes(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=basinpulse_sample_inputs.csv"})


@app.on_event("startup")
def _warm() -> None:   # load the model in the background so the first live forecast is quick
    threading.Thread(target=model, daemon=True).start()


@app.post("/api/forecast")
def forecast(file: UploadFile = File(...)):   # plain def: runs in a worker thread, never blocks the site
    if not BUSY.acquire(timeout=30):
        raise HTTPException(503, "The model is busy with other forecasts; please try again in a minute")
    try:
        return _forecast(file)
    finally:
        BUSY.release()


def _forecast(file: UploadFile):
    t0 = time.time()
    raw = file.file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"File too large: please upload at most {MAX_BYTES // 1_000_000} MB")
    name = file.filename or ""
    try:
        df = pd.read_parquet(io.BytesIO(raw)) if name.endswith(".parquet") else pd.read_csv(
            io.BytesIO(raw), dtype={"gauge_id": str, "catchment_id": str, "station_id": str})
    except Exception as e:
        raise HTTPException(400, f"Could not read the file: {e}")
    try:
        test = P.normalise(df)
    except Exception as e:
        raise HTTPException(400, f"The file needs a basin id column (gauge_id) and a date column: {e}")
    if len(test) > MAX_ROWS:
        raise HTTPException(413, f"{len(test):,} rows: please upload at most {MAX_ROWS:,} rows for a live forecast")
    keys = test[[P.ID, P.DATE]].drop_duplicates().reset_index(drop=True)
    rows = P.continuous(test[[P.ID, P.DATE, P.OBS] + P.WEATHER])
    static = P.derive_static(rows, None)   # fills real attributes of known gauges (models/catchment_attributes)
    thr = P.derive_thresholds(rows, None)
    m = model()
    preds = m.predict(m, rows, rows.iloc[:0], static, thr)
    preds = keys.merge(preds, on=[P.ID, P.DATE], how="left")
    for c in ("predicted_discharge", "lower", "upper"):
        preds[c] = preds[c].fillna(0.0)
    preds["p_high"] = preds["p_high"].fillna(0.05)
    preds["p_extreme"] = np.minimum(preds["p_extreme"].fillna(0.01), preds["p_high"])
    errors = submission.validate(preds, keys)
    out = []
    for gid, g in preds.groupby(P.ID):
        obs = rows[rows[P.ID] == gid].set_index(P.DATE)[P.OBS]
        series = {"id": gid, "q_high": None if pd.isna(thr.set_index(P.ID).loc[gid, "q_high"])
                  else round(float(thr.set_index(P.ID).loc[gid, "q_high"]), 3)}
        for h in (1, 3, 7):
            gh = g[g.horizon == h].sort_values(P.DATE)
            days = gh[P.DATE] + pd.Timedelta(days=h)
            series[f"h{h}"] = {
                "dates": days.dt.strftime("%Y-%m-%d").tolist(),
                "forecast": gh["predicted_discharge"].round(3).tolist(),
                "lower": gh["lower"].round(3).tolist(), "upper": gh["upper"].round(3).tolist(),
                "p_high": gh["p_high"].round(3).tolist(),
                "observed": [None if pd.isna(v) else round(float(v), 3) for v in obs.reindex(days.to_numpy())]}
        out.append(series)
    return {"basins": out, "n_origins": len(keys), "valid_submission": errors == [], "problems": errors,
            "attributes_found": int(static.attrs.get("matched", 0)),
            "seconds": round(time.time() - t0, 1)}


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():   # browsers ask for /favicon.ico even when the page names an SVG icon
    return FileResponse(STATIC / "favicon.svg", media_type="image/svg+xml",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
