"""predict.py runs on unseen-style inputs: identical to the pipeline on complete inputs, robust on messy ones."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src import submission
from src.bs_models import RobustMeanModel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "caravan"
pytestmark = pytest.mark.skipif(not (DATA / "train.parquet").exists(), reason="data/caravan not present")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    train = pd.read_parquet(DATA / "train.parquet")
    test = pd.read_parquet(DATA / "test.parquet")
    static = pd.read_parquet(DATA / "static.parquet")
    thr = pd.read_csv(DATA / "thresholds.csv", dtype={"gauge_id": str})
    ids = sorted(train.gauge_id.unique())[:10]
    tr = train[train.gauge_id.isin(ids)].reset_index(drop=True)
    unseen = sorted(set(test.gauge_id) - set(train.gauge_id))[:2]
    te = test[test.gauge_id.isin(ids[:2] + unseen)]
    te = te[te.date < "2009-06-01"].reset_index(drop=True)
    m = RobustMeanModel(max_rows=20_000, n_estimators=30).fit(tr, static, thr)
    joblib.dump(m, d / "m.joblib")
    hist = pd.concat([tr[te.columns], test[test.gauge_id.isin(unseen) & (test.date < "2008-10-01")]])
    hist = hist[hist.gauge_id.isin(te.gauge_id.unique()) & (hist.date >= "2008-01-01")]
    return d, m, tr, te, hist, static, thr


def run(*args):
    r = subprocess.run([sys.executable, str(ROOT / "predict.py"), *map(str, args)], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_cli_matches_pipeline_on_complete_inputs(setup):
    d, m, tr, te, hist, static, thr = setup
    te.to_parquet(d / "in.parquet"); hist.to_parquet(d / "hist.parquet")
    static.to_parquet(d / "st.parquet"); thr.to_csv(d / "thr.csv", index=False)
    run("--inputs", d / "in.parquet", "--history", d / "hist.parquet", "--static", d / "st.parquet",
        "--thresholds", d / "thr.csv", "--model", d / "m.joblib", "--out", d / "s.jsonl.gz", "--long-out", d / "p.parquet")
    got = pd.read_parquet(d / "p.parquet")
    exp = m.predict(m, te, hist, static, thr)
    k = ["gauge_id", "date", "horizon"]
    got, exp = got.sort_values(k).reset_index(drop=True), exp.sort_values(k).reset_index(drop=True)
    for c in ("predicted_discharge", "lower", "upper", "p_high", "p_extreme"):
        assert np.allclose(got[c], exp[c], rtol=1e-5), c


def test_cli_survives_messy_inputs(setup):
    d, m, tr, te, hist, static, thr = setup
    messy = te.rename(columns={"gauge_id": "catchment_id", "date": "forecast_origin"}).drop(
        columns=["surface_net_solar_radiation_mean", "dewpoint_temperature_2m_mean"])
    messy.to_csv(d / "messy.csv", index=False)          # CSV, other column names, two weather inputs missing,
    run("--inputs", d / "messy.csv", "--model", d / "m.joblib", "--out", d / "m.jsonl.gz")  # no history/static/thr
    back = submission.read_jsonl(d / "m.jsonl.gz")
    assert submission.validate(back, te[["gauge_id", "date"]]) == []
