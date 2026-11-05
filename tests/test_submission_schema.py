"""The written submission matches docs/submission.schema.json (BASINSHIFT 8.3), on real predictions.

A small checker for the schema keywords we use (no extra dependency). Skipped without data/caravan.
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from src import baselines, submission

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "caravan"
SCHEMA = json.loads((ROOT / "docs" / "submission.schema.json").read_text())
pytestmark = pytest.mark.skipif(not (DATA / "train.parquet").exists(), reason="data/caravan not present")


def check(value, schema, root=SCHEMA, path="$"):
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        return check(value, root["$defs"][name], root, path)
    errs = []
    t = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "number": (int, float)}
    if t and not isinstance(value, types[t]):
        return [f"{path}: expected {t}"]
    if "const" in schema and value != schema["const"]:
        errs.append(f"{path}: expected {schema['const']}")
    if "pattern" in schema and not re.match(schema["pattern"], value):
        errs.append(f"{path}: {value!r} does not match {schema['pattern']}")
    if "minimum" in schema and value < schema["minimum"]:
        errs.append(f"{path}: {value} < {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        errs.append(f"{path}: {value} > {schema['maximum']}")
    if t == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 10**9):
            errs.append(f"{path}: wrong length {len(value)}")
        if "items" in schema:
            for i, v in enumerate(value):
                errs += check(v, schema["items"], root, f"{path}[{i}]")
    if t == "object":
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}: missing {k}")
        if schema.get("additionalProperties") is False:
            errs += [f"{path}: unexpected {k}" for k in value if k not in schema["properties"]]
        for k, sub in schema.get("properties", {}).items():
            if k in value:
                errs += check(value[k], sub, root, f"{path}.{k}")
    return errs


def test_written_submission_matches_schema(tmp_path):
    train = pd.read_parquet(DATA / "train.parquet")
    test = pd.read_parquet(DATA / "test.parquet")
    thr = pd.read_csv(DATA / "thresholds.csv", dtype={"gauge_id": str})
    ids = sorted(test["gauge_id"].unique())[:5]
    test = test[test["gauge_id"].isin(ids)].reset_index(drop=True)
    tr = train[train["gauge_id"].isin(ids)]
    m = baselines.fit(tr if len(tr) else train.head(50_000), None, thr)
    preds = baselines.predict(m, test, tr, None, thr)
    path = submission.write_jsonl(preds, tmp_path / "s.jsonl.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    assert len(records) == len(test)
    errs = [e for r in records for e in check(r, SCHEMA)]
    assert errs == [], errs[:5]


def test_schema_rejects_bad_records():
    good = {"catchment_id": "camels_01013500", "forecast_origin": "2009-01-15T00:00:00Z",
            "forecast_horizons_days": [1, 3, 7], "predicted_discharge": [1.0, 2.0, 3.0],
            "high_flow_probability": [0.1, 0.2, 0.3], "extreme_flow_probability": [0.0, 0.1, 0.1],
            "prediction_intervals": {"lower": [0.5, 1.0, 1.5], "upper": [2.0, 3.0, 4.0], "nominal_coverage": 0.8},
            "units": "mm/day"}
    assert check(good, SCHEMA) == []
    bad = [dict(good, high_flow_probability=[0.1, 1.2, 0.3]), dict(good, predicted_discharge=[-1.0, 2.0, 3.0]),
           dict(good, forecast_horizons_days=[1, 3]), dict(good, catchment_id="1013500"),
           {k: v for k, v in good.items() if k != "units"}, dict(good, extra=1)]
    for b in bad:
        assert check(b, SCHEMA) != []
