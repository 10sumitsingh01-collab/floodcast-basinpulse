"""Tests for the BASINSHIFT LightGBM model (Lane C) on a slice of the REAL Caravan package.

Skipped when data/caravan/ is not present. No synthetic data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import submission
from src.bs_models import (BasinshiftModel, MeanPointModel, NoFlowHistoryModel, NoStaticModel,
                           RobustMeanModel, build_features)
from src.evaluate import DATE, HI, ID, LO, PE, PH, PRED

DATA = Path(__file__).resolve().parents[1] / "data" / "caravan"
pytestmark = pytest.mark.skipif(not (DATA / "train.parquet").exists(), reason="data/caravan not present")


@pytest.fixture(scope="module")
def real():
    train = pd.read_parquet(DATA / "train.parquet")
    test = pd.read_parquet(DATA / "test.parquet")
    static = pd.read_parquet(DATA / "static.parquet")
    thr = pd.read_csv(DATA / "thresholds.csv", dtype={ID: str})
    tr_ids = sorted(train[ID].unique())[:12]
    test_ids = set(tr_ids[:3]) | set(sorted(set(test[ID]) - set(train[ID]))[:3])  # seen + unseen basins
    train = train[train[ID].isin(tr_ids)].reset_index(drop=True)
    test = test[test[ID].isin(test_ids)].reset_index(drop=True)
    return train, test, static, thr


def inputs_of(df):
    return df[[c for c in df.columns if not (c.startswith("streamflow_h") or c == "streamflow")]]


def test_features_use_only_the_past(real):
    train, _, static, thr = real
    x = inputs_of(train)
    cut = x[DATE].quantile(0.5)
    a = build_features(x, static, thr)
    a = a[a[DATE] <= cut].sort_values([ID, DATE]).reset_index(drop=True)
    b = build_features(x[x[DATE] <= cut], static, thr).sort_values([ID, DATE]).reset_index(drop=True)
    num = a.columns.drop([ID, DATE])
    assert np.allclose(a[num].to_numpy(float), b[num].to_numpy(float), equal_nan=True)


def test_features_keep_row_order_and_reject_targets(real):
    train, _, static, thr = real
    x = inputs_of(train).sample(frac=1.0, random_state=3)
    f = build_features(x, static, thr)
    assert (f[ID].to_numpy() == x[ID].to_numpy()).all() and (f[DATE].to_numpy() == x[DATE].to_numpy()).all()
    with pytest.raises(ValueError):
        build_features(train, static, thr)


@pytest.mark.parametrize("cls", [BasinshiftModel, MeanPointModel, RobustMeanModel, NoStaticModel,
                                 NoFlowHistoryModel])
def test_model_output_passes_submission_rules(real, cls):
    train, test, static, thr = real
    m = cls(max_rows=20_000, n_estimators=30).fit(train, static, thr)
    p = m.predict(m, test, train, static, thr)
    assert submission.validate(p, test[[ID, DATE]]) == []
    assert (p[LO] <= p[PRED]).all() and (p[PRED] <= p[HI]).all()
    assert (p[PE] <= p[PH]).all()


def test_prediction_does_not_depend_on_later_test_rows(real):
    train, test, static, thr = real
    m = BasinshiftModel(max_rows=20_000, n_estimators=30).fit(train, static, thr)
    full = m.predict(m, test, train, static, thr)
    cut = test[DATE].quantile(0.5)
    part = m.predict(m, test[test[DATE] <= cut].reset_index(drop=True), train, static, thr)
    key = [ID, DATE, "horizon"]
    a = full[full[DATE] <= cut].sort_values(key).reset_index(drop=True)
    b = part.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-5)
