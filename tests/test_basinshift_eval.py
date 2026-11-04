"""Tests for Lane D BASINSHIFT evaluation on a slice of the REAL Caravan package (no synthetic data).

Skipped when data/caravan/ is not present. Metric tests check known identities on real flows
(perfect forecast, forecasting the mean) and compare against independent direct formulas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import baselines, evaluate, stress, submission
from src.evaluate import DATE, HI, HORIZON, ID, LO, PE, PH, PRED

DATA = Path(__file__).resolve().parents[1] / "data" / "caravan"
pytestmark = pytest.mark.skipif(not (DATA / "test_labels.parquet").exists(), reason="data/caravan not present")


@pytest.fixture(scope="module")
def real():
    test = pd.read_parquet(DATA / "test.parquet")
    labels = pd.read_parquet(DATA / "test_labels.parquet")
    thr = pd.read_csv(DATA / "thresholds.csv", dtype={ID: str})
    train = pd.read_parquet(DATA / "train.parquet")
    # 6 basins per test set keeps tests fast
    keep = set()
    for ts in ("temporal", "geo", "geo_temporal"):
        keep |= set(sorted(labels.loc[labels.test_set == ts, ID].unique())[:6])
    test, labels = test[test[ID].isin(keep)], labels[labels[ID].isin(keep)]
    train = train[train[ID].isin(keep)]
    return train, test, labels, thr


def oracle(labels: pd.DataFrame, thr: pd.DataFrame) -> pd.DataFrame:
    """A 'forecast' equal to the real observed answer, to test metric identities."""
    lab = evaluate.labels_long(labels)
    p = lab[[ID, DATE, HORIZON]].copy()
    y = lab["y_true"].fillna(0).to_numpy()
    p[PRED], p[LO], p[HI] = y, y, y
    t = lab.merge(thr, on=ID)
    p[PH] = (t["y_true"] > t["q_high"]).astype(float).to_numpy()
    p[PE] = (t["y_true"] > t["q_extreme"]).astype(float).to_numpy()
    return p


def test_perfect_forecast_identities(real):
    _, _, labels, thr = real
    m, _ = evaluate.score(oracle(labels, thr), labels, thr, n_boot=0)
    v = m.set_index(["test_set", "horizon", "metric"])["value"]
    for ts in ("temporal", "geo", "geo_temporal"):
        assert v[(ts, 1, "nse_median")] == pytest.approx(1.0)
        assert v[(ts, 1, "kge_median")] == pytest.approx(1.0)
        assert v[(ts, 1, "mae_pooled")] == pytest.approx(0.0)
        assert v[(ts, 1, "brier_high")] == pytest.approx(0.0)
        assert v[(ts, 1, "auprc_high")] == pytest.approx(1.0)
        assert v[(ts, 1, "interval_coverage")] == pytest.approx(1.0)


def test_mean_forecast_has_zero_nse(real):
    _, _, labels, thr = real
    p = oracle(labels, thr)
    lab = evaluate.labels_long(labels).dropna(subset=["y_true"])
    means = lab.groupby(["test_set", HORIZON, ID])["y_true"].transform("mean")
    p = p.merge(lab[[ID, DATE, HORIZON]].assign(m=means.to_numpy()), on=[ID, DATE, HORIZON], how="left")
    p[PRED] = p["m"].fillna(0)
    p = p.drop(columns="m")
    # a basin appears in only one test set per date range, so per-set basin means are exact
    m, _ = evaluate.score(p, labels, thr, n_boot=0)
    v = m.set_index(["test_set", "horizon", "metric"])["value"]
    assert v[("temporal", 1, "nse_median")] == pytest.approx(0.0, abs=1e-6)


def test_metrics_match_direct_formulas(real):
    train, test, labels, thr = real
    model = baselines.fit(train, None, thr)
    preds = baselines.predict(model, test, train, None, thr)
    m, _ = evaluate.score(preds, labels, thr, n_boot=0)
    v = m.set_index(["test_set", "horizon", "metric"])["value"]
    j = evaluate.join(preds, labels, thr)
    g = j[(j.test_set == "geo") & (j[HORIZON] == 3)]
    assert v[("geo", 3, "mae_pooled")] == pytest.approx(np.mean(np.abs(g.y_true - g[PRED])))
    assert v[("geo", 3, "brier_high")] == pytest.approx(np.mean((g[PH] - g.ev_high) ** 2))
    cov = np.mean((g.y_true >= g[LO]) & (g.y_true <= g[HI]))
    assert v[("geo", 3, "interval_coverage")] == pytest.approx(cov)
    nse = g.groupby(ID).apply(lambda d: 1 - ((d.y_true - d[PRED]) ** 2).sum()
                              / ((d.y_true - d.y_true.mean()) ** 2).sum(), include_groups=False)
    assert v[("geo", 3, "nse_median")] == pytest.approx(nse.median())


def test_binned_auprc_close_to_exact(real):
    train, test, labels, thr = real
    model = baselines.fit(train, None, thr)
    preds = baselines.predict(model, test, train, None, thr)
    g = evaluate.join(preds, labels, thr)
    g = g[g[HORIZON] == 1]
    sums, hists = evaluate.basin_stats(g)
    from sklearn.metrics import average_precision_score
    exact = average_precision_score(g.ev_high, g[PH])
    assert evaluate._binned_ap(hists["high"].sum(axis=0)) == pytest.approx(exact, abs=0.01)


def test_baseline_submission_is_valid_and_round_trips(real, tmp_path):
    train, test, labels, thr = real
    model = baselines.fit(train, None, thr)
    preds = baselines.predict(model, test, train, None, thr)
    assert submission.validate(preds, test[[ID, DATE]]) == []
    path = submission.write_jsonl(preds, tmp_path / "s.jsonl.gz")
    back = submission.read_jsonl(path)
    assert submission.validate(back, test[[ID, DATE]]) == []
    assert len(back) == len(preds)


@pytest.mark.parametrize("breakage, expected", [
    (lambda p: p.assign(**{PRED: -p[PRED] - 1}), "negative predicted_discharge"),
    (lambda p: p.assign(**{PH: p[PH] + 2}), "p_high outside"),
    (lambda p: p.assign(**{LO: p[HI] + 1}), "lower > upper"),
    (lambda p: p.assign(**{PE: 1.0, PH: 0.0}), "p_extreme > p_high"),
    (lambda p: p.assign(**{PRED: np.nan}), "NaN or infinity"),
    (lambda p: pd.concat([p, p.head(5)]), "duplicate"),
    (lambda p: p.iloc[10:], "missing"),
    (lambda p: p.assign(**{HORIZON: p[HORIZON].replace(7, 5)}), "unexpected horizons"),
    (lambda p: p.assign(**{ID: p[ID].where(p.index > 0, "camels_99999999")}), "unknown catchment"),
])
def test_validator_rejects_bad_submissions(real, breakage, expected):
    train, test, _, thr = real
    preds = baselines.predict(baselines.fit(train, None, thr), test, train, None, thr)
    errs = submission.validate(breakage(preds.copy()), test[[ID, DATE]])
    assert any(expected in e for e in errs), errs


def test_stress_scenarios_only_remove_or_perturb(real):
    _, test, _, _ = real
    for sc in stress.SCENARIOS:
        pt = stress.perturb(test, sc, seed=1)
        assert len(pt) == len(test) and list(pt.columns) == list(test.columns)
        assert set(pt[ID]) == set(test[ID])
    m30 = stress.perturb(test, "mask_30pct", seed=1)
    frac = m30["total_precipitation_sum"].isna().mean() - test["total_precipitation_sum"].isna().mean()
    assert 0.27 < frac < 0.33
    assert stress.perturb(test, "mask_10pct", seed=7).equals(stress.perturb(test, "mask_10pct", seed=7))


def test_baseline_uses_only_past_observations(real):
    """Persistence at origin t must not change when every later observation is removed."""
    train, test, _, thr = real
    model = baselines.fit(train, None, thr)
    full = baselines.predict(model, test, train, None, thr)
    cut = test[DATE].quantile(0.5)
    part = baselines.predict(model, test[test[DATE] <= cut], train, None, thr)
    a = full[full[DATE] <= cut].sort_values([ID, DATE, HORIZON]).reset_index(drop=True)
    b = part.sort_values([ID, DATE, HORIZON]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
