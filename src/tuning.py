"""Accuracy tuning candidates and seed ensembling (validation only; see docs/EXPERIMENTS.md).

All candidates are the adopted DelayAwareModel (robustness training, data-age input, mean forecast) with a
larger training sample and different LightGBM settings. Usage:
    python basinshift_run.py --validate --folds 0 --no-temporal --extra-train data/caravan_extra \\
        --model src.tuning:T1 --name tune_T1
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.bs_models import DelayAwareModel
from src.evaluate import DATE, HI, HORIZON, ID, LO, PE, PH, PRED


class T0(DelayAwareModel):
    """Reference: validated LightGBM settings, 600k training rows, 500 trees."""
    def __init__(self, **kw):
        kw.setdefault("max_rows", 600_000)
        kw.setdefault("n_estimators", 500)
        super().__init__(**kw)


class T1(T0):
    """More leaves, smaller minimum leaf size."""
    def __init__(self, **kw):
        kw.setdefault("lgb_params", {"num_leaves": 127, "min_child_samples": 100})
        super().__init__(**kw)


class T2(T0):
    """Much more capacity, slower learning."""
    def __init__(self, **kw):
        kw.setdefault("n_estimators", 700)
        kw.setdefault("lgb_params", {"num_leaves": 255, "learning_rate": 0.04, "min_child_samples": 300})
        super().__init__(**kw)


class T3(T0):
    """More leaves with stronger feature subsampling (decorrelated trees)."""
    def __init__(self, **kw):
        kw.setdefault("n_estimators", 600)
        kw.setdefault("lgb_params", {"num_leaves": 127, "colsample_bytree": 0.6, "learning_rate": 0.05})
        super().__init__(**kw)


class SeedEnsemble:
    """Average of independently trained members (different seeds -> different samples, masks, trees).

    Quantiles and probabilities are averaged per row (quantile averaging keeps the range calibrated in practice);
    order constraints are re-applied afterwards.
    """
    member_cls = T0
    seeds = (42, 7, 2024)

    def __init__(self, **kw):
        self.kw = kw
        self.members = []

    def fit(self, train, static, thresholds):
        self.members = []
        for s in self.seeds:
            print(f"  ensemble member seed {s}", flush=True)
            self.members.append(self.member_cls(seed=s, **self.kw).fit(train, static, thresholds))
        return self

    def predict(self, model, test, train, static, thresholds):
        k = [ID, DATE, HORIZON]
        preds = [m.predict(m, test, train, static, thresholds).sort_values(k).reset_index(drop=True)
                 for m in self.members]
        out = preds[0][k].copy()
        for c in (PRED, LO, HI, PH, PE):
            out[c] = np.mean([p[c].to_numpy(float) for p in preds], axis=0)
        out[LO] = np.minimum(out[LO], out[PRED])
        out[HI] = np.maximum(out[HI], out[PRED])
        out[PE] = np.minimum(out[PE], out[PH])
        return out


def ensemble_of(member):
    return type(f"Ensemble{member.__name__}", (SeedEnsemble,), {"member_cls": member})


EnsembleT0 = ensemble_of(T0)
EnsembleT1 = ensemble_of(T1)
EnsembleT2 = ensemble_of(T2)
EnsembleT3 = ensemble_of(T3)
