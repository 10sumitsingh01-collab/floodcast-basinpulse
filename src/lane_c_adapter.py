"""Lane C's model (src/lane_c/models.py) with Lane B's features (src/bs_features.py), in the standard interface.

    python basinshift_run.py --validate --folds 0,1 --model src.lane_c_adapter:LaneCModel --name val_lane_c

fit(train, static, thresholds) / predict(model, test, train, static, thresholds) -> long predictions, exactly like
every other model, so validation and scoring are identical. Lane C predicts the flow level in log1p space with
quantile LightGBM (Q10/Q50/Q90) and calibrated exceedance classifiers whose thresholds are the 95th / 99th
percentile of each training basin's flow (computed inside Lane C's code).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import bs_features as B
from src.lane_c import models as C
from src.evaluate import DATE, HI, HORIZON, ID, LO, PE, PH, PRED

HORIZONS = (1, 3, 7)
HISTORY_DAYS = 150


class LaneCModel:
    def fit(self, train: pd.DataFrame, static: pd.DataFrame, thresholds: pd.DataFrame) -> "LaneCModel":
        train = train.reset_index(drop=True)
        X = B.build_features(train[B.INPUT_COLS], static)[B.FEATURE_COLS]
        self.bundle = C.fit_models(X, train_df=train[[ID, DATE, "streamflow"]], station_col=ID,
                                   target_col="streamflow", include_baselines=False)
        return self

    def predict(self, model, test: pd.DataFrame, train: pd.DataFrame, static: pd.DataFrame,
                thresholds: pd.DataFrame) -> pd.DataFrame:
        start = test.groupby(ID)[DATE].min()
        hist = train.loc[train[ID].isin(start.index), B.INPUT_COLS]
        first = hist[ID].map(start)
        hist = hist[(hist[DATE] >= first - pd.Timedelta(days=HISTORY_DAYS)) & (hist[DATE] < first)]
        both = pd.concat([hist, test[B.INPUT_COLS]], ignore_index=True)
        X = B.build_features(both, static).iloc[len(hist):].reset_index(drop=True)[B.FEATURE_COLS]
        keys = test[[ID, DATE]].reset_index(drop=True)
        p = C.predict_multi_horizon(self.bundle, X, df=keys, include_baselines=False)
        parts = []
        for h in HORIZONS:
            q = np.sort(np.column_stack([p[f"q{k}_h{h}"].to_numpy(float) for k in ("10", "50", "90")]), axis=1)
            q = np.clip(np.nan_to_num(q, nan=0.0), 0, None)
            ph = np.clip(np.nan_to_num(p[f"prob_q_high_h{h}"].to_numpy(float), nan=0.05), 0, 1)
            pe = np.clip(np.nan_to_num(p[f"prob_q_extreme_h{h}"].to_numpy(float), nan=0.01), 0, 1)
            out = keys.copy()
            out[ID] = out[ID].astype(str)
            out[HORIZON] = h
            out[LO], out[PRED], out[HI] = q[:, 0], q[:, 1], q[:, 2]
            out[PH], out[PE] = ph, np.minimum(pe, ph)
            parts.append(out)
        return pd.concat(parts, ignore_index=True)
