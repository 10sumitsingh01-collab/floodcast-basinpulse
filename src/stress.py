"""Stress tests on REAL test inputs (BASINSHIFT 7.5, 10.1 "Robustness"), Lane D.

Each scenario corrupts test.parquet inputs reproducibly (fixed seed), then the SAME model pipeline
rebuilds features and predicts. Robustness = how much each metric degrades vs the clean run.
Nothing here creates data: values are only removed, delayed or perturbed with noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ID, DATE = "gauge_id", "date"
OBS = "streamflow_obs"
PRECIP = "total_precipitation_sum"

SCENARIOS = {
    "mask_10pct": "10% of input values removed at random",
    "mask_30pct": "30% of input values removed at random",
    "gaps_3to7d": "random 3-7 day gaps covering ~10% of days, all inputs missing",
    "noise_weather": "Gaussian noise, sd = 10% of each basin's sd, on weather inputs",
    "obs_delay_1d": "latest observed flow arrives 1 day late (age not reported)",
    "obs_delay_1d_flagged": "latest observed flow arrives 1 day late, and its age is reported (obs_age_days = 1)",
}


def input_columns(test: pd.DataFrame) -> list[str]:
    return [c for c in test.columns if c not in (ID, DATE)]


def perturb(test: pd.DataFrame, scenario: str, seed: int = 42) -> pd.DataFrame:
    if scenario not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario}; choose from {list(SCENARIOS)}")
    rng = np.random.default_rng(seed)
    df = test.copy()
    cols = input_columns(df)

    if scenario.startswith("mask_"):
        frac = int(scenario.split("_")[1].rstrip("pct")) / 100
        for c in cols:
            df.loc[rng.random(len(df)) < frac, c] = np.nan

    elif scenario == "gaps_3to7d":
        df = df.sort_values([ID, DATE]).reset_index(drop=True)
        hit = np.zeros(len(df), bool)
        for _, idx in df.groupby(ID).indices.items():
            n = len(idx)
            target = int(0.10 * n)
            covered = 0
            while covered < target:
                L = int(rng.integers(3, 8))
                s = int(rng.integers(0, max(1, n - L)))
                hit[idx[s:s + L]] = True
                covered += L
        df.loc[hit, cols] = np.nan

    elif scenario == "noise_weather":
        wx = [c for c in cols if c != OBS]
        sd = df.groupby(ID)[wx].transform("std").fillna(0)
        df[wx] = df[wx] + rng.normal(0, 1, (len(df), len(wx))) * 0.1 * sd.to_numpy()
        if PRECIP in df:
            df[PRECIP] = df[PRECIP].clip(lower=0)

    elif scenario.startswith("obs_delay_1d") and OBS in df:
        prev = df[[ID, DATE, OBS]].assign(**{DATE: df[DATE] + pd.Timedelta(days=1)})
        df = df.drop(columns=OBS).merge(prev, on=[ID, DATE], how="left")[test.columns]

    for c in cols:
        if pd.api.types.is_float_dtype(test[c]):
            df[c] = df[c].astype(test[c].dtype)
    if scenario == "obs_delay_1d_flagged":
        df["obs_age_days"] = 1.0   # models that read it know the reading is a day old; others ignore it
    return df
