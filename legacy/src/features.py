"""Feature engineering module (Lane B).

Shared interfaces:
  - FEATURE_COLS: list[str]
  - build_features(df: pd.DataFrame) -> pd.DataFrame

Hard rules enforced:
  - Rule 1: NEVER use lagged/rolling discharge as a feature.
  - Rule 2: NEVER use discharge from the test period anywhere.
  - Rule 8: Never compute target-based aggregates here (CV-fold train only in models).
  - Preserves exact row ordering, index, and original columns.
  - Output contains all columns named in FEATURE_COLS.
"""

from typing import List
import numpy as np
import pandas as pd

from src.config import CORE_FEATURE_COLS, DATE_COL, STATION_COL, TARGET_COL

# Explicit list of all features engineered by build_features
ENGINEERED_FEATURE_COLS: List[str] = [
    # Basin morphology & scale
    "log_drainage_area",
    "relief_ratio",
    "concentration_time_proxy",
    # Rainfall volume interactions
    "precip_volume",
    "log_precip_volume",
    "precip_soil",
    "precip_soil_volume",
    # Lagged precipitation
    "precip_lag1",
    "precip_lag2",
    "precip_lag3",
    "precip_lag5",
    "precip_lag7",
    "precip_lag14",
    # Rolling precipitation sums and maximums
    "precip_roll_sum_3d",
    "precip_roll_sum_7d",
    "precip_roll_sum_14d",
    "precip_roll_sum_30d",
    "precip_roll_sum_60d",
    "precip_roll_max_7d",
    "precip_roll_max_14d",
    # Exponentially weighted precipitation (unit hydrographs)
    "precip_ewm_2d",
    "precip_ewm_5d",
    "precip_ewm_10d",
    "precip_ewm_30d",
    # Soil moisture dynamics
    "soil_moisture_lag1",
    "soil_moisture_diff1",
    "soil_moisture_roll_mean_7d",
    "soil_moisture_roll_mean_30d",
    # Temperature and snowmelt proxies
    "temp_mean",
    "temp_range",
    "freeze_flag",
    "melt_potential",
    "temp_mean_roll_7d",
    "temp_mean_roll_30d",
    # Seasonality / Climatology proxies
    "sin_doy",
    "cos_doy",
    "sin_month",
    "cos_month",
    # Interaction terms
    "effective_storm",
    "basin_effective_storm",
    "forest_attenuation",
]

# Total feature columns: core physical inputs + engineered features
FEATURE_COLS: List[str] = CORE_FEATURE_COLS + ENGINEERED_FEATURE_COLS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build hydrologically-informed features from meteorological and catchment attributes.

    Input:
      df: pd.DataFrame containing train+test rows concatenated (sorted by station_id, date).
          discharge_cfs may be present or NaN and MUST be ignored.
    Output:
      pd.DataFrame with same rows/index, original columns plus engineered columns;
      every column named in FEATURE_COLS exists.
    """
    # Defensive copy preserving original index
    orig_index = df.index
    out = df.copy()

    # Rule 1 & Rule 2: NEVER use discharge_cfs for any feature calculation!
    # Even if present, we do not touch it.

    # 1. Basin morphology & scaling
    area = np.maximum(0.1, out["drainage_area_sqmi"].values)
    elev = np.maximum(0.0, out["mean_elevation_ft"].values)
    slope = np.maximum(0.1, out["slope_pct"].values)
    forest = np.clip(out["forest_cover_pct"].values, 0.0, 100.0)

    out["log_drainage_area"] = np.log1p(area)
    out["relief_ratio"] = elev / np.sqrt(area)
    out["concentration_time_proxy"] = np.sqrt(area) / slope

    # 2. Precipitation volume & soil interaction
    precip = np.maximum(0.0, out["precipitation_mm"].values)
    soil = np.clip(out["soil_moisture"].values, 0.0, 1.0)

    precip_vol = precip * area
    out["precip_volume"] = precip_vol
    out["log_precip_volume"] = np.log1p(precip_vol)
    out["precip_soil"] = precip * soil
    out["precip_soil_volume"] = precip_vol * soil

    # 3. Temperature & snowmelt proxies
    t_max = out["temp_max_c"].values
    t_min = out["temp_min_c"].values
    t_mean = (t_max + t_min) / 2.0
    t_range = t_max - t_min

    out["temp_mean"] = t_mean
    out["temp_range"] = t_range
    out["freeze_flag"] = (t_min < 0.0).astype(float)
    # Melt potential: temperature above freezing following freezing conditions
    out["melt_potential"] = np.maximum(0.0, t_max) * (t_min <= 2.0).astype(float)

    # 4. Seasonality cyclical features
    dates = pd.to_datetime(out[DATE_COL])
    doy = dates.dt.day_of_year.values if hasattr(dates.dt, "day_of_year") else dates.dt.dayofyear.values
    month = dates.dt.month.values

    out["sin_doy"] = np.sin(2.0 * np.pi * doy / 365.25)
    out["cos_doy"] = np.cos(2.0 * np.pi * doy / 365.25)
    out["sin_month"] = np.sin(2.0 * np.pi * month / 12.0)
    out["cos_month"] = np.cos(2.0 * np.pi * month / 12.0)

    # 5. Station-wise temporal features (lags, rolling sums, EWMAs)
    # Ensure computation happens strictly within each station grouping
    def _station_time_features(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        p = g["precipitation_mm"]
        sm = g["soil_moisture"]
        tm = g["temp_mean"]

        # Lags (neutral fillna with 0 for rain, previous soil moisture)
        g["precip_lag1"] = p.shift(1).fillna(0.0)
        g["precip_lag2"] = p.shift(2).fillna(0.0)
        g["precip_lag3"] = p.shift(3).fillna(0.0)
        g["precip_lag5"] = p.shift(5).fillna(0.0)
        g["precip_lag7"] = p.shift(7).fillna(0.0)
        g["precip_lag14"] = p.shift(14).fillna(0.0)

        # Rolling sums & max
        g["precip_roll_sum_3d"] = p.rolling(3, min_periods=1).sum()
        g["precip_roll_sum_7d"] = p.rolling(7, min_periods=1).sum()
        g["precip_roll_sum_14d"] = p.rolling(14, min_periods=1).sum()
        g["precip_roll_sum_30d"] = p.rolling(30, min_periods=1).sum()
        g["precip_roll_sum_60d"] = p.rolling(60, min_periods=1).sum()
        g["precip_roll_max_7d"] = p.rolling(7, min_periods=1).max()
        g["precip_roll_max_14d"] = p.rolling(14, min_periods=1).max()

        # EWMAs representing hydrological baseflow/recession response
        g["precip_ewm_2d"] = p.ewm(span=2, min_periods=1).mean()
        g["precip_ewm_5d"] = p.ewm(span=5, min_periods=1).mean()
        g["precip_ewm_10d"] = p.ewm(span=10, min_periods=1).mean()
        g["precip_ewm_30d"] = p.ewm(span=30, min_periods=1).mean()

        # Soil moisture temporal dynamics
        sm_lag = sm.shift(1).bfill().fillna(0.2)
        g["soil_moisture_lag1"] = sm_lag
        g["soil_moisture_diff1"] = sm - sm_lag
        g["soil_moisture_roll_mean_7d"] = sm.rolling(7, min_periods=1).mean()
        g["soil_moisture_roll_mean_30d"] = sm.rolling(30, min_periods=1).mean()

        # Temperature rolling
        g["temp_mean_roll_7d"] = tm.rolling(7, min_periods=1).mean()
        g["temp_mean_roll_30d"] = tm.rolling(30, min_periods=1).mean()

        return g

    # Group by station while preserving index
    grouped_res = out.groupby(STATION_COL, group_keys=False, sort=False).apply(_station_time_features)
    out = grouped_res

    # 6. Higher-order interactions using rolling features
    eff_storm = out["precip_roll_sum_3d"].values * out["soil_moisture"].values
    out["effective_storm"] = eff_storm
    out["basin_effective_storm"] = eff_storm * area
    out["forest_attenuation"] = out["precip_roll_sum_7d"].values * (1.0 - forest / 100.0)

    # Ensure all FEATURE_COLS are present and finite
    for col in FEATURE_COLS:
        if col not in out.columns:
            raise KeyError(f"Feature column '{col}' missing from engineered output")
        # Fill any residual edge case NaNs safely with 0.0
        out[col] = out[col].fillna(0.0)

    # Restore exact input index and order
    out = out.reindex(orig_index)

    return out
