"""BASINSHIFT Feature Engineering (Lane B).

Builds hydrologically informed, leak-free features for streamflow forecasting across
seen and unseen river basins (Caravan v1.6 / CAMELS-US).

Interfaces:
    INPUT_COLS: list[str]
    FEATURE_COLS: list[str]
    build_features(inputs: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame

Hard Rules Enforced:
    1. Causality: Every feature at date t uses only information available at or before date t.
    2. Zero target leakage: inputs must NEVER receive streamflow or streamflow_h* targets.
    3. Calendar alignment: Shifts and rolling calculations are computed on calendar dates per basin.
    4. Missing values: Missing inputs stay NaN (never forward-filled across long gaps).
    5. Efficiency: float32 outputs, minimal memory overhead (< 1 GB for 3.6M rows).
"""
from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd
from scipy.signal import lfilter

# Required input columns (forecast origin at end of day t)
INPUT_COLS: List[str] = [
    "gauge_id",
    "date",
    "streamflow_obs",
    "total_precipitation_sum",
    "temperature_2m_mean",
    "temperature_2m_min",
    "temperature_2m_max",
    "potential_evaporation_sum_FAO_PENMAN_MONTEITH",
    "snow_depth_water_equivalent_mean",
    "surface_net_solar_radiation_mean",
    "dewpoint_temperature_2m_mean",
    "volumetric_soil_water_layer_1_mean",
    "volumetric_soil_water_layer_3_mean",
]

# Static catchment attribute columns from static.parquet
STATIC_ATTR_COLS: List[str] = [
    "area_km2",
    "gauge_lat",
    "gauge_lon",
    "p_mean",
    "pet_mean",
    "aridity",
    "frac_snow",
    "high_prec_freq",
    "low_prec_freq",
    "p_seasonality",
    "ele_mt_sav",
    "slp_dg_sav",
    "for_pc_sse",
    "crp_pc_sse",
    "urb_pc_sse",
    "lka_pc_sse",
    "dor_pc_pva",
    "gwt_cm_sav",
    "cly_pc_sav",
    "snd_pc_sav",
    "slt_pc_sav",
    "soc_th_sav",
    "kar_pc_sse",
    "prm_pc_sse",
    "gla_pc_sse",
    "ppd_pk_sav",
    "swc_pc_syr",
    "snw_pc_syr",
]

# Explicit list of all engineered feature columns
FEATURE_COLS: List[str] = [
    # 1. Past flow dynamics from streamflow_obs
    "flow_obs",
    "flow_obs_lag_1d",
    "flow_obs_lag_2d",
    "flow_obs_lag_3d",
    "flow_obs_lag_7d",
    "flow_obs_roll_mean_3d",
    "flow_obs_roll_mean_7d",
    "flow_obs_roll_mean_30d",
    "flow_obs_diff_1d",
    "flow_obs_diff_3d",
    "log_flow_obs",
    "log_flow_obs_lag_1d",
    "log_flow_obs_lag_2d",
    "log_flow_obs_lag_3d",
    "log_flow_obs_lag_7d",
    "log_flow_obs_roll_mean_3d",
    "log_flow_obs_roll_mean_7d",
    "log_flow_obs_roll_mean_30d",
    "days_since_last_obs",
    "flow_norm_pmean",
    "log_flow_norm_pmean",
    # 2. Precipitation dynamics
    "precip_lag_1d",
    "precip_lag_2d",
    "precip_lag_3d",
    "precip_lag_4d",
    "precip_lag_5d",
    "precip_lag_6d",
    "precip_lag_7d",
    "precip_roll_sum_3d",
    "precip_roll_sum_7d",
    "precip_roll_sum_14d",
    "precip_roll_sum_30d",
    "precip_roll_sum_60d",
    "precip_roll_max_3d",
    "precip_roll_max_7d",
    "api_k90",
    # 3. Temperature, Snow, Soil, Evaporation
    "temp_roll_mean_3d",
    "temp_roll_mean_7d",
    "temp_roll_mean_30d",
    "temp_range",
    "swe",
    "swe_diff_1d",
    "swe_diff_3d",
    "swe_diff_7d",
    "soil_l1_roll_mean_3d",
    "soil_l1_roll_mean_7d",
    "soil_l1_roll_mean_30d",
    "soil_l3_roll_mean_3d",
    "soil_l3_roll_mean_7d",
    "soil_l3_roll_mean_30d",
    "pet_roll_sum_7d",
    "pet_roll_sum_30d",
    # 4. Seasonality
    "sin_doy",
    "cos_doy",
    # 5. Static Catchment Attributes + Basin Morphology
    "log_area_km2",
    *STATIC_ATTR_COLS,
]


def _build_basin_features(basin_df: pd.DataFrame, basin_static: pd.Series) -> pd.DataFrame:
    """Compute causal features on a calendar date spine for a single river basin."""
    indexed = basin_df.set_index("date").sort_index()
    if indexed.index.has_duplicates:
        indexed = indexed[~indexed.index.duplicated(keep="first")]

    min_date = indexed.index.min()
    max_date = indexed.index.max()
    spine_dates = pd.date_range(min_date, max_date, freq="D")
    daily = indexed.reindex(spine_dates)

    out = pd.DataFrame(index=spine_dates)

    # --------------------------------------------------------------------------
    # 1. Past flow from streamflow_obs
    # --------------------------------------------------------------------------
    obs = pd.to_numeric(daily["streamflow_obs"], errors="coerce").astype(np.float32)
    out["flow_obs"] = obs

    for lag in (1, 2, 3, 7):
        out[f"flow_obs_lag_{lag}d"] = obs.shift(lag).astype(np.float32)

    for w in (3, 7, 30):
        out[f"flow_obs_roll_mean_{w}d"] = (
            obs.rolling(w, min_periods=1).mean().astype(np.float32)
        )

    out["flow_obs_diff_1d"] = (obs - out["flow_obs_lag_1d"]).astype(np.float32)
    out["flow_obs_diff_3d"] = (obs - out["flow_obs_lag_3d"]).astype(np.float32)

    # Log1p transforms
    out["log_flow_obs"] = np.log1p(np.maximum(0.0, obs)).astype(np.float32)
    for lag in (1, 2, 3, 7):
        out[f"log_flow_obs_lag_{lag}d"] = np.log1p(
            np.maximum(0.0, out[f"flow_obs_lag_{lag}d"])
        ).astype(np.float32)
    for w in (3, 7, 30):
        out[f"log_flow_obs_roll_mean_{w}d"] = np.log1p(
            np.maximum(0.0, out[f"flow_obs_roll_mean_{w}d"])
        ).astype(np.float32)

    # Days since last non-NaN flow observation
    is_obs = obs.notna()
    last_obs_date = pd.Series(
        np.where(is_obs, spine_dates, pd.NaT), index=spine_dates
    ).ffill()
    out["days_since_last_obs"] = (
        (spine_dates - last_obs_date).dt.days.astype(np.float32)
    )

    # Flow normalised by basin mean precipitation (p_mean)
    p_mean = float(basin_static.get("p_mean", 1.0))
    p_mean_safe = max(p_mean, 0.01)
    flow_norm = (obs / p_mean_safe).astype(np.float32)
    out["flow_norm_pmean"] = flow_norm
    out["log_flow_norm_pmean"] = np.log1p(np.maximum(0.0, flow_norm)).astype(np.float32)

    # --------------------------------------------------------------------------
    # 2. Rain (total_precipitation_sum)
    # --------------------------------------------------------------------------
    precip = pd.to_numeric(daily["total_precipitation_sum"], errors="coerce").astype(np.float32)

    for lag in (1, 2, 3, 4, 5, 6, 7):
        out[f"precip_lag_{lag}d"] = precip.shift(lag).astype(np.float32)

    for w in (3, 7, 14, 30, 60):
        out[f"precip_roll_sum_{w}d"] = (
            precip.rolling(w, min_periods=1).sum().astype(np.float32)
        )

    for w in (3, 7):
        out[f"precip_roll_max_{w}d"] = (
            precip.rolling(w, min_periods=1).max().astype(np.float32)
        )

    # Antecedent Precipitation Index (k=0.90) via fast causal recursive filter
    precip_clean = precip.fillna(0.0).to_numpy(dtype=np.float32)
    out["api_k90"] = lfilter([1.0], [1.0, -0.9], precip_clean).astype(np.float32)

    # --------------------------------------------------------------------------
    # 3. Temperature, Snow, Soil, Evaporation
    # --------------------------------------------------------------------------
    t_mean = pd.to_numeric(daily["temperature_2m_mean"], errors="coerce").astype(np.float32)
    t_min = pd.to_numeric(daily["temperature_2m_min"], errors="coerce").astype(np.float32)
    t_max = pd.to_numeric(daily["temperature_2m_max"], errors="coerce").astype(np.float32)

    for w in (3, 7, 30):
        out[f"temp_roll_mean_{w}d"] = (
            t_mean.rolling(w, min_periods=1).mean().astype(np.float32)
        )
    out["temp_range"] = (t_max - t_min).astype(np.float32)

    # Snow water equivalent and snowmelt signals
    swe = pd.to_numeric(daily["snow_depth_water_equivalent_mean"], errors="coerce").astype(np.float32)
    out["swe"] = swe
    out["swe_diff_1d"] = (swe - swe.shift(1)).astype(np.float32)
    out["swe_diff_3d"] = (swe - swe.shift(3)).astype(np.float32)
    out["swe_diff_7d"] = (swe - swe.shift(7)).astype(np.float32)

    # Soil water layer 1 and layer 3
    soil1 = pd.to_numeric(daily["volumetric_soil_water_layer_1_mean"], errors="coerce").astype(np.float32)
    for w in (3, 7, 30):
        out[f"soil_l1_roll_mean_{w}d"] = (
            soil1.rolling(w, min_periods=1).mean().astype(np.float32)
        )

    soil3 = pd.to_numeric(daily["volumetric_soil_water_layer_3_mean"], errors="coerce").astype(np.float32)
    for w in (3, 7, 30):
        out[f"soil_l3_roll_mean_{w}d"] = (
            soil3.rolling(w, min_periods=1).mean().astype(np.float32)
        )

    # Evaporation (FAO Penman-Monteith)
    pet = pd.to_numeric(
        daily["potential_evaporation_sum_FAO_PENMAN_MONTEITH"], errors="coerce"
    ).astype(np.float32)
    for w in (7, 30):
        out[f"pet_roll_sum_{w}d"] = (
            pet.rolling(w, min_periods=1).sum().astype(np.float32)
        )

    # --------------------------------------------------------------------------
    # 4. Day-of-year seasonality (sin / cos)
    # --------------------------------------------------------------------------
    doy = spine_dates.day_of_year.values if hasattr(spine_dates, "day_of_year") else spine_dates.dayofyear.values
    out["sin_doy"] = np.sin(2.0 * np.pi * (doy - 1.0) / 365.25).astype(np.float32)
    out["cos_doy"] = np.cos(2.0 * np.pi * (doy - 1.0) / 365.25).astype(np.float32)

    # --------------------------------------------------------------------------
    # 5. Static Catchment Attributes + Log Area
    # --------------------------------------------------------------------------
    area = float(basin_static.get("area_km2", 1.0))
    out["log_area_km2"] = np.float32(np.log1p(max(0.0, area)))

    for col in STATIC_ATTR_COLS:
        val = float(basin_static.get(col, np.nan))
        out[col] = np.float32(val)

    # Reindex back to each original row's own date (rows may arrive in any order), then restore its index.
    # (Fix: reindexing by the date-sorted index and then assigning basin_df.index by position put features
    #  on the wrong rows whenever inputs were not already sorted by date.)
    res = out.reindex(pd.DatetimeIndex(basin_df["date"].to_numpy()))
    res.index = basin_df.index
    return res


def build_features(inputs: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """Build causal, leak-free features for the BASINSHIFT forecasting challenge.

    Parameters:
        inputs: pd.DataFrame containing INPUT_COLS only (train rows, or train+test concatenated).
                Must NEVER contain target columns (streamflow, streamflow_h1, etc.).
        static: pd.DataFrame containing static attributes per basin (static.parquet).

    Returns:
        pd.DataFrame with gauge_id, date, and all FEATURE_COLS as float32,
        preserving the exact row count and order of inputs.
    """
    # Hard Rule 2: NEVER accept targets in inputs
    forbidden = [
        c for c in inputs.columns if c == "streamflow" or c.startswith("streamflow_h")
    ]
    if forbidden:
        raise ValueError(
            f"inputs must NEVER receive target columns! Found forbidden columns: {forbidden}"
        )

    orig_index = inputs.index
    work = inputs.copy()
    work["gauge_id"] = work["gauge_id"].astype(str)
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()

    # Pre-index static attributes by gauge_id
    static_by_id = static.set_index("gauge_id")

    basin_frames = []
    for gid, grp in work.groupby("gauge_id", sort=False, dropna=False):
        b_static = (
            static_by_id.loc[gid]
            if gid in static_by_id.index
            else pd.Series(dtype=float)
        )
        basin_frames.append(_build_basin_features(grp, b_static))

    engineered = pd.concat(basin_frames).loc[orig_index]

    # Assemble output dataframe: gauge_id, date + FEATURE_COLS
    out = pd.DataFrame(index=orig_index)
    out["gauge_id"] = inputs["gauge_id"].astype(str).values
    out["date"] = pd.to_datetime(inputs["date"]).values

    for col in FEATURE_COLS:
        out[col] = engineered[col].to_numpy(dtype=np.float32)

    return out
