"""Data loading and ingestion module (Lane A).

Interfaces:
  - load_train() -> pd.DataFrame
  - load_test() -> pd.DataFrame
  - load_station_metadata() -> pd.DataFrame

Enforces hard rules:
  - station_id is ALWAYS a string (dtype={'station_id': str})
  - date is ALWAYS pandas datetime64[ns], normalized to midnight
  - Strict column validation
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    DATE_COL,
    RANDOM_STATE,
    RECOMMENDED_STATIONS,
    ROOT_STATION_METADATA_PATH,
    ROOT_TEST_PATH,
    ROOT_TRAIN_PATH,
    SAMPLE_SUBMISSION_PATH,
    STATION_COL,
    STATION_METADATA_COLUMNS,
    STATION_METADATA_PATH,
    TARGET_COL,
    TEST_COLUMNS,
    TEST_PATH,
    TRAIN_COLUMNS,
    TRAIN_PATH,
)


def _resolve_path(preferred: Path, fallback: Path) -> Path:
    """Return preferred path if it exists, otherwise fallback."""
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    return preferred


def _enforce_dtypes(df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
    """Enforce strict dtypes and datetime normalization."""
    df = df.copy()

    # Rule 3: station_id is ALWAYS string
    df[STATION_COL] = df[STATION_COL].astype(str).str.strip().str.zfill(8)

    # Rule 4: date is ALWAYS datetime64[ns] normalized to midnight
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).dt.normalize().astype("datetime64[ns]")

    # Float columns
    num_cols = [
        "precipitation_mm",
        "temp_max_c",
        "temp_min_c",
        "soil_moisture",
        "drainage_area_sqmi",
        "mean_elevation_ft",
        "slope_pct",
        "forest_cover_pct",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    if not is_test and TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").astype(float)

    return df


def generate_starter_dataset(
    start_train: str = "2018-01-01",
    end_train: str = "2022-12-31",
    start_test: str = "2023-01-01",
    end_test: str = "2023-12-31",
    save_to_disk: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates realistic hydrological starter datasets for the 5 stations.

    Used when train.csv / test.csv are not yet present on disk.
    Follows physics-informed hydrological simulation (rainfall-runoff, snowmelt,
    soil moisture memory, catchment size scaling).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    train_dates = pd.date_range(start_train, end_train, freq="D")
    test_dates = pd.date_range(start_test, end_test, freq="D")

    train_rows = []
    test_rows = []
    meta_rows = []

    for sid, info in RECOMMENDED_STATIONS.items():
        meta_rows.append(
            {
                "station_id": str(sid).zfill(8),
                "station_name": info["name"],
                "latitude": info["lat"],
                "longitude": info["lon"],
                "state": info["state"],
                "river_name": info["river"],
            }
        )

        all_dates = pd.date_range(start_train, end_test, freq="D")
        n = len(all_dates)

        doy = all_dates.dayofyear.values
        # Seasonal temperature cycle based on latitude/elevation
        base_temp = 16.0 - (info["lat"] - 30.0) * 0.8 - (info["mean_elevation_ft"] / 1000.0) * 2.0
        t_max = base_temp + 12.0 * np.sin(2 * np.pi * (doy - 105) / 365.25) + rng.normal(0, 3.5, n)
        t_min = t_max - rng.uniform(8.0, 16.0, n)

        # Precipitation: intermittent storm events
        rain_prob = 0.22 + 0.08 * np.cos(2 * np.pi * (doy - 50) / 365.25)
        rain_occur = rng.uniform(0, 1, n) < rain_prob
        precip = np.where(rain_occur, rng.exponential(scale=12.0, size=n), 0.0)
        # Extreme storm events
        extreme_storm = rng.uniform(0, 1, n) < 0.015
        precip = np.where(extreme_storm, precip + rng.exponential(scale=35.0, size=n), precip)
        precip = np.round(precip, 2)

        # Soil moisture: autoregressive process driven by rain and drying
        soil_moisture = np.zeros(n)
        sm = 0.25
        for i in range(n):
            infil = min(0.4, precip[i] / 50.0)
            et = max(0.005, (t_max[i] - 5.0) * 0.002) if t_max[i] > 5 else 0.002
            sm = np.clip(sm * 0.96 + infil - et, 0.05, 0.95)
            soil_moisture[i] = round(sm, 3)

        # Discharge simulation (hydrological routing)
        # Q ~ Area * (baseflow + runoff_coeff * precip * soil_moisture + snowmelt)
        area = info["drainage_area_sqmi"]
        baseflow = 0.15 * area * (0.8 + 0.4 * np.sin(2 * np.pi * (doy - 80) / 365.25))
        
        # Hydrological memory (lagged unit hydrograph)
        routed_flow = np.zeros(n)
        for i in range(n):
            runoff_eff = precip[i] * soil_moisture[i]
            # Snowmelt contribution if temp transitions above 0C
            melt = max(0.0, t_max[i]) * 0.5 if (t_min[i] < 2.0 and t_max[i] > 0.0) else 0.0
            daily_input = (runoff_eff + melt) * area * 0.6
            routed_flow[i] = daily_input

        # Convolution with exponential recession
        decay = np.exp(-1.0 / max(2.0, np.sqrt(area) / 15.0))
        discharge = np.zeros(n)
        current_q = baseflow[0]
        for i in range(n):
            current_q = current_q * decay + (1 - decay) * baseflow[i] + routed_flow[i]
            # Noise
            q_val = max(1.0, current_q * rng.lognormal(0, 0.12))
            discharge[i] = round(q_val, 1)

        station_df = pd.DataFrame(
            {
                "station_id": str(sid).zfill(8),
                "date": all_dates,
                "discharge_cfs": discharge,
                "precipitation_mm": precip,
                "temp_max_c": np.round(t_max, 1),
                "temp_min_c": np.round(t_min, 1),
                "soil_moisture": soil_moisture,
                "drainage_area_sqmi": float(area),
                "mean_elevation_ft": float(info["mean_elevation_ft"]),
                "slope_pct": float(info["slope_pct"]),
                "forest_cover_pct": float(info["forest_cover_pct"]),
            }
        )

        train_part = station_df[station_df["date"].isin(train_dates)].copy()
        test_part = station_df[station_df["date"].isin(test_dates)].copy()
        test_part = test_part.drop(columns=[TARGET_COL])

        train_rows.append(train_part)
        test_rows.append(test_part)

    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)
    meta_df = pd.DataFrame(meta_rows)

    train_df = _enforce_dtypes(train_df, is_test=False)
    test_df = _enforce_dtypes(test_df, is_test=True)
    meta_df[STATION_COL] = meta_df[STATION_COL].astype(str).str.zfill(8)

    if save_to_disk:
        train_df.to_csv(TRAIN_PATH, index=False)
        test_df.to_csv(TEST_PATH, index=False)
        meta_df.to_csv(STATION_METADATA_PATH, index=False)
        
        # sample_submission.csv with zeros as defined in contract
        sample_sub = test_df[[STATION_COL, DATE_COL]].copy()
        sample_sub[TARGET_COL] = 0.0
        sample_sub.to_csv(SAMPLE_SUBMISSION_PATH, index=False)

    return train_df, test_df, meta_df


def load_train() -> pd.DataFrame:
    """Load training data.

    cols: station_id, date, discharge_cfs, precipitation_mm,
          temp_max_c, temp_min_c, soil_moisture, drainage_area_sqmi,
          mean_elevation_ft, slope_pct, forest_cover_pct
    """
    path = _resolve_path(TRAIN_PATH, ROOT_TRAIN_PATH)
    if not path.exists():
        train_df, _, _ = generate_starter_dataset(save_to_disk=True)
        return train_df

    df = pd.read_csv(path, dtype={STATION_COL: str})
    df = _enforce_dtypes(df, is_test=False)

    missing = [c for c in TRAIN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"train.csv is missing required columns: {missing}")

    return df[TRAIN_COLUMNS]


def load_test() -> pd.DataFrame:
    """Load test data (same cols minus discharge_cfs)."""
    path = _resolve_path(TEST_PATH, ROOT_TEST_PATH)
    if not path.exists():
        _, test_df, _ = generate_starter_dataset(save_to_disk=True)
        return test_df

    df = pd.read_csv(path, dtype={STATION_COL: str})
    df = _enforce_dtypes(df, is_test=True)

    if TARGET_COL in df.columns:
        # Rule 2: NEVER use discharge from the test period anywhere
        df = df.drop(columns=[TARGET_COL])

    missing = [c for c in TEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"test.csv is missing required columns: {missing}")

    return df[TEST_COLUMNS]


def load_station_metadata() -> pd.DataFrame:
    """Load station metadata.

    cols: station_id, station_name, latitude, longitude, state, river_name
    """
    path = _resolve_path(STATION_METADATA_PATH, ROOT_STATION_METADATA_PATH)
    if not path.exists():
        _, _, meta_df = generate_starter_dataset(save_to_disk=True)
        return meta_df

    df = pd.read_csv(path, dtype={STATION_COL: str})
    df[STATION_COL] = df[STATION_COL].astype(str).str.strip().str.zfill(8)

    missing = [c for c in STATION_METADATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"station_metadata.csv is missing required columns: {missing}")

    return df[STATION_METADATA_COLUMNS]
