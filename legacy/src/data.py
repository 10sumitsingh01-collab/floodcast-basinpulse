"""Data loading and ingestion module (Lane A).

Interfaces:
  - load_train() -> pd.DataFrame
  - load_test() -> pd.DataFrame
  - load_station_metadata() -> pd.DataFrame

Enforces hard rules:
  - station_id is ALWAYS a string (dtype={'station_id': str})
  - date is ALWAYS pandas datetime64[ns], normalized to midnight
  - Strict column validation
  - NEVER invent or simulate data. Missing files raise FileNotFoundError.
"""

from pathlib import Path
import pandas as pd

from src.config import (
    DATE_COL,
    ROOT_STATION_METADATA_PATH,
    ROOT_TEST_PATH,
    ROOT_TRAIN_PATH,
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


def load_train() -> pd.DataFrame:
    """Load training data from real data file.

    cols: station_id, date, discharge_cfs, precipitation_mm,
          temp_max_c, temp_min_c, soil_moisture, drainage_area_sqmi,
          mean_elevation_ft, slope_pct, forest_cover_pct
    """
    path = _resolve_path(TRAIN_PATH, ROOT_TRAIN_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Real training data file not found at '{path}' (or '{ROOT_TRAIN_PATH}'). "
            f"Real data pending download."
        )

    df = pd.read_csv(path, dtype={STATION_COL: str})
    df = _enforce_dtypes(df, is_test=False)

    missing = [c for c in TRAIN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"train.csv is missing required columns: {missing}")

    return df[TRAIN_COLUMNS]


def load_test() -> pd.DataFrame:
    """Load test data from real data file (same cols minus discharge_cfs)."""
    path = _resolve_path(TEST_PATH, ROOT_TEST_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Real test data file not found at '{path}' (or '{ROOT_TEST_PATH}'). "
            f"Real data pending download."
        )

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
    """Load station metadata from real metadata file.

    cols: station_id, station_name, latitude, longitude, state, river_name
    """
    path = _resolve_path(STATION_METADATA_PATH, ROOT_STATION_METADATA_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Real station metadata file not found at '{path}' (or '{ROOT_STATION_METADATA_PATH}'). "
            f"Real data pending download."
        )

    df = pd.read_csv(path, dtype={STATION_COL: str})
    df[STATION_COL] = df[STATION_COL].astype(str).str.strip().str.zfill(8)

    missing = [c for c in STATION_METADATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"station_metadata.csv is missing required columns: {missing}")

    return df[STATION_METADATA_COLUMNS]
