"""Shared constants and configuration for the FLOODCAST project.

Adheres strictly to project constraints:
- Random seed: 42
- Target: discharge_cfs
- station_id: string
- date: pandas datetime64[ns] normalized to midnight
- Metric weights: 40% overall RMSE, 30% top-10% RMSE, 20% MAE, 10% R2
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
STATION_METADATA_PATH = DATA_DIR / "station_metadata.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
PREDICTIONS_PATH = BASE_DIR / "predictions.csv"
EXPERIMENTS_PATH = BASE_DIR / "experiments" / "results.csv"

# Fallback root paths if files are kept in repository root
ROOT_TRAIN_PATH = BASE_DIR / "train.csv"
ROOT_TEST_PATH = BASE_DIR / "test.csv"
ROOT_STATION_METADATA_PATH = BASE_DIR / "station_metadata.csv"

# Global Random Seed
RANDOM_STATE = 42

# Column specifications
STATION_COL = "station_id"
DATE_COL = "date"
TARGET_COL = "discharge_cfs"

CORE_FEATURE_COLS = [
    "precipitation_mm",
    "temp_max_c",
    "temp_min_c",
    "soil_moisture",
    "drainage_area_sqmi",
    "mean_elevation_ft",
    "slope_pct",
    "forest_cover_pct",
]

TRAIN_COLUMNS = [
    "station_id",
    "date",
    "discharge_cfs",
    *CORE_FEATURE_COLS,
]

TEST_COLUMNS = [
    "station_id",
    "date",
    *CORE_FEATURE_COLS,
]

STATION_METADATA_COLUMNS = [
    "station_id",
    "station_name",
    "latitude",
    "longitude",
    "state",
    "river_name",
]

SUBMISSION_COLUMNS = [
    "station_id",
    "date",
    "discharge_cfs",
]

# Competition Scoring Weights
SCORING_WEIGHTS = {
    "rmse": 0.40,
    "top10_rmse": 0.30,
    "mae": 0.20,
    "r2": 0.10,
}

# Extreme event sample weighting defaults
DEFAULT_WEIGHT_Q = 0.90
DEFAULT_WEIGHT_BOOST = 5.0

# 5 Starter stations provided in problem statement
RECOMMENDED_STATIONS = {
    "01646500": {
        "name": "Potomac River near Washington, DC",
        "lat": 38.949,
        "lon": -77.127,
        "state": "DC",
        "river": "Potomac River",
        "drainage_area_sqmi": 11560.0,
        "mean_elevation_ft": 980.0,
        "slope_pct": 3.8,
        "forest_cover_pct": 52.4,
    },
    "07010000": {
        "name": "Mississippi River at St. Louis, MO",
        "lat": 38.625,
        "lon": -90.179,
        "state": "MO",
        "river": "Mississippi River",
        "drainage_area_sqmi": 697000.0,
        "mean_elevation_ft": 920.0,
        "slope_pct": 1.2,
        "forest_cover_pct": 32.1,
    },
    "08158000": {
        "name": "Colorado River at Austin, TX",
        "lat": 30.246,
        "lon": -97.699,
        "state": "TX",
        "river": "Colorado River",
        "drainage_area_sqmi": 38230.0,
        "mean_elevation_ft": 1450.0,
        "slope_pct": 2.1,
        "forest_cover_pct": 24.5,
    },
    "02087500": {
        "name": "Neuse River near Clayton, NC",
        "lat": 35.647,
        "lon": -78.472,
        "state": "NC",
        "river": "Neuse River",
        "drainage_area_sqmi": 1140.0,
        "mean_elevation_ft": 310.0,
        "slope_pct": 4.5,
        "forest_cover_pct": 58.7,
    },
    "11447650": {
        "name": "Sacramento River at Freeport, CA",
        "lat": 38.455,
        "lon": -121.501,
        "state": "CA",
        "river": "Sacramento River",
        "drainage_area_sqmi": 23500.0,
        "mean_elevation_ft": 2400.0,
        "slope_pct": 8.5,
        "forest_cover_pct": 41.2,
    },
}
