"""Shared constants and configuration for the FLOODCAST project.

Phase 1: Real USGS and Open-Meteo dataset configuration.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
HOLDOUT_DIR = DATA_DIR / "holdout"

TRAIN_PARQUET_PATH = PROCESSED_DIR / "train.parquet"
TEST_PARQUET_PATH = PROCESSED_DIR / "test_features.parquet"
FULL_PARQUET_PATH = PROCESSED_DIR / "full.parquet"
HOLDOUT_LABELS_PATH = HOLDOUT_DIR / "holdout_labels.parquet"
STATION_METADATA_PARQUET_PATH = PROCESSED_DIR / "station_metadata.parquet"
STATION_STATIC_EXTRA_PATH = PROCESSED_DIR / "station_static_extra.parquet"
DISCHARGE_QUALIFIERS_PATH = PROCESSED_DIR / "discharge_qualifiers.parquet"

# CSV twins
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
STATION_METADATA_PATH = PROCESSED_DIR / "station_metadata.csv"
SAMPLE_SUBMISSION_PATH = PROCESSED_DIR / "sample_submission.csv"

# Legacy / root fallbacks
ROOT_TRAIN_PATH = DATA_DIR / "train.csv"
ROOT_TEST_PATH = DATA_DIR / "test.csv"
ROOT_STATION_METADATA_PATH = DATA_DIR / "station_metadata.csv"

PREDICTIONS_PATH = BASE_DIR / "predictions.csv"
EXPERIMENTS_PATH = BASE_DIR / "experiments" / "results.csv"

# Dataset Version
DATA_VERSION = "v2"

# Global Random Seed
RANDOM_STATE = 42

# Date splits
START_DATE = "2000-01-01"
END_DATE = "2023-12-31"
TRAIN_END_DATE = "2021-12-31"
TEST_START_DATE = "2022-01-01"

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
    "elev_range_ft",
    "elev_std_ft",
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

# 14 Verified Active USGS Stations with 100% complete records across train & holdout
STATIONS = {
    # 4 Remaining Mandatory Stations
    "01646500": {"name": "Potomac River near Washington, DC", "region": "Mid-Atlantic", "state": "DC"},
    "07010000": {"name": "Mississippi River at St. Louis, MO", "region": "Midwest", "state": "MO"},
    "08158000": {"name": "Colorado River at Austin, TX", "region": "Southwest", "state": "TX"},
    "02087500": {"name": "Neuse River near Clayton, NC", "region": "Southeast", "state": "NC"},
    # New England / Northeast
    "01092000": {"name": "Merrimack River near Goffs Falls, NH", "region": "New England", "state": "NH"},
    "01184000": {"name": "Connecticut River at Thompsonville, CT", "region": "New England", "state": "CT"},
    "01357500": {"name": "Mohawk River at Cohoes, NY", "region": "Northeast", "state": "NY"},
    "01438500": {"name": "Delaware River at Montague, NJ", "region": "Northeast", "state": "NJ"},
    "01570500": {"name": "Susquehanna River at Harrisburg, PA", "region": "Northeast", "state": "PA"},
    # Flashy Small Basins (< 500 sq mi)
    "01644000": {"name": "Goose Creek near Leesburg, VA", "region": "Mid-Atlantic", "state": "VA"},
    "02085000": {"name": "Eno River at Hillsborough, NC", "region": "Southeast", "state": "NC"},
    "08378500": {"name": "Pecos River near Pecos, NM", "region": "Southwest", "state": "NM"},
    "11264500": {"name": "Merced River at Happy Isles Bridge, CA", "region": "Sierra", "state": "CA"},
    "12119000": {"name": "Cedar River at Renton, WA", "region": "Pacific NW", "state": "WA"},
}

# 12 Verified Candidate Stations for Phase 2+ Expansion (USGS DV & NLDI verified; weather pending API reset)
EXTRA_STATION_CANDIDATES = [
    "11266500",  # Merced River at Pohono Bridge, CA (Sierra, 321 sq mi, Sacramento replacement)
    "14048000",  # John Day River at McDonald Ferry, OR (Pacific NW, 7,580 sq mi, undammed)
    "09058000",  # Colorado River near Kremmling, CO (Rocky Mountain snowmelt, 2,379 sq mi)
    "06191500",  # Yellowstone River at Corwin Springs, MT (Northern Rockies, 2,615 sq mi)
    "06054500",  # Missouri River at Toston, MT (Missouri headwaters, 14,641 sq mi)
    "08155200",  # Barton Creek at SH 71 nr Oak Hill, TX (Southwest flashy, 89.7 sq mi)
    "01643000",  # Monocacy River at Jug Bridge, MD (Mid-Atlantic, 817 sq mi)
    "02320500",  # Suwannee River at Branford, FL (Humid Southeast, 7,880 sq mi)
    "02105500",  # Cape Fear River at Huske Lock, NC (Southeast, 4,852 sq mi)
    "03335500",  # Wabash River at Lafayette, IN (Midwest plains, 7,267 sq mi)
    "06892350",  # Kansas River at DeSoto, KS (Midwest mega-basin, 59,756 sq mi)
    "05378500",  # Mississippi River at Winona, MN (Upper Mississippi mega-basin, 59,200 sq mi)
]
