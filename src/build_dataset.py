"""Dataset builder for Phase 1 (Lane A).

CLI:
  python -m src.build_dataset --stations all

Builds real dataset across 25 verified stations from:
  - Source A: USGS Daily Discharge (parameter 00060, stat 00003)
  - Source B: USGS Site Info RDB (expanded)
  - Source C: Open-Meteo Archive API (5-year cached chunks, with backoff)
  - Source D: USGS StreamStats NSS Services (timeboxed, logged)

Enforces zero-fabrication:
  - Missing real values remain NaN.
  - Strict assertions on row counts, schemas, and date ranges.
"""

import argparse
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    CACHE_DIR,
    CORE_FEATURE_COLS,
    DATE_COL,
    DISCHARGE_QUALIFIERS_PATH,
    END_DATE,
    FULL_PARQUET_PATH,
    HOLDOUT_DIR,
    HOLDOUT_LABELS_PATH,
    PROCESSED_DIR,
    SAMPLE_SUBMISSION_PATH,
    START_DATE,
    STATION_COL,
    STATION_METADATA_COLUMNS,
    STATION_METADATA_PARQUET_PATH,
    STATION_METADATA_PATH,
    STATION_STATIC_EXTRA_PATH,
    STATIONS,
    TARGET_COL,
    TEST_COLUMNS,
    TEST_PARQUET_PATH,
    TEST_PATH,
    TEST_START_DATE,
    TRAIN_COLUMNS,
    TRAIN_END_DATE,
    TRAIN_PARQUET_PATH,
    TRAIN_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_dataset")


def parse_river_name(station_nm: str) -> str:
    """Extract river name from station name (text preceding ' at ', ' near ', ' a ', ' nr ')."""
    cleaned = station_nm.strip()
    match = re.split(r"\s+(?:at|near|nr|a|blw|above|near|nr\.)\s+", cleaned, maxsplit=1, flags=re.IGNORECASE)
    if match and match[0]:
        return match[0].strip()
    return cleaned


def fetch_with_retry(
    url: str,
    cache_path: Path,
    timeout: int = 30,
    max_retries: int = 3,
    backoff_sleep: int = 60,
) -> str:
    """Fetch URL with caching, timeout, and HTTP 429 backoff retry."""
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "FloodcastResearch/1.0 (Academic Hydrology)"})

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content = resp.read().decode("utf-8")
                cache_path.write_text(content, encoding="utf-8")
                return content
        except urllib.error.HTTPError as e:
            if e.code == 429:
                logger.warning(f"HTTP 429 (Rate Limit) on {url}. Backing off {backoff_sleep}s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(backoff_sleep)
            elif e.code == 404:
                logger.warning(f"HTTP 404 for {url}")
                raise
            else:
                logger.warning(f"HTTP {e.code} for {url} (Attempt {attempt+1}/{max_retries})")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2.0)
        except Exception as e:
            logger.warning(f"Network error on {url}: {e} (Attempt {attempt+1}/{max_retries})")
            if attempt == max_retries - 1:
                raise
            time.sleep(2.0)

    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts")


def fetch_source_b_metadata(site_no: str) -> Dict[str, Any]:
    """Fetch USGS Site RDB metadata (Source B)."""
    cache_file = CACHE_DIR / f"usgs_site_{site_no}.rdb"
    url = f"https://waterservices.usgs.gov/nwis/site/?format=rdb&sites={site_no}&siteOutput=expanded"
    content = fetch_with_retry(url, cache_file, timeout=20)

    lines = content.splitlines()
    data_lines = [l for l in lines if not l.startswith("#")]
    if len(data_lines) < 3:
        raise ValueError(f"Invalid Source B RDB response for {site_no}")

    headers = data_lines[0].split("\t")
    vals = data_lines[2].split("\t")
    row = dict(zip(headers, vals))

    station_nm = row.get("station_nm", "")
    lat = float(row.get("dec_lat_va", 0.0))
    lon = float(row.get("dec_long_va", 0.0))
    state = row.get("state_cd", "")
    # Fallback state if numeric code or absent
    if not state or state.isdigit():
        state = STATIONS.get(site_no, {}).get("state", "US")

    drain_area_str = row.get("drain_area_va", "")
    drain_area = float(drain_area_str) if drain_area_str not in ["", None] else np.nan

    alt_str = row.get("alt_va", "")
    alt_va = float(alt_str) if alt_str not in ["", None] else np.nan

    river = parse_river_name(station_nm)

    return {
        "station_id": str(site_no).zfill(8),
        "station_name": station_nm,
        "latitude": lat,
        "longitude": lon,
        "state": state,
        "river_name": river,
        "drain_area_va": drain_area,
        "alt_va": alt_va,
    }


def fetch_source_a_discharge(site_no: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch USGS Daily Discharge (Source A).

    Returns:
      (discharge_df, qualifiers_df)
    """
    cache_file = CACHE_DIR / f"usgs_dv_{site_no}.json"
    url = (
        f"https://waterservices.usgs.gov/nwis/dv/?format=json&sites={site_no}"
        f"&startDT={START_DATE}&endDT={END_DATE}&parameterCd=00060&statCd=00003"
    )
    content = fetch_with_retry(url, cache_file, timeout=40)
    data = json.loads(content)

    ts_list = data.get("value", {}).get("timeSeries", [])
    if not ts_list:
        logger.warning(f"No timeSeries returned for station {site_no}")
        return pd.DataFrame(columns=[STATION_COL, DATE_COL, TARGET_COL]), pd.DataFrame(columns=[STATION_COL, DATE_COL, "qualifier"])

    values = ts_list[0].get("values", [{}])[0].get("value", [])

    q_records = []
    qual_records = []

    for v in values:
        dt_str = v.get("dateTime", "")[:10]
        val_str = v.get("value", "")
        quals = v.get("qualifiers", [])
        qual_code = ",".join(quals) if quals else ""

        # Quality parsing: -999999, negative, or ice-affected missing -> NaN
        is_missing = (
            val_str in [None, "", "-999999", "Ice"]
            or "Ice" in quals
            or "Discontinued" in quals
        )
        if is_missing:
            val_cfs = np.nan
        else:
            try:
                num = float(val_str)
                val_cfs = num if num >= 0.0 else np.nan
            except ValueError:
                val_cfs = np.nan

        dt_norm = pd.to_datetime(dt_str).normalize()
        q_records.append({
            STATION_COL: str(site_no).zfill(8),
            DATE_COL: dt_norm,
            TARGET_COL: val_cfs,
        })
        qual_records.append({
            STATION_COL: str(site_no).zfill(8),
            DATE_COL: dt_norm,
            "qualifier": qual_code,
        })

    q_df = pd.DataFrame(q_records)
    qual_df = pd.DataFrame(qual_records)

    # De-duplicate if needed
    q_df = q_df.drop_duplicates(subset=[STATION_COL, DATE_COL])
    qual_df = qual_df.drop_duplicates(subset=[STATION_COL, DATE_COL])

    return q_df, qual_df


def fetch_source_c_weather(site_no: str, lat: float, lon: float) -> pd.DataFrame:
    """Fetch Open-Meteo weather in 5-year chunks (Source C)."""
    chunks = [
        ("2000-01-01", "2004-12-31"),
        ("2005-01-01", "2009-12-31"),
        ("2010-01-01", "2014-12-31"),
        ("2015-01-01", "2019-12-31"),
        ("2020-01-01", "2023-12-31"),
    ]

    chunk_dfs = []
    for s_date, e_date in chunks:
        cache_file = CACHE_DIR / f"openmeteo_{site_no}_{s_date}_{e_date}.json"
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?latitude={lat:.4f}&longitude={lon:.4f}"
            f"&start_date={s_date}&end_date={e_date}"
            f"&daily=precipitation_sum,temperature_2m_max,temperature_2m_min,soil_moisture_0_to_7cm_mean&timezone=UTC"
        )
        content = fetch_with_retry(url, cache_file, timeout=25, max_retries=4, backoff_sleep=60)
        data = json.loads(content)

        daily = data.get("daily", {})
        times = daily.get("time", [])
        precip = daily.get("precipitation_sum", [])
        tmax = daily.get("temperature_2m_max", [])
        tmin = daily.get("temperature_2m_min", [])
        soil = daily.get("soil_moisture_0_to_7cm_mean", [])

        df_chunk = pd.DataFrame({
            STATION_COL: str(site_no).zfill(8),
            DATE_COL: pd.to_datetime(times).normalize().astype("datetime64[ns]"),
            "precipitation_mm": pd.to_numeric(precip, errors="coerce").astype(float),
            "temp_max_c": pd.to_numeric(tmax, errors="coerce").astype(float),
            "temp_min_c": pd.to_numeric(tmin, errors="coerce").astype(float),
            "soil_moisture": pd.to_numeric(soil, errors="coerce").astype(float),
        })
        chunk_dfs.append(df_chunk)
        time.sleep(0.15)  # respectful pacing

    weather_df = pd.concat(chunk_dfs, ignore_index=True)
    weather_df = weather_df.drop_duplicates(subset=[STATION_COL, DATE_COL])
    return weather_df


def fetch_source_d_streamstats(site_no: str, state: str, meta: Dict[str, Any], timeout_s: int = 90) -> Dict[str, float]:
    """Query StreamStats NSS services for static catchment attributes (Source D).

    Time-boxed at 90s max per station.
    Returns:
      dict with drainage_area_sqmi, mean_elevation_ft, slope_pct, forest_cover_pct
    """
    cache_file = CACHE_DIR / f"streamstats_nss_{state}.json"
    url = f"https://streamstats.usgs.gov/nssservices/scenarios?rcode={state}"

    attrs: Dict[str, float] = {
        "drainage_area_sqmi": meta.get("drain_area_va", np.nan),
        "mean_elevation_ft": np.nan,
        "slope_pct": np.nan,
        "forest_cover_pct": np.nan,
    }

    try:
        t0 = time.time()
        content = fetch_with_retry(url, cache_file, timeout=min(timeout_s, 30))
        data = json.loads(content)

        # Scenarios parameters metadata inspection
        param_dict = {}
        for scen in data:
            for rr in scen.get("regressionRegions", []):
                for p in rr.get("parameters", []):
                    code = p.get("code")
                    unit = p.get("unit")
                    param_dict[code] = p

        logger.info(f"  StreamStats NSS ({state}) returned {len(param_dict)} parameter definitions in {time.time()-t0:.2f}s")
        # In NSS services, actual computed point values are unavailable without interactive GIS delineation.
        # As instructed: Anything unavailable stays NaN and is logged.
        # Do NOT substitute gauge altitude (alt_va) into mean_elevation_ft.
    except Exception as e:
        logger.warning(f"  StreamStats NSS ({state}) unavailable: {e}. Keeping static attributes as NaN.")

    return attrs


def build_all_dataset(stations_to_process: Optional[List[str]] = None) -> None:
    """Build the complete, verified, real dataset across all stations."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    station_list = stations_to_process or list(STATIONS.keys())
    logger.info(f"Starting Phase 1 Dataset Build for {len(station_list)} stations...")

    all_station_data = []
    metadata_records = []
    extra_static_records = []
    qualifier_dfs = []

    # Weather spine template: exact calendar days from 2000-01-01 to 2023-12-31
    spine_dates = pd.date_range(START_DATE, END_DATE, freq="D")
    expected_days_per_station = len(spine_dates)
    logger.info(f"Target date range: {START_DATE} to {END_DATE} ({expected_days_per_station:,} days per station)")

    total_start_time = time.time()

    for idx, site_no in enumerate(station_list):
        st_start = time.time()
        logger.info(f"[{idx+1}/{len(station_list)}] Processing station {site_no} ({STATIONS.get(site_no, {}).get('name')})...")

        # 1. Source B: Metadata
        meta = fetch_source_b_metadata(site_no)
        metadata_records.append({
            "station_id": meta["station_id"],
            "station_name": meta["station_name"],
            "latitude": meta["latitude"],
            "longitude": meta["longitude"],
            "state": meta["state"],
            "river_name": meta["river_name"],
        })
        extra_static_records.append({
            "station_id": meta["station_id"],
            "gauge_altitude_ft": meta["alt_va"],
        })

        # 2. Source D: StreamStats Static Catchment Attributes
        st_attrs = fetch_source_d_streamstats(site_no, meta["state"], meta)

        # 3. Source C: Weather spine (every date 2000-01-01..2023-12-31)
        weather_df = fetch_source_c_weather(site_no, meta["latitude"], meta["longitude"])

        # Enforce exact spine dates
        spine_df = pd.DataFrame({
            STATION_COL: str(site_no).zfill(8),
            DATE_COL: spine_dates.astype("datetime64[ns]"),
        })
        weather_spine = pd.merge(spine_df, weather_df, on=[STATION_COL, DATE_COL], how="left")
        assert len(weather_spine) == expected_days_per_station, f"Weather spine row count mismatch for {site_no}"

        # 4. Source A: Daily Discharge
        discharge_df, qual_df = fetch_source_a_discharge(site_no)
        qualifier_dfs.append(qual_df)

        # Left join discharge on (station_id, date)
        station_df = pd.merge(weather_spine, discharge_df, on=[STATION_COL, DATE_COL], how="left")
        assert len(station_df) == expected_days_per_station, f"Discharge merge changed row count for {site_no}!"

        # Attach static attributes
        for col in ["drainage_area_sqmi", "mean_elevation_ft", "slope_pct", "forest_cover_pct"]:
            station_df[col] = float(st_attrs.get(col, np.nan))
        assert len(station_df) == expected_days_per_station, f"Static merge changed row count for {site_no}!"

        all_station_data.append(station_df)
        logger.info(
            f"  -> Station {site_no} complete in {time.time()-st_start:.2f}s "
            f"| Valid Q days: {station_df[TARGET_COL].notna().sum():,}/{expected_days_per_station:,}"
        )

    # 5. Concatenate full dataset
    full_df = pd.concat(all_station_data, ignore_index=True)
    total_expected_rows = len(station_list) * expected_days_per_station
    assert len(full_df) == total_expected_rows, f"Full dataset rows ({len(full_df)}) != expected ({total_expected_rows})"

    # Save full.parquet
    full_df.to_parquet(FULL_PARQUET_PATH, index=False)
    logger.info(f"Saved full dataset ({len(full_df):,} rows) to: {FULL_PARQUET_PATH}")

    # 6. Build Train Table
    # dates < 2022-01-01, discharge included, rows with NaN discharge dropped
    train_mask = (full_df[DATE_COL] < pd.to_datetime(TEST_START_DATE)) & full_df[TARGET_COL].notna()
    train_df = full_df[train_mask].copy()[TRAIN_COLUMNS]

    # Save train.parquet and train.csv
    train_df.to_parquet(TRAIN_PARQUET_PATH, index=False)
    train_df.to_csv(TRAIN_PATH, index=False)
    logger.info(f"Saved train dataset ({len(train_df):,} rows) to: {TRAIN_PARQUET_PATH} & {TRAIN_PATH}")

    # 7. Build Test Features and Holdout Labels
    # dates >= 2022-01-01, only rows whose label exists in holdout labels
    test_period_mask = (full_df[DATE_COL] >= pd.to_datetime(TEST_START_DATE)) & full_df[TARGET_COL].notna()
    test_rows = full_df[test_period_mask].copy()

    # Holdout labels: station_id, date, discharge_cfs
    holdout_labels = test_rows[[STATION_COL, DATE_COL, TARGET_COL]].copy()
    holdout_labels.to_parquet(HOLDOUT_LABELS_PATH, index=False)
    logger.info(f"Saved holdout labels ({len(holdout_labels):,} rows) to: {HOLDOUT_LABELS_PATH}")

    # Test features: exact test columns minus discharge_cfs
    test_features = test_rows[TEST_COLUMNS].copy()
    test_features.to_parquet(TEST_PARQUET_PATH, index=False)
    test_features.to_csv(TEST_PATH, index=False)
    logger.info(f"Saved test features ({len(test_features):,} rows) to: {TEST_PARQUET_PATH} & {TEST_PATH}")

    # 8. Sample submission: station_id, date, discharge_cfs = 0.0
    sample_sub = test_features[[STATION_COL, DATE_COL]].copy()
    sample_sub[TARGET_COL] = 0.0
    sample_sub.to_csv(SAMPLE_SUBMISSION_PATH, index=False)
    logger.info(f"Saved sample submission ({len(sample_sub):,} rows) to: {SAMPLE_SUBMISSION_PATH}")

    # 9. Station metadata & static extra
    meta_df = pd.DataFrame(metadata_records)[STATION_METADATA_COLUMNS]
    meta_df.to_parquet(STATION_METADATA_PARQUET_PATH, index=False)
    meta_df.to_csv(STATION_METADATA_PATH, index=False)
    logger.info(f"Saved station metadata to: {STATION_METADATA_PARQUET_PATH} & {STATION_METADATA_PATH}")

    extra_static_df = pd.DataFrame(extra_static_records)
    extra_static_df.to_parquet(STATION_STATIC_EXTRA_PATH, index=False)
    logger.info(f"Saved station static extra to: {STATION_STATIC_EXTRA_PATH}")

    all_qualifiers_df = pd.concat(qualifier_dfs, ignore_index=True)
    all_qualifiers_df.to_parquet(DISCHARGE_QUALIFIERS_PATH, index=False)
    logger.info(f"Saved discharge qualifiers ({len(all_qualifiers_df):,} rows) to: {DISCHARGE_QUALIFIERS_PATH}")

    # Assertions
    assert not train_df.duplicated(subset=[STATION_COL, DATE_COL]).any(), "Duplicate keys in train!"
    assert not test_features.duplicated(subset=[STATION_COL, DATE_COL]).any(), "Duplicate keys in test!"
    assert pd.api.types.is_string_dtype(train_df[STATION_COL]), "station_id must be string in train!"
    assert pd.api.types.is_string_dtype(test_features[STATION_COL]), "station_id must be string in test!"
    assert (train_df[TARGET_COL] >= 0.0).all(), "Negative discharge in train!"
    assert TARGET_COL not in test_features.columns, "Discharge leaked into test!"
    assert train_df[DATE_COL].max() < test_features[DATE_COL].min(), "Train max date >= test min date!"
    assert len(test_features) == len(holdout_labels), "Test features and holdout row count mismatch!"
    assert (test_features[STATION_COL].values == holdout_labels[STATION_COL].values).all(), "Test and holdout station mismatch!"
    assert (test_features[DATE_COL].values == holdout_labels[DATE_COL].values).all(), "Test and holdout date mismatch!"

    logger.info(f"ALL DATASET BUILD ASSERTIONS PASSED! Total time: {time.time()-total_start_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Build real FLOODCAST dataset")
    parser.add_argument("--stations", type=str, default="all", help="'all' or comma-separated list of station IDs")
    args = parser.parse_args()

    if args.stations == "all":
        st_list = list(STATIONS.keys())
    else:
        st_list = [s.strip().zfill(8) for s in args.stations.split(",")]

    build_all_dataset(st_list)


if __name__ == "__main__":
    main()
