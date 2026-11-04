"""Fetch real USGS and Open-Meteo observations for FLOODCAST stations.

Usage:
  python scripts/fetch_real_data.py
"""

from pathlib import Path
import json
import urllib.parse
import urllib.request
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

STATIONS = {
    "01646500": {"name": "Potomac River near Wash, DC", "lat": 38.9498, "lon": -77.1276},
    "07010000": {"name": "Mississippi River at St. Louis, MO", "lat": 38.6256, "lon": -90.1793},
    "08158000": {"name": "Colorado River at Austin, TX", "lat": 30.2461, "lon": -97.6994},
    "02087500": {"name": "Neuse River near Clayton, NC", "lat": 35.6479, "lon": -78.4728},
    "11447650": {"name": "Sacramento River at Freeport, CA", "lat": 38.4557, "lon": -121.5011},
}


def fetch_usgs_discharge(site_no: str, start_date: str, end_date: str) -> pd.DataFrame:
    url = (
        f"https://waterservices.usgs.gov/nwis/dv/?format=json&sites={site_no}"
        f"&startDT={start_date}&endDT={end_date}&parameterCd=00060&statCd=00003"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    ts_list = data["value"]["timeSeries"]
    if not ts_list:
        return pd.DataFrame(columns=["date", "discharge_cfs"])

    values = ts_list[0]["values"][0]["value"]
    records = []
    for v in values:
        records.append({
            "date": pd.to_datetime(v["dateTime"]).normalize(),
            "discharge_cfs": float(v["value"]) if v["value"] not in ["-999999", None, ""] else None,
        })
    df = pd.DataFrame(records)
    df["station_id"] = str(site_no).zfill(8)
    return df


def fetch_openmeteo_weather(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    # Note: soil_moisture_0_to_7cm_mean is the correct Open-Meteo daily parameter
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=precipitation_sum,temperature_2m_max,temperature_2m_min,soil_moisture_0_to_7cm_mean&timezone=UTC"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    daily = data.get("daily", {})
    df = pd.DataFrame({
        "date": pd.to_datetime(daily.get("time", [])).normalize(),
        "precipitation_mm": daily.get("precipitation_sum", []),
        "temp_max_c": daily.get("temperature_2m_max", []),
        "temp_min_c": daily.get("temperature_2m_min", []),
        "soil_moisture": daily.get("soil_moisture_0_to_7cm_mean", []),
    })
    return df


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Testing real data fetch script for station 01646500 (2020-01-01..2020-01-03)...")
    q_df = fetch_usgs_discharge("01646500", "2020-01-01", "2020-01-03")
    w_df = fetch_openmeteo_weather(38.9498, -77.1276, "2020-01-01", "2020-01-03")
    merged = pd.merge(q_df, w_df, on="date", how="inner")
    print(merged)

