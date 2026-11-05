"""Catchment attributes for every Caravan gauge, so forecasts on new files use real basin descriptors.

    python scripts/build_attribute_table.py --caravan ~/BasinPulse-data/caravan

Writes models/catchment_attributes.parquet: one row per gauge, the model's 28 static columns
(predict.STATIC_COLS). predict.py and the website look a gauge up here when the uploaded data has no attribute
file. These are fixed descriptions of the catchment (area, location, terrain, soil, land cover, long-term
climate), never streamflow: the BASINSHIFT task lists them as inputs (A(c), section 8.1).

Sources, in order of preference:
1. CAMELS-US (671) and LaMAH-CE (579): the exact attribute files our training and evaluation used
   (data/final_camels, data/transfer_lamah), so these gauges get bit-identical inputs.
2. Every other Caravan gauge: area and location from attributes_other, HydroATLAS descriptors from
   attributes_hydroatlas, climate indices from attributes_caravan. Six of the seven climate indices match our
   recipe (correlation 0.94-1.00 on the 671 CAMELS-US gauges); Caravan's seasonality index is defined
   differently, so p_seasonality is left empty there and computed from the uploaded weather instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predict import STATIC_COLS  # noqa: E402
from scripts.caravan.prepare_caravan import HYDROATLAS  # noqa: E402

ID = "gauge_id"
CLIMATE = {"p_mean": "p_mean", "pet_mean": "pet_mean_FAO_PM", "aridity": "aridity_FAO_PM", "frac_snow": "frac_snow",
           "high_prec_freq": "high_prec_freq", "low_prec_freq": "low_prec_freq"}


def from_caravan(src: Path) -> pd.DataFrame:
    name = src.name
    other = pd.read_csv(src / f"attributes_other_{name}.csv", dtype={ID: str})
    hyd = pd.read_csv(src / f"attributes_hydroatlas_{name}.csv", dtype={ID: str})
    clim = pd.read_csv(src / f"attributes_caravan_{name}.csv", dtype={ID: str})
    out = other[[ID, "area", "gauge_lat", "gauge_lon"]].rename(columns={"area": "area_km2"})
    out = out.merge(hyd[[ID] + [c for c in HYDROATLAS if c in hyd.columns]], on=ID, how="left")
    out = out.merge(clim[[ID] + list(CLIMATE.values())].rename(columns={v: k for k, v in CLIMATE.items()}),
                    on=ID, how="left")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--caravan", type=Path, default=Path.home() / "BasinPulse-data" / "caravan")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "catchment_attributes.parquet")
    a = ap.parse_args()

    exact = pd.concat([pd.read_parquet(ROOT / "data" / "final_camels" / "static.parquet"),
                       pd.read_parquet(ROOT / "data" / "transfer_lamah" / "static.parquet")], ignore_index=True)
    exact["source"] = "benchmark files"
    rest = []
    for src in sorted((a.caravan / "attributes").iterdir()):
        if src.is_dir():
            d = from_caravan(src)
            d["source"] = f"caravan {src.name}"
            rest.append(d[~d[ID].isin(exact[ID])])
    table = pd.concat([exact] + rest, ignore_index=True)
    for c in STATIC_COLS:
        if c not in table.columns:
            table[c] = float("nan")
    table = table[[ID] + STATIC_COLS + ["source"]].drop_duplicates(ID).sort_values(ID).reset_index(drop=True)
    table[STATIC_COLS] = table[STATIC_COLS].astype("float32")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(a.out, index=False)
    print(f"{len(table):,} gauges -> {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")
    print(table["source"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
