"""Catchment-attribute lookup used by predict.py and the website (models/catchment_attributes.parquet).

Runs without the data package: only the shipped attribute table is needed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import predict as P

TABLE = Path(__file__).resolve().parents[1] / "models" / "catchment_attributes.parquet"
pytestmark = pytest.mark.skipif(not TABLE.exists(), reason="models/catchment_attributes.parquet not present")


def _rows(ids, days=400):
    d = pd.date_range("2012-01-01", periods=days)
    out = []
    for g in ids:
        x = pd.DataFrame({P.ID: g, P.DATE: d, P.OBS: 1.0})
        for c in P.WEATHER:
            x[c] = 1.0
        out.append(x)
    return pd.concat(out, ignore_index=True)


def test_table_has_no_streamflow_and_all_static_columns():
    t = pd.read_parquet(TABLE)
    assert t[P.ID].is_unique and len(t) > 10_000
    assert set(P.STATIC_COLS) <= set(t.columns)
    assert not [c for c in t.columns if "streamflow" in c or c.startswith("q_")]


def test_exact_and_bare_ids_are_matched_unknown_ids_are_not():
    m = P.match_known(["camels_09210500", "09210500", "lamah_204636", "not_a_gauge", "123"], P.known_attributes())
    assert m["camels_09210500"] == "camels_09210500"
    assert m["09210500"] == "camels_09210500"
    assert m["lamah_204636"] == "lamah_204636"
    assert "not_a_gauge" not in m and "123" not in m


def test_lookup_fills_real_attributes_and_user_values_win():
    rows = _rows(["camels_09210500", "unknown_1"])
    st = P.derive_static(rows, None).set_index(P.ID)
    ref = P.known_attributes().set_index(P.ID).loc["camels_09210500"]
    assert st.attrs["matched"] == 1
    assert np.isclose(st.loc["camels_09210500", "ele_mt_sav"], ref["ele_mt_sav"])
    assert np.isnan(st.loc["unknown_1", "ele_mt_sav"])              # unknown gauge: nothing invented
    user = pd.DataFrame({P.ID: ["camels_09210500"], "ele_mt_sav": [1234.0]})
    st2 = P.derive_static(rows, user).set_index(P.ID)
    assert st2.loc["camels_09210500", "ele_mt_sav"] == 1234.0       # the user's own file always wins
    st3 = P.derive_static(rows, None, lookup=False).set_index(P.ID)
    assert np.isnan(st3.loc["camels_09210500", "ele_mt_sav"])       # lookup can be switched off
