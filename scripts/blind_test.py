"""Blind test of the deployed prototype on rivers it has never seen, through the public website.

    python scripts/blind_test.py                                     # against https://basinpulse.onrender.com
    python scripts/blind_test.py --url http://localhost:8000         # against a local server

1. Rivers: the 3 Central European (LaMAH-CE) rivers whose June 2013 flood was the most severe relative to their own
   extreme-flow line (99th percentile of 1981-2008). Chosen by flood severity only, before any forecast was made.
   The submitted model was trained on US rivers only (models/basinpulse_final.json: training_packages final_camels),
   so these rivers, this country and this flood were never seen in training.
2. Inputs CSV (docs/blind_test/inputs.csv): what a forecaster has each day: the flow measured so far and the
   weather, 2008-10-01 to 2013-12-31. No forecast targets.
   Answer key (docs/blind_test/answers.csv): the real flow 1, 3 and 7 days after each 2013 day, kept apart.
3. The inputs CSV is uploaded to the website's live endpoint (/api/forecast), exactly like "Drop a CSV".
4. The returned forecasts for 2013 are scored against the answer key, next to persistence ("flow stays as today").

Writes docs/blind_test/RESULTS.md and docs/blind_test/forecasts.csv.
"""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transfer_lamah"
OUT = ROOT / "docs" / "blind_test"
ID, DATE = "gauge_id", "date"
HS = (1, 3, 7)
INPUT_FROM, SCORE_FROM, SCORE_TO = "2008-10-01", "2013-01-01", "2013-12-31"
FLOOD = ("2013-05-25", "2013-06-15")


def pick_rivers(test: pd.DataFrame, thr: pd.DataFrame, n: int = 3) -> list[str]:
    win = test[(test[DATE] >= FLOOD[0]) & (test[DATE] <= FLOOD[1])]
    peak = win.groupby(ID)["streamflow_obs"].max().rename("peak")
    span = test[(test[DATE] >= INPUT_FROM) & (test[DATE] <= SCORE_TO)]
    cov = span.groupby(ID)["streamflow_obs"].apply(lambda s: s.notna().mean()).rename("cov")
    j = thr.set_index(ID).join(peak).join(cov)
    j = j[j["cov"] > 0.99]
    return (j["peak"] / j["q_extreme"]).sort_values(ascending=False).index[:n].tolist()


def post_csv(url: str, csv: bytes) -> dict:
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"inputs.csv\"\r\n"
            f"Content-Type: text/csv\r\n\r\n").encode() + csv + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(url.rstrip("/") + "/api/forecast", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}",
                                          "User-Agent": "basinpulse-blind-test"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    return float(1 - np.sum((sim - obs) ** 2) / np.sum((obs - obs.mean()) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="https://basinpulse.onrender.com")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    test = pd.read_parquet(DATA / "test.parquet")
    labels = pd.read_parquet(DATA / "test_labels.parquet")
    thr = pd.read_csv(DATA / "thresholds.csv", dtype={ID: str})
    rivers = pick_rivers(test, thr)

    inputs = test[test[ID].isin(rivers) & (test[DATE] >= INPUT_FROM) & (test[DATE] <= SCORE_TO)]
    inputs = inputs.sort_values([ID, DATE]).reset_index(drop=True)
    assert not [c for c in inputs.columns if c.startswith("streamflow_h")], "inputs must not contain answers"
    inputs.to_csv(OUT / "inputs.csv", index=False, date_format="%Y-%m-%d")
    ans = labels[labels[ID].isin(rivers) & (labels[DATE] >= SCORE_FROM) & (labels[DATE] <= SCORE_TO)]
    ans = ans[[ID, DATE, "streamflow_h1", "streamflow_h3", "streamflow_h7"]].sort_values([ID, DATE])
    ans.to_csv(OUT / "answers.csv", index=False, date_format="%Y-%m-%d")
    print(f"inputs: {len(inputs):,} rows, rivers {rivers}; answers kept apart: {len(ans):,} rows", flush=True)

    t0 = time.time()
    res = post_csv(a.url, (OUT / "inputs.csv").read_bytes())
    took = time.time() - t0
    print(f"website answered in {took:.0f} s: {res['n_origins']:,} forecasts, valid={res['valid_submission']}", flush=True)

    rows = []
    for b in res["basins"]:
        for h in HS:
            s = b[f"h{h}"]
            d = pd.DataFrame({"target_date": pd.to_datetime(s["dates"]), "forecast": s["forecast"],
                              "lower": s["lower"], "upper": s["upper"], "p_high": s["p_high"]})
            d[ID], d["horizon"] = b["id"], h
            d[DATE] = d["target_date"] - pd.Timedelta(days=h)
            rows.append(d)
    fc = pd.concat(rows)
    fc = fc[(fc[DATE] >= SCORE_FROM) & (fc[DATE] <= SCORE_TO)]
    long = ans.melt(id_vars=[ID, DATE], var_name="horizon", value_name="actual")
    long["horizon"] = long["horizon"].str[-1].astype(int)
    today = inputs[[ID, DATE, "streamflow_obs"]].rename(columns={"streamflow_obs": "today"})
    df = fc.merge(long, on=[ID, DATE, "horizon"]).merge(today, on=[ID, DATE]).merge(thr[[ID, "q_high"]], on=ID)
    df = df.dropna(subset=["actual", "today", "forecast"])
    df[[ID, DATE, "horizon", "target_date", "forecast", "lower", "upper", "p_high", "actual", "today"]].to_csv(
        OUT / "forecasts.csv", index=False, date_format="%Y-%m-%d")

    names = {r: r for r in rivers}
    L = ["# Blind test of the live prototype on unseen rivers", "",
         f"Generated by `scripts/blind_test.py` against {a.url} ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}).", "",
         "- **Rivers:** the 3 Central European rivers (LaMAH-CE) with the most severe June 2013 flood relative to "
         "their own extreme-flow line, chosen by flood severity before any forecast was made. The submitted model "
         "was trained on US rivers only, so these rivers, this country and this flood were never seen.",
         "- **What the website got:** `inputs.csv`: measured flow so far and weather, 2008-10 to 2013-12. "
         "No answers. Uploaded to `/api/forecast` exactly like *Drop a CSV*.",
         f"- **Website:** {res['n_origins']:,} forecast days in {took:.0f} s; output "
         f"{'passes' if res['valid_submission'] else 'fails'} the BASINSHIFT format checks.",
         "- **Answer key:** `answers.csv`, the real flow 1, 3 and 7 days after each day of 2013, compared only "
         "afterwards. Persistence = \"the flow stays as it is today\".", "",
         "## Accuracy over 2013 (all 3 rivers)", "",
         "| | 1 day | 3 days | 7 days |", "|---|---|---|---|"]
    per = {h: df[df.horizon == h] for h in HS}
    L.append("| NSE, BasinPulse (1 = perfect) | " + " | ".join(
        f"**{np.median([nse(g.actual.values, g.forecast.values) for _, g in per[h].groupby(ID)]):.2f}**" for h in HS)
        + " |")
    L.append("| NSE, persistence | " + " | ".join(
        f"{np.median([nse(g.actual.values, g.today.values) for _, g in per[h].groupby(ID)]):.2f}" for h in HS) + " |")
    L.append("| Mean absolute error, BasinPulse (mm/day) | " + " | ".join(
        f"**{(per[h].forecast - per[h].actual).abs().mean():.2f}**" for h in HS) + " |")
    L.append("| Mean absolute error, persistence | " + " | ".join(
        f"{(per[h].today - per[h].actual).abs().mean():.2f}" for h in HS) + " |")
    L.append("| Real value inside the 10–90% range (target 80%) | " + " | ".join(
        f"**{((per[h].actual >= per[h].lower) & (per[h].actual <= per[h].upper)).mean():.0%}**" for h in HS) + " |")
    L.append("| Days scored | " + " | ".join(f"{len(per[h]):,}" for h in HS) + " |")
    L += ["", "NSE is the median over the 3 rivers.", "", "## The June 2013 flood peak, river by river", "",
          "| River | Peak day | Real peak flow | Forecast 1 day before (range) | P(high) 1 day before "
          "| Forecast 3 days before | P(high) 3 days before | Persistence 1 day before |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rivers:
        g = df[(df[ID] == r) & (df.target_date >= FLOOD[0]) & (df.target_date <= FLOOD[1])]
        if g.empty:
            continue
        pk = g.loc[g.actual.idxmax(), "target_date"]
        one = g[(g.horizon == 1) & (g.target_date == pk)].iloc[0]
        three = g[(g.horizon == 3) & (g.target_date == pk)]
        t3 = (f"{three.iloc[0].forecast:.1f} | {three.iloc[0].p_high:.0%}" if len(three) else "– | –")
        L.append(f"| {names[r]} | {pk:%Y-%m-%d} | {one.actual:.1f} | {one.forecast:.1f} ({one.lower:.1f}–{one.upper:.1f}) "
                 f"| {one.p_high:.0%} | {t3} | {one.today:.1f} |")
    L += ["", "Flows in mm/day (Caravan convention). P(high) is the website's probability that the flow exceeds "
          "the river's high-flow line (95th percentile of the flow in the uploaded file).", "",
          "Files: `inputs.csv` (what the website got; upload it yourself on the site), `answers.csv` (the real "
          "answers), `forecasts.csv` (every forecast next to its answer). Data: LaMAH-CE (Klingler et al. 2021, "
          "CC-BY-SA 4.0) via Caravan; real observations, redistributed here under the same licence."]
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
