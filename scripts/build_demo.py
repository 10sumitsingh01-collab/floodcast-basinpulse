"""Build the offline interactive demo: docs/demo/index.html (single file, no internet, no installs).

    python scripts/build_demo.py --run outputs/lgbm_final --baseline outputs/persistence_baseline

For the 60 basins never seen in training (test period, geo_temporal): a map coloured by 3-day skill;
click a basin to see the forecast vs what happened around its largest flood, with the 10-90% range,
P(high flow) and persistence, for 1 / 3 / 7 days ahead. Built from the logged test run only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import evaluate  # noqa: E402

ID, DATE = "gauge_id", "date"


def r(x, n=3):
    return None if not np.isfinite(x) else round(float(x), n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=ROOT / "outputs" / "lgbm_final")
    ap.add_argument("--baseline", type=Path, default=ROOT / "outputs" / "persistence_baseline")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "caravan")
    ap.add_argument("--caravan", type=Path, default=Path.home() / "BasinPulse-data" / "caravan")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "demo" / "index.html")
    ap.add_argument("--days", type=int, default=150)
    ap.add_argument("--artifact-out", type=Path, default=None, help="also write the page body without <html>/<head>")
    a = ap.parse_args()

    labels = pd.read_parquet(a.data / "test_labels.parquet")
    lab = labels[labels.test_set == "geo_temporal"]
    thr = pd.read_csv(a.data / "thresholds.csv", dtype={ID: str}).set_index(ID)
    st = pd.read_parquet(a.data / "static.parquet").set_index(ID)
    names = pd.read_csv(a.caravan / "attributes" / "camels" / "attributes_other_camels.csv",
                        dtype={ID: str}).set_index(ID)["gauge_name"]
    p = pd.read_parquet(a.run / "predictions.parquet")
    b = pd.read_parquet(a.baseline / "predictions.parquet", columns=[ID, DATE, "horizon", "predicted_discharge"])
    basins = sorted(lab[ID].unique())
    p, b = p[p[ID].isin(basins)], b[b[ID].isin(basins)]

    # per-basin skill (3-day NSE, model and persistence)
    j = evaluate.join(p, lab, thr.reset_index())
    jb = evaluate.join(b.assign(lower=0.0, upper=0.0, p_high=0.0, p_extreme=0.0), lab, thr.reset_index())
    nse = {h: evaluate.basin_metrics(j[j.horizon == h])["nse"] for h in (1, 3, 7)}
    nse_b = {h: evaluate.basin_metrics(jb[jb.horizon == h])["nse"] for h in (1, 3, 7)}

    out = []
    for g in basins:
        # window around the basin's largest observed flow in the test period (flow on t+1 = h1 label at t)
        lg = lab[lab[ID] == g].sort_values(DATE)
        peak_t = lg.loc[lg.streamflow_h1.idxmax(), DATE] + pd.Timedelta(days=1)
        lo_d, hi_d = peak_t - pd.Timedelta(days=int(a.days * 0.7)), peak_t + pd.Timedelta(days=int(a.days * 0.3))
        obs = lg.assign(day=lg[DATE] + pd.Timedelta(days=1)).set_index("day")["streamflow_h1"]
        days = pd.date_range(lo_d, hi_d, freq="D")
        series = {"dates": [d.strftime("%Y-%m-%d") for d in days],
                  "obs": [r(v) for v in obs.reindex(days).to_numpy(float)]}
        for h in (1, 3, 7):
            ph = p[(p[ID] == g) & (p.horizon == h)].assign(day=lambda x: x[DATE] + pd.Timedelta(days=h)).set_index("day")
            bh = b[(b[ID] == g) & (b.horizon == h)].assign(day=lambda x: x[DATE] + pd.Timedelta(days=h)).set_index("day")
            series[f"h{h}"] = {k: [r(v) for v in ph[k].reindex(days).to_numpy(float)]
                               for k in ("predicted_discharge", "lower", "upper", "p_high")}
            series[f"h{h}"]["persistence"] = [r(v) for v in bh["predicted_discharge"].reindex(days).to_numpy(float)]
        out.append({"id": g, "name": str(names.get(g, g)), "lat": r(st.loc[g, "gauge_lat"], 3),
                    "lon": r(st.loc[g, "gauge_lon"], 3), "area": r(st.loc[g, "area_km2"], 0),
                    "q_high": r(thr.loc[g, "q_high"]), "q_extreme": r(thr.loc[g, "q_extreme"]),
                    "nse": {str(h): r(nse[h].get(g, np.nan), 2) for h in (1, 3, 7)},
                    "nse_persistence": {str(h): r(nse_b[h].get(g, np.nan), 2) for h in (1, 3, 7)},
                    "series": series})

    m = pd.read_csv(a.run / "metrics.csv")
    m = m[m.test_set == "geo_temporal"].set_index(["metric", "horizon"])["value"]
    mb = pd.read_csv(a.baseline / "metrics.csv")
    mb = mb[mb.test_set == "geo_temporal"].set_index(["metric", "horizon"])["value"]
    summary = {k: {str(h): [r(m.get((k, h), np.nan)), r(mb.get((k, h), np.nan))] for h in (1, 3, 7)}
               for k in ("nse_median", "auprc_high", "auprc_extreme", "interval_coverage")}
    payload = json.dumps({"basins": out, "summary": summary}, separators=(",", ":"))
    fragment = TEMPLATE.replace("__DATA__", payload)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(FRAME.replace("__FRAGMENT__", fragment), encoding="utf-8")   # offline, open in any browser
    if a.artifact_out:
        a.artifact_out.parent.mkdir(parents=True, exist_ok=True)
        a.artifact_out.write_text(fragment, encoding="utf-8")                       # page body for a hosted page
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB, {len(out)} basins)")


FRAME = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
</head><body>
__FRAGMENT__
</body></html>
"""

TEMPLATE = r"""<title>BasinPulse Forecast Explorer</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
/* Layout: summary tiles, then map of unseen rivers beside the forecast chart of the selected river */
:root{--bg:#f7f8f6;--fg:#1b2329;--muted:#57636d;--line:#d7dcd8;--card:#ffffff;--blue:#2b67ad;--band:rgba(43,103,173,.18);
--grey:#878f96;--red:#c0392f;--orange:#d9740f;--good:#2a7a4c;--bad:#b0412c}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121619;--fg:#e6e8e6;--muted:#9ba6ae;--line:#2c3439;--card:#192024;
--blue:#6ea3e6;--band:rgba(110,163,230,.22);--grey:#8a9097;--red:#ef6b5f;--orange:#f0a050;--good:#5cbf86;--bad:#e07a65;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121619;--fg:#e6e8e6;--muted:#9ba6ae;--line:#2c3439;--card:#192024;
--blue:#6ea3e6;--band:rgba(110,163,230,.22);--grey:#8a9097;--red:#ef6b5f;--orange:#f0a050;--good:#5cbf86;--bad:#e07a65;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding-block:20px 40px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
h1,h2{text-wrap:balance}.kpi b{font-variant-numeric:tabular-nums}.grid>*{min-width:0}
button:focus-visible,select:focus-visible,.dot:focus-visible{outline:2px solid var(--blue);outline-offset:2px}h1{font-size:22px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:16px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}.kpi b{font-size:20px;display:block}
.kpi span{color:var(--muted);font-size:12.5px}.grid{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,7fr);gap:14px}
@media(max-width:860px){.grid{grid-template-columns:1fr}}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h2{font-size:15px;margin:0 0 8px}svg{width:100%;height:auto;display:block}.dot{cursor:pointer;stroke:var(--card);stroke-width:1}
.dot.sel{stroke:var(--fg);stroke-width:2.5}.legend{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:12.5px;margin-top:6px}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:middle;margin-right:4px}
.tabs{display:flex;gap:6px;margin:6px 0 8px}.tabs button{border:1px solid var(--line);background:var(--bg);color:var(--fg);
border-radius:6px;padding:4px 10px;cursor:pointer;font:inherit;font-size:13px}.tabs button.on{background:var(--blue);color:#fff;border-color:var(--blue)}
.meta{color:var(--muted);font-size:13px;margin-bottom:4px}.note{color:var(--muted);font-size:12.5px;margin-top:10px}
select{font:inherit;padding:4px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--fg);max-width:100%}
.axis text{fill:var(--muted);font-size:10px}.axis line,.axis path{stroke:var(--line)}
</style></head><body><main>
<h1>BasinPulse: flood forecasts on rivers the model never saw</h1>
<p class="sub">60 US rivers held out of training, test years 2008–2014. Real observed flow (CAMELS-US via Caravan); the model forecasts 1, 3 and 7 days ahead.</p>
<div class="kpis" id="kpis"></div>
<div class="grid">
 <div class="card"><h2>Unseen rivers: 3-day forecast skill (NSE)</h2><svg id="map" viewBox="0 0 600 330" role="img" aria-label="Map of held-out basins"></svg>
 <div class="legend"><span><i class="sw" style="background:var(--good)"></i>NSE ≥ 0.4</span><span><i class="sw" style="background:var(--orange)"></i>0 – 0.4</span><span><i class="sw" style="background:var(--bad)"></i>&lt; 0</span><span>Click a river</span></div>
 <p class="note">Persistence ("tomorrow = today") 3-day NSE is below 0 on most of these rivers.</p></div>
 <div class="card"><h2 id="title">River</h2><div class="meta" id="meta"></div>
 <select id="pick" aria-label="Choose a river"></select>
 <div class="tabs" id="tabs"></div><svg id="chart" viewBox="0 0 700 330" role="img" aria-label="Forecast chart"></svg>
 <div class="legend"><span><i class="sw" style="background:var(--fg)"></i>observed</span><span><i class="sw" style="background:var(--blue)"></i>BasinPulse forecast</span><span><i class="sw" style="background:var(--band)"></i>10–90% range</span><span><i class="sw" style="background:var(--grey)"></i>persistence</span><span><i class="sw" style="background:var(--orange)"></i>P(high flow)</span><span><i class="sw" style="background:var(--red)"></i>high-flow threshold</span></div>
 <p class="note">Each point is the forecast made h days earlier for that day. Window: the river's largest flood in the test period.</p></div>
</div>
<p class="note">Research benchmark, not a flood-warning service. Data: Caravan v1.6 / CAMELS-US (CC-BY-4.0).</p>
</main>
<script>
const D=__DATA__;let sel=null,H=1;const css=k=>getComputedStyle(document.documentElement).getPropertyValue(k).trim();
const S=D.summary;const kp=[["nse_median","Median NSE"],["auprc_high","High-flow AUPRC"],["auprc_extreme","Extreme-flow AUPRC"],["interval_coverage","Range coverage (target 0.80)"]];
document.getElementById("kpis").innerHTML=kp.map(([k,l])=>`<div class="kpi"><b>${S[k]["3"][0].toFixed(2)}</b><span>${l}, 3 days ahead · persistence ${S[k]["3"][1].toFixed(2)}</span></div>`).join("");
const col=v=>v==null?css("--grey"):v>=0.4?css("--good"):v>=0?css("--orange"):css("--bad");
function drawMap(){const s=document.getElementById("map"),W=600,Hh=330,x=lo=>(lo+125)/(125-66)*(W-30)+15,y=la=>(50-la)/(50-24)*(Hh-30)+15;
let g=`<rect x="0" y="0" width="${W}" height="${Hh}" fill="none"/>`;for(let la=25;la<=50;la+=5)g+=`<line x1="10" x2="${W-10}" y1="${y(la)}" y2="${y(la)}" stroke="${css("--line")}" stroke-dasharray="2 4"/>`;
for(let lo=-120;lo<=-70;lo+=10)g+=`<line y1="10" y2="${Hh-10}" x1="${x(lo)}" x2="${x(lo)}" stroke="${css("--line")}" stroke-dasharray="2 4"/>`;
D.basins.forEach((b,i)=>{g+=`<circle class="dot${sel===i?" sel":""}" cx="${x(b.lon)}" cy="${y(b.lat)}" r="6.5" fill="${col(b.nse["3"])}" data-i="${i}"><title>${b.name}: 3-day NSE ${b.nse["3"]} (persistence ${b.nse_persistence["3"]})</title></circle>`});
s.innerHTML=g;s.querySelectorAll(".dot").forEach(c=>c.onclick=()=>pick(+c.dataset.i))}
function drawChart(){const b=D.basins[sel],s=b.series,f=s["h"+H],n=s.dates.length,W=700,Hh=330,L=44,R=40,T=12,B=28;
const vals=[...s.obs,...f.upper,...f.persistence,b.q_high].filter(v=>v!=null);const ymax=Math.max(...vals)*1.05||1;
const X=i=>L+i/(n-1)*(W-L-R),Y=v=>T+(1-v/ymax)*(Hh-T-B),P=v=>T+(1-v)*(Hh-T-B);
const path=(a,fy)=>{let d="",on=false;a.forEach((v,i)=>{if(v==null){on=false;return}d+=(on?"L":"M")+X(i).toFixed(1)+","+fy(v).toFixed(1);on=true});return d};
let band="";const up=[],dn=[];f.upper.forEach((v,i)=>{if(v!=null&&f.lower[i]!=null){up.push([X(i),Y(v)]);dn.push([X(i),Y(f.lower[i])])}});
if(up.length)band=`<path d="M${up.map(p=>p.join(",")).join("L")}L${dn.reverse().map(p=>p.join(",")).join("L")}Z" fill="${css("--band")}"/>`;
let ax="";for(let k=0;k<=4;k++){const v=ymax*k/4;ax+=`<g class="axis"><line x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}" stroke-dasharray="2 4"/><text x="${L-6}" y="${Y(v)+3}" text-anchor="end">${v.toFixed(v<10?1:0)}</text></g>`}
for(let k=0;k<=2;k++)ax+=`<g class="axis"><text x="${W-R+6}" y="${P(k/2)+3}">${(k/2).toFixed(1)}</text></g>`;
[0,Math.floor(n/2),n-1].forEach(i=>ax+=`<g class="axis"><text x="${X(i)}" y="${Hh-8}" text-anchor="middle">${s.dates[i]}</text></g>`);
const svg=document.getElementById("chart");svg.innerHTML=ax+band+
`<line x1="${L}" x2="${W-R}" y1="${Y(b.q_high)}" y2="${Y(b.q_high)}" stroke="${css("--red")}" stroke-dasharray="5 4"/>`+
`<path d="${path(f.p_high,P)}" fill="none" stroke="${css("--orange")}" stroke-width="1.2" opacity=".85"/>`+
`<path d="${path(f.persistence,Y)}" fill="none" stroke="${css("--grey")}" stroke-width="1.3" stroke-dasharray="4 3"/>`+
`<path d="${path(s.obs,Y)}" fill="none" stroke="${css("--fg")}" stroke-width="1.8"/>`+
`<path d="${path(f.predicted_discharge,Y)}" fill="none" stroke="${css("--blue")}" stroke-width="1.8"/>`+
`<text x="${L}" y="${T+2}" class="axis" fill="${css("--muted")}" font-size="10">flow, mm/day</text><text x="${W-R}" y="${T+2}" text-anchor="end" fill="${css("--muted")}" font-size="10">P(high)</text>`}
function pick(i){sel=i;const b=D.basins[i];document.getElementById("title").textContent=b.name;
document.getElementById("meta").textContent=`${b.id} · ${b.area.toLocaleString()} km² · NSE 1/3/7 days: ${b.nse["1"]} / ${b.nse["3"]} / ${b.nse["7"]} (persistence ${b.nse_persistence["1"]} / ${b.nse_persistence["3"]} / ${b.nse_persistence["7"]})`;
document.getElementById("pick").value=i;drawMap();drawChart()}
document.getElementById("pick").innerHTML=D.basins.map((b,i)=>`<option value="${i}">${b.name}</option>`).join("");
document.getElementById("pick").onchange=e=>pick(+e.target.value);
document.getElementById("tabs").innerHTML=[1,3,7].map(h=>`<button data-h="${h}" class="${h===1?"on":""}">${h} day${h>1?"s":""} ahead</button>`).join("");
document.querySelectorAll("#tabs button").forEach(bt=>bt.onclick=()=>{H=+bt.dataset.h;document.querySelectorAll("#tabs button").forEach(x=>x.classList.toggle("on",x===bt));drawChart()});
let best=0;D.basins.forEach((b,i)=>{if((b.nse["3"]??-9)>(D.basins[best].nse["3"]??-9))best=i});pick(best);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change",()=>{drawMap();drawChart()});
</script>
"""

if __name__ == "__main__":
    main()
