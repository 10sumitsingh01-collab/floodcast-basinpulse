/* BasinPulse site. Scroll drives the film (frame scrubbing), chapter titles scramble, statements type in,
   then the film shrinks into a card over black. Below the film, every number comes from /api (logged runs). */
"use strict";
const $ = (s) => document.querySelector(s);
const N_FRAMES = 232;
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const css = (k) => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
const fmt = (v, d = 2) => (v == null || !isFinite(v) ? "–" : Number(v).toFixed(d));

/* ---------------- film: preload frames, draw cover-fit ------------------------------------------------ */
const canvas = $("#frames"), ctx = canvas.getContext("2d");
const frames = new Array(N_FRAMES);
let lastDrawn = -1;
function src(i) { return `/static/frames/f_${String(i + 1).padStart(4, "0")}.jpg`; }
function loadFrames() {
  const order = [];
  for (let step = 16; step >= 1; step = Math.floor(step / 2)) for (let i = 0; i < N_FRAMES; i += step) if (!order.includes(i)) order.push(i);
  let k = 0;
  const next = () => { if (k >= order.length) return; const i = order[k++]; const im = new Image();
    im.src = src(i);
    (im.decode ? im.decode() : new Promise((ok, no) => { im.onload = ok; im.onerror = no; }))
      .then(() => { frames[i] = im; if (lastDrawn < 0 || Math.abs(i - want) < Math.abs(lastDrawn - want)) draw(want); })
      .catch(() => {}).finally(next); };
  for (let c = 0; c < 6; c++) next();
}
function sizeCanvas() { const r = Math.max(1, Math.min(devicePixelRatio || 1, 1280 / innerWidth)); canvas.width = innerWidth * r; canvas.height = innerHeight * r; lastDrawn = -1; draw(want); }
function nearest(i) { for (let d = 0; d < N_FRAMES; d++) { if (frames[i - d]) return i - d; if (frames[i + d]) return i + d; } return -1; }
function draw(i) {
  const j = nearest(i); if (j < 0 || j === lastDrawn) return;
  const im = frames[j], cw = canvas.width, ch = canvas.height, s = Math.max(cw / im.width, ch / im.height);
  const w = im.width * s, h = im.height * s; ctx.drawImage(im, (cw - w) / 2, (ch - h) / 2, w, h); lastDrawn = j;
}
let want = 0;

/* ---------------- chapters ------------------------------------------------------------------------------ */
const title = $("#chapterTitle"), statement = $("#statement"), lede = $("#lede");
const metaA = $("#metaA"), metaB = $("#metaB"), metaC = $("#metaC");
let S = null; // summary
const CHAPTERS = [
  { until: 0.24, title: "BASINPULSE", meta: ["[ 1 · 3 · 7 DAYS AHEAD ]", "60 RIVERS IT HAS NEVER SEEN", "CAMELS-US · 2008–2014"] },
  { until: 0.50, title: "THE MELT", meta: ["[ ZERO-SHOT ]", "579 ALPINE RIVERS · 0 RETRAINING", "LAMAH-CE · 2008–2014"],
    text: () => S && S.europe ? `TRAINED ON AMERICAN RIVERS. SENT, UNTOUCHED, TO ${S.europe_basins} IN THE ALPS. ONE DAY AHEAD IT SCORES NSE ${fmt(S.europe["BasinPulse (final)"].nse_median["1"])}.`
      : "TRAINED ON AMERICAN RIVERS. SENT, UNTOUCHED, TO THE ALPS." },
  { until: 0.78, title: "THE SURGE", meta: ["[ HIGH-FLOW ALERTS ]", "P(FLOW > 95TH PERCENTILE)", "60 UNSEEN RIVERS"],
    text: () => { if (!S) return "IT WARNS BEFORE THE SURGE, AND SAYS HOW SURE IT IS.";
      const m = S.model.geo_temporal, p = S.persistence.geo_temporal;
      return `${Math.round(100 * m.precision_high["1"].value)}% OF ITS FLOOD ALERTS ARE RIGHT. FALSE ALARMS DOWN FROM ${fmt(p.false_alarms_per_basin_year_high["1"].value, 1)} TO ${fmt(m.false_alarms_per_basin_year_high["1"].value, 1)} A YEAR PER RIVER.`; } },
  { until: 1.01, title: "BEFORE IT STRIKES", meta: ["[ THE BENCHMARK ]", "TEST SCORED ONCE", "15 LEAKAGE CHECKS PASS"] },
];
let chapter = -1, scrambleTimer = null, onPanels = false;
function scramble(el, text) {
  if (reduced) { el.textContent = text; return; }
  clearInterval(scrambleTimer);
  const glyphs = "ᚠᚢᚦᚨᚱᚲᚷᚹᚺᚾᛁᛃᛇᛈᛉᛊᛏᛒᛖᛗᛚᛜᛞᛟ#/%_";
  let t = 0; const n = text.length;
  scrambleTimer = setInterval(() => {
    t += 1; el.textContent = [...text].map((c, i) => (c === " " ? " " : i < t - 6 ? c : glyphs[(Math.random() * glyphs.length) | 0])).join("");
    if (t > n + 6) { el.textContent = text; clearInterval(scrambleTimer); }
  }, 28);
}
function setChapter(i) {
  if (i === chapter) return; chapter = i; const c = CHAPTERS[i];
  scramble(title, c.title); [metaA.textContent, metaB.textContent, metaC.textContent] = c.meta;
  lede.style.opacity = i === 0 ? 1 : 0; lede.style.transform = i === 0 ? "none" : "translateY(10px)";
}

/* ---------------- scroll handler -------------------------------------------------------------------------- */
const story = $("#story"), film = $("#film"), stack = $("#stack"), lines = [...document.querySelectorAll("#stackLines li")];
const bar = $("#progressBar"), topEl = $("#top");
let shown = 0, raf = 0, lastStatement = null;
function tick() {
  raf = 0;
  const r = story.getBoundingClientRect(), total = story.offsetHeight - innerHeight;
  const target = Math.min(1, Math.max(0, -r.top / total));
  shown = reduced || Math.abs(target - shown) < 0.0004 ? target : shown + (target - shown) * 0.16;
  onScroll(shown, r);
  if (shown !== target) raf = requestAnimationFrame(tick);
}
function onScroll(p, r) {
  bar.style.width = `${p * 100}%`;
  const filmEnd = 0.78;
  // the last frames fade to black, so the shrinking card holds the flood at its height (frame ~205)
  want = p < filmEnd ? Math.round((p / filmEnd) * 205) : 205; draw(want);
  const ci = CHAPTERS.findIndex((c) => p < c.until); setChapter(ci < 0 ? CHAPTERS.length - 1 : ci);
  // typewriter statement, driven by scroll inside the chapter
  const c = CHAPTERS[chapter];
  if (c.text) { const from = chapter === 0 ? 0 : CHAPTERS[chapter - 1].until, k = Math.min(1, (p - from) / ((c.until - from) * 0.7));
    const full = c.text(), html = full.slice(0, Math.ceil(full.length * k)) + (k < 1 ? '<span class="caret">&nbsp;</span>' : "");
    if (html !== lastStatement) { statement.innerHTML = html; lastStatement = html; } }
  else if (lastStatement !== "") { statement.textContent = ""; lastStatement = ""; }
  // final chapter: shrink the film into a tilted card over black, then the stacked lines
  const q = Math.min(1, Math.max(0, (p - filmEnd) / 0.1));
  const s = 1 - 0.52 * q, tilt = 9 * Math.sin(Math.PI * Math.min(1, q * 1.4));
  film.style.transform = `perspective(1400px) translateY(${-14 * q}vh) rotateX(${tilt}deg) scale(${s})`;
  film.style.borderRadius = q > 0 ? "6px" : "0";
  stack.classList.toggle("on", q > 0.6);
  lines.forEach((li, i) => li.classList.toggle("on", p > filmEnd + 0.08 + i * 0.045));
  const past = r.bottom <= innerHeight * 0.6;
  document.body.classList.toggle("on-panels", past);
  topEl.classList.toggle("compact", past || q > 0);   // small title once the card appears: no overlap
  if (past !== onPanels) { onPanels = past; scramble(title, past ? "BASINPULSE" : CHAPTERS[chapter].title); }
  $("#scrollHint").style.opacity = p > 0.02 ? 0 : 1;
}
const kick = () => { if (!raf) raf = requestAnimationFrame(tick); };
addEventListener("scroll", kick, { passive: true });
addEventListener("resize", () => { sizeCanvas(); kick(); });

/* ---------------- API ------------------------------------------------------------------------------------------ */
const api = (p) => fetch(p).then((r) => (r.ok ? r.json() : Promise.reject(r.statusText)));

function renderResults() {
  const m = S.model.geo_temporal, p = S.persistence.geo_temporal;
  const kpi = (lab, k, h, d = 2, pct = false) => {
    const v = m[k][h], b = p[k][h], f = (x) => (pct ? `${Math.round(100 * x)}%` : fmt(x, d));
    return `<div class="kpi"><span class="lab">${lab}</span><span class="big">${f(v.value)}</span>
      <span class="vs">persistence <b>${f(b.value)}</b>${v.lo != null ? ` · 95% CI ${f(v.lo)}–${f(v.hi)}` : ""}</span></div>`; };
  $("#kpis").innerHTML = kpi("NSE · 1 day ahead", "nse_median", "1") + kpi("NSE · 3 days ahead", "nse_median", "3")
    + kpi("Flood detection (AUPRC) · 1 day", "auprc_high", "1") + kpi("Range coverage · target 80%", "interval_coverage", "3", 2, true);
  const rows = [["NSE (median over rivers)", "nse_median", 2], ["KGE", "kge_median", 2], ["MAE, mm/day", "mae_pooled", 2],
    ["High-flow AUPRC", "auprc_high", 2], ["Extreme-flow AUPRC", "auprc_extreme", 2], ["Brier skill (high flow)", "brier_skill_high", 2],
    ["Alert precision (p ≥ 0.5)", "precision_high", 2], ["False alarms / river-year", "false_alarms_per_basin_year_high", 1],
    ["10–90% range coverage", "interval_coverage", 2], ["10–90% range width, mm/day", "interval_width", 2]];
  $("#compare").innerHTML = `<table><thead><tr><th>Unseen rivers, later years</th><th>1 day</th><th>3 days</th><th>7 days</th></tr></thead><tbody>${
    rows.map(([lab, k, d]) => `<tr><td>${lab}</td>${["1", "3", "7"].map((h) => `<td>${fmt(m[k][h] && m[k][h].value, d)}<span class="p">${fmt(p[k][h] && p[k][h].value, d)}</span></td>`).join("")}</tr>`).join("")
  }</tbody></table><p class="mono small">BasinPulse first, persistence in grey.</p>`;
  if (S.europe) { const e = S.europe["BasinPulse (final)"], ep = S.europe["Persistence"];
    $("#europe").innerHTML = `<p class="mono kicker">ZERO-SHOT · CENTRAL EUROPE</p><p class="body">Trained only on US rivers, then run without any retraining on
      <b>${S.europe_basins} rivers in Austria, Germany and Switzerland</b>: NSE <b>${fmt(e.nse_median["1"])}</b> one day ahead and
      <b>${fmt(e.nse_median["3"])}</b> three days ahead (persistence ${fmt(ep.nse_median["1"])} / ${fmt(ep.nse_median["3"])}); extreme-flow detection
      <b>${fmt(e.auprc_extreme["1"])}</b> vs ${fmt(ep.auprc_extreme["1"])}.</p>`; }
  const f = $("#stackForecasts"); if (f) f.textContent = `${(S.counts.forecasts / 1e6).toFixed(1)}M FORECASTS`;
}

/* ---------------- map + river detail ------------------------------------------------------------------------ */
let RIVERS = [], sel = 0, H = 1, DETAIL = null;
const colr = (v) => (v == null ? css("--mute") : v >= 0.4 ? css("--good") : v >= 0 ? css("--mid") : css("--bad"));
function drawMap() {
  const W = 640, Hh = 380, x = (lo) => ((lo + 125) / 59) * (W - 40) + 20, y = (la) => ((50 - la) / 26) * (Hh - 40) + 20;
  let g = '<g class="grid">';
  for (let la = 25; la <= 50; la += 5) g += `<line x1="10" x2="${W - 10}" y1="${y(la)}" y2="${y(la)}"/>`;
  for (let lo = -120; lo <= -70; lo += 10) g += `<line y1="10" y2="${Hh - 10}" x1="${x(lo)}" x2="${x(lo)}"/>`;
  g += "</g>";
  RIVERS.forEach((r, i) => {
    if (r.novelty_group === "very new") g += `<circle class="novring" cx="${x(r.lon)}" cy="${y(r.lat)}" r="11"/>`;
    g += `<circle class="pt${i === sel ? " sel" : ""}" data-i="${i}" cx="${x(r.lon)}" cy="${y(r.lat)}" r="6.5" fill="${colr(r.nse["3"])}" tabindex="0">
      <title>${r.name} · 3-day NSE ${fmt(r.nse["3"])} (persistence ${fmt(r.nse_persistence["3"])}) · ${r.novelty_group} catchment</title></circle>`;
  });
  const svg = $("#map"); svg.innerHTML = g;
  svg.querySelectorAll(".pt").forEach((c) => { c.onclick = () => pickRiver(+c.dataset.i); c.onkeydown = (e) => { if (e.key === "Enter") pickRiver(+c.dataset.i); }; });
}
async function pickRiver(i) {
  sel = i; const r = RIVERS[i]; $("#pick").value = i; drawMap();
  $("#rName").textContent = r.name;
  $("#rMeta").textContent = `${r.id} · ${Number(r.area_km2).toLocaleString()} km² · ${r.novelty_group} catchment (novelty ${fmt(r.novelty)}) · NSE 1/3/7 days ${fmt(r.nse["1"])} / ${fmt(r.nse["3"])} / ${fmt(r.nse["7"])}`;
  DETAIL = await api(`/api/river/${r.id}`); drawChart(); renderAlert();
  $("#rSrc").innerHTML = `<b>Real data, not a simulation.</b> White line: the flow actually measured at ${usgs(r.id)} `
    + `(Caravan v1.6 / CAMELS-US), ${DETAIL.dates[0]} to ${DETAIL.dates[DETAIL.dates.length - 1]}, around this river’s largest flood in the test years. `
    + `Blue line and range: the forecasts BasinPulse actually made in the one-time test run (outputs/final_T0), each using only data from before the forecast was made (1, 3 or 7 days ahead). `
    + `This river was never used in training. Shown from that logged run, so it appears without any upload.`;
}
function lineChart(svg, d, opts) {
  const W = +svg.viewBox.baseVal.width, Hh = +svg.viewBox.baseVal.height, L = 46, R = 46, T = 14, B = 30, n = d.dates.length;
  const vals = [...d.observed, ...(d.upper || []), ...(d.persistence || []), opts.thr].filter((v) => v != null && isFinite(v));
  const ymax = Math.max(...vals) * 1.06 || 1;
  const X = (i) => L + (i / Math.max(1, n - 1)) * (W - L - R), Y = (v) => T + (1 - v / ymax) * (Hh - T - B), P = (v) => T + (1 - v) * (Hh - T - B);
  const path = (a, fy) => { let s = "", on = false; a.forEach((v, i) => { if (v == null) { on = false; return; } s += `${on ? "L" : "M"}${X(i).toFixed(1)},${fy(v).toFixed(1)}`; on = true; }); return s; };
  let band = "";
  if (d.upper) { const up = [], dn = []; d.upper.forEach((v, i) => { if (v != null && d.lower[i] != null) { up.push([X(i), Y(v)]); dn.push([X(i), Y(d.lower[i])]); } });
    if (up.length) band = `<path d="M${up.map((p) => p.join(",")).join("L")}L${dn.reverse().map((p) => p.join(",")).join("L")}Z" fill="rgba(143,211,232,.22)"/>`; }
  let ax = "";
  for (let k = 0; k <= 4; k++) { const v = (ymax * k) / 4; ax += `<g class="axis"><line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" stroke-dasharray="2 5"/><text x="${L - 6}" y="${Y(v) + 3}" text-anchor="end">${v.toFixed(v < 10 ? 1 : 0)}</text></g>`; }
  if (d.p_high) for (let k = 0; k <= 2; k++) ax += `<g class="axis"><text x="${W - R + 6}" y="${P(k / 2) + 3}">${k * 50}%</text></g>`;
  [0, Math.floor(n / 2), n - 1].forEach((i) => (ax += `<g class="axis"><text x="${X(i)}" y="${Hh - 8}" text-anchor="middle">${d.dates[i]}</text></g>`));
  let alerts = "";
  if (d.drivers) d.drivers.forEach((dr, i) => { if (dr) alerts += `<rect x="${X(i) - 1}" y="${Hh - B}" width="2" height="5" fill="${css("--signal")}"/>`; });
  svg.innerHTML = ax + band + (opts.thr ? `<line x1="${L}" x2="${W - R}" y1="${Y(opts.thr)}" y2="${Y(opts.thr)}" stroke="${css("--bad")}" stroke-dasharray="5 4"/>` : "")
    + (d.p_high ? `<path d="${path(d.p_high, P)}" fill="none" stroke="${css("--signal")}" stroke-width="1.2" opacity=".85"/>` : "")
    + (d.persistence ? `<path d="${path(d.persistence, Y)}" fill="none" stroke="${css("--mute")}" stroke-width="1.2" stroke-dasharray="4 3"/>` : "")
    + `<path d="${path(d.observed, Y)}" fill="none" stroke="${css("--paper")}" stroke-width="1.8"/>`
    + `<path d="${path(d.forecast, Y)}" fill="none" stroke="${css("--ice")}" stroke-width="1.8"/>` + alerts
    + `<line id="${svg.id}Cursor" y1="${T}" y2="${Hh - B}" stroke="${css("--paper")}" stroke-opacity=".35" visibility="hidden"/>`
    + `<text x="${L}" y="${T - 2}" class="axis" fill="${css("--mute")}" font-size="10">flow, mm/day</text>`;
  return { X, n, L, R, W };
}
// hover / touch read-out for a chart drawn by lineChart; row(i) gives the tooltip for point i
function hover(svg, tip, g, row) {
  const cur = svg.querySelector(`#${svg.id}Cursor`);
  const show = (cx, cy) => {
    const b = svg.getBoundingClientRect(), fx = ((cx - b.left) / b.width) * g.W;
    const i = Math.round(((fx - g.L) / (g.W - g.L - g.R)) * (g.n - 1)); if (i < 0 || i >= g.n) return;
    cur.setAttribute("x1", g.X(i)); cur.setAttribute("x2", g.X(i)); cur.setAttribute("visibility", "visible");
    tip.innerHTML = row(i); tip.hidden = false;
    const px = cx - b.left; tip.style.left = `${Math.max(0, Math.min(px + 14, b.width - 220))}px`; tip.style.top = `${cy - b.top + 14}px`;
  };
  const hide = () => { tip.hidden = true; cur.setAttribute("visibility", "hidden"); };
  svg.onmousemove = (e) => show(e.clientX, e.clientY);
  svg.onmouseleave = hide;
  svg.ontouchstart = svg.ontouchmove = (e) => { const t = e.touches[0]; if (t) show(t.clientX, t.clientY); };
  svg.ontouchend = hide;
}
// official USGS page of a CAMELS gauge (camels_09210500 -> 09210500), so anyone can check the measured flow
const usgs = (id) => { const g = String(id).replace(/^camels_/, "");
  return /^\d{8}$/.test(g) ? `<a href="https://waterdata.usgs.gov/monitoring-location/USGS-${g}/" target="_blank" rel="noopener">USGS gauge ${g}</a>` : esc(id); };
const pct = (v) => (v == null ? "–" : Math.round(v * 100) + "%");
const H_LABEL = (h) => `forecast made ${h} day${h > 1 ? "s" : ""} before`;
function readout(date, h, i, obs) {
  return `${date} · ${H_LABEL(h.horizon)}<br>observed <b style="color:var(--paper)">${fmt(obs)}</b> · forecast <b style="color:var(--ice)">${fmt(h.forecast[i])}</b><br>range ${fmt(h.lower[i])}–${fmt(h.upper[i])} · P(high) <b>${pct(h.p_high[i])}</b>`;
}
function drawChart() {
  const d = DETAIL, h = { ...d["h" + H], horizon: H };
  const g = lineChart($("#chart"), { dates: d.dates, observed: d.observed, ...h }, { thr: d.q_high });
  hover($("#chart"), $("#tip"), g, (i) => { const dr = h.drivers[i];
    return readout(d.dates[i], h, i, d.observed[i])
      + (dr ? `<br>why: ${dr.filter((x) => x[1] > 0).map((x) => `${x[0]} ${Math.round(x[1] * 100)}%`).join(" · ")}` : ""); });
}
function renderAlert() {
  const d = DETAIL, h = d.h1; let best = -1;
  h.p_high.forEach((p, i) => { if (h.drivers[i] && (best < 0 || p > h.p_high[best])) best = i; });
  const r = RIVERS[sel];
  if (best < 0) { $("#sgAlert").innerHTML = `<h4>${r.name}</h4><p class="mono when">No flood alert in this window.</p>`; return; }
  const drv = h.drivers[best].filter((x) => x[1] > 0);
  $("#sgAlert").innerHTML = `<h4>${r.name}</h4><p class="mono when">Forecast for ${d.dates[best]}, issued one day before · P(high flow) ${Math.round(h.p_high[best] * 100)}%</p>
    <p><span class="badge hot">${r.novelty_group} catchment</span><span class="badge">novelty ${fmt(r.novelty)}</span></p>
    ${drv.map((x) => `<div class="driver"><span>${x[0]}</span><span class="bar"><i style="width:${Math.round(x[1] * 100)}%"></i></span><span class="pct">${Math.round(x[1] * 100)}%</span></div>`).join("")}
    <p class="mono small">Shares of the push towards “flood”, from the model’s own tree contributions.</p>`;
}
const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
// tab row; default: the three horizons. labels (optional) gives one tab per entry, valued by its index
function tabs(el, onPick, active, labels) {
  const items = labels ? labels.map((l, i) => [i, l]) : [1, 3, 7].map((h) => [h, `${h} DAY${h > 1 ? "S" : ""} AHEAD`]);
  el.innerHTML = items.map(([v, l]) => `<button type="button" role="tab" aria-selected="${v === active}" data-v="${v}" class="${v === active ? "on" : ""}">${esc(l)}</button>`).join("");
  el.querySelectorAll("button").forEach((b) => (b.onclick = () => {
    el.querySelectorAll("button").forEach((x) => { x.classList.toggle("on", x === b); x.setAttribute("aria-selected", x === b); });
    onPick(+b.dataset.v); }));
}

/* ---------------- try the model ------------------------------------------------------------------------------- */
let TRY = null, TRY_H = 1, TRY_I = 0, busy = false;
async function runForecast(file, label) {
  if (busy) return; busy = true;
  const st = $("#tryStatus"), btn = $("#useSample"), t0 = Date.now();
  st.classList.remove("err"); btn.disabled = true; drop.classList.add("busy");
  $("#trySrc").textContent = `Forecasting ${label}`;
  const tick = () => { st.textContent = `Running the submitted model… ${Math.round((Date.now() - t0) / 1000)} s`; };
  tick(); const timer = setInterval(tick, 500);
  const fd = new FormData(); fd.append("file", file);
  try {
    const r = await fetch("/api/forecast", { method: "POST", body: fd });
    const j = await r.json().catch(() => ({ detail: r.statusText || "the server did not answer" }));
    if (!r.ok) throw new Error(j.detail || r.statusText);
    if (!j.basins || !j.basins.length) throw new Error("no rivers with usable rows in this file");
    TRY = { ...j, label }; TRY_I = 0;
    st.textContent = `${j.n_origins.toLocaleString()} forecast days · ${j.basins.length} river${j.basins.length > 1 ? "s" : ""} · ${j.seconds} s on the server · output ${j.valid_submission ? "passes" : "fails"} the BASINSHIFT format checks`;
    $("#trySrc").innerHTML = `Live forecast of <b>${esc(label)}</b>`;
    $("#trySrcNote").innerHTML = (label.startsWith("the real sample")
      ? `<b>Real data, forecast live just now.</b> Inputs: the measured record of ${j.basins.map((x) => usgs(x.id)).join(" and ")} `
        + `(Caravan v1.6 / CAMELS-US: observed flow and ERA5-Land weather, Oct 2012 – Dec 2013). `
      : `<b>Forecast live just now from your file.</b> `)
      + `The white line is the observed flow from the file; the blue line and range are what the submitted model predicted, each using only data from before the forecast was made (1, 3 or 7 days ahead).`;
    $("#tryEmpty").hidden = true; $("#tryResult").hidden = false;
    tabs($("#tryRivers"), (i) => { TRY_I = i; drawTry(); }, 0, j.basins.map((x) => x.id.replace("camels_", "river ")));
    drawTry();
  } catch (e) {
    st.classList.add("err"); st.textContent = `Could not forecast: ${e.message}`;
    if (TRY) $("#trySrc").innerHTML = `Live forecast of <b>${esc(TRY.label)}</b> (the new file could not be used)`;
    else $("#trySrc").textContent = "No forecast yet.";
  } finally { clearInterval(timer); busy = false; btn.disabled = false; drop.classList.remove("busy"); }
}
function drawTry() {
  if (!TRY) return; const b = TRY.basins[TRY_I], h = { ...b["h" + TRY_H], horizon: TRY_H };
  const g = lineChart($("#tryChart"), h, { thr: b.q_high });
  hover($("#tryChart"), $("#tryTip"), g, (i) => readout(h.dates[i], h, i, h.observed[i]));
}
const drop = $("#drop");
$("#file").onchange = (e) => { const f = e.target.files[0]; if (f) runForecast(f, `your file “${f.name}”`); e.target.value = ""; };
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) runForecast(f, `your file “${f.name}”`); });
$("#useSample").onclick = async () => {
  try { const r = await fetch("/api/sample"); if (!r.ok) throw new Error(r.statusText);
    runForecast(new File([await r.blob()], "sample.csv", { type: "text/csv" }), "the real sample (2 USGS rivers the model never saw, 2013)");
  } catch (e) { $("#tryStatus").classList.add("err"); $("#tryStatus").textContent = `Could not load the sample: ${e.message}`; } };
tabs($("#tryTabs"), (h) => { TRY_H = h; drawTry(); }, 1);

/* ---------------- boot ------------------------------------------------------------------------------------------- */
sizeCanvas(); loadFrames(); kick();
api("/api/summary").then((s) => { S = s; renderResults(); lastStatement = null; chapter = -1; kick();
  const m = s.model.geo_temporal;
  $("#sgFacts").innerHTML = `<li>Ranges hold <b>${Math.round(100 * m.interval_coverage["3"].value)}%</b> of real values (target 80%)</li>
    <li>Late flow data, age reported: range coverage stays at <b>81%</b> (was 57%)</li>
    <li>No flow data at all: still beats persistence beyond one day</li>`; }).catch(() => {});
api("/api/rivers").then((r) => { RIVERS = r; $("#pick").innerHTML = r.map((x, i) => `<option value="${i}">${x.name}</option>`).join("");
  $("#pick").onchange = (e) => pickRiver(+e.target.value); tabs($("#tabs"), (h) => { H = h; drawChart(); }, 1);
  let best = 0; r.forEach((x, i) => { if ((x.nse["3"] ?? -9) > (r[best].nse["3"] ?? -9)) best = i; }); pickRiver(best); }).catch(() => {});
