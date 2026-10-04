"""AstroRisk geomagnetic view.

Everything drawn here is derived from live feeds or from published physical models:

* aurora oval  - NOAA SWPC OVATION nowcast (1-degree cells)
* routes       - great circles between airports, sampled against that oval
* power grids  - reference cities, distance to the oval edge
* magnetopause - Shue et al. (1998) empirical model, driven by live solar-wind pressure and Bz
* Earth, coastlines - Natural Earth 110m land (public domain), assets/land_110m.json

No aircraft positions, no simulated data. If a feed is missing the view says so.
This module has no Streamlit dependency; app.py calls build() and panel_html().
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

OVATION_URL = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
UA = {"User-Agent": "AstroRisk/4.0 (+student project)"}

AURORA_KEEP = 3            # cells below this are not sent to the browser
AURORA_MIN = 10            # a cell counts as "under the oval" from here up (heuristic; the legend says so)
STALE_OVAL_MIN = 60        # reuse the previous oval for at most this long if NOAA is unreachable
POLE = (80.7, -72.7)       # approximate dipole pole of the geomagnetic field (IGRF, c. 2020-25); drifts slowly
RE_KM = 6371.0
GEO_RE = 42164.0 / RE_KM   # geostationary orbit radius in Earth radii (6.62)
GNSS_RE = 26560.0 / RE_KM  # GPS orbit radius (4.17)
LEO_KM = (400, 2000)

AIRPORTS = {               # approximate coordinates, enough for a globe-scale picture
    "JFK": (40.64, -73.78), "EWR": (40.69, -74.17), "ORD": (41.97, -87.91), "SFO": (37.62, -122.38),
    "LAX": (33.94, -118.41), "YVR": (49.19, -123.18), "LHR": (51.47, -0.46), "DXB": (25.25, 55.36),
    "DOH": (25.27, 51.61), "DEL": (28.56, 77.10), "BLR": (13.20, 77.71), "HKG": (22.31, 113.91),
    "SIN": (1.36, 103.99),
}
ROUTES = [("JFK", "HKG"), ("ORD", "DEL"), ("SFO", "BLR"), ("SFO", "DXB"), ("JFK", "DXB"),
          ("LAX", "DOH"), ("EWR", "SIN"), ("YVR", "HKG"), ("LHR", "JFK")]
CITIES = [("Edmonton", 53.55, -113.49), ("Montreal", 45.50, -73.57), ("Chicago", 41.88, -87.63),
          ("New York", 40.71, -74.01), ("Dallas", 32.78, -96.80), ("London", 51.51, -0.13),
          ("Stockholm", 59.33, 18.07), ("Helsinki", 60.17, 24.94), ("Dubai", 25.20, 55.27),
          ("Johannesburg", -26.20, 28.05), ("Auckland", -36.85, 174.76)]

LAND_PATH = Path(__file__).parent / "assets" / "land_110m.json"


# ─────────────────────────────────────────────────────────────────────────────
# geometry
# ─────────────────────────────────────────────────────────────────────────────
def mag_lat(lat: float, lon: float) -> float:
    """Geomagnetic latitude from the dipole pole."""
    p, l0 = math.radians(POLE[0]), math.radians(POLE[1])
    f, l = math.radians(lat), math.radians(lon)
    s = math.sin(f) * math.sin(p) + math.cos(f) * math.cos(p) * math.cos(l - l0)
    return math.degrees(math.asin(max(-1.0, min(1.0, s))))


def great_circle(a: Tuple[float, float], b: Tuple[float, float], n: int = 90) -> Tuple[List[Tuple[float, float]], float]:
    """n+1 points along the great circle a->b, and its length in km."""
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    v1 = np.array([math.cos(la1) * math.cos(lo1), math.cos(la1) * math.sin(lo1), math.sin(la1)])
    v2 = np.array([math.cos(la2) * math.cos(lo2), math.cos(la2) * math.sin(lo2), math.sin(la2)])
    d = math.acos(max(-1.0, min(1.0, float(v1 @ v2))))
    pts = []
    for i in range(n + 1):
        t = i / n
        v = ((math.sin((1 - t) * d) * v1 + math.sin(t * d) * v2) / math.sin(d)) if d > 1e-9 else v1
        pts.append((math.degrees(math.atan2(v[2], math.hypot(v[0], v[1]))), math.degrees(math.atan2(v[1], v[0]))))
    return pts, d * RE_KM


def _wrap_lon(lon: float) -> int:
    x = int(round(lon))
    x = (x + 180) % 360 - 180
    return x


# ─────────────────────────────────────────────────────────────────────────────
# live aurora (NOAA OVATION)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_ovation(timeout: int = 25) -> Dict:
    r = requests.get(OVATION_URL, headers=UA, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    rows = j.get("coordinates") or []
    if len(rows) < 1000:
        raise ValueError(f"OVATION file has only {len(rows)} grid cells")
    cells = []
    for row in rows:
        lon, lat, v = row[0], row[1], row[2]
        if v is not None and v >= AURORA_KEEP:
            cells.append([_wrap_lon(lon), int(round(lat)), int(round(v))])
    return {"obs": j.get("Observation Time"), "forecast": j.get("Forecast Time"), "cells": cells}


def _age_min(iso: Optional[str], now: datetime) -> Optional[float]:
    try:
        t = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        return max(0.0, (now - t).total_seconds() / 60.0)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# magnetopause (Shue et al. 1998)
# ─────────────────────────────────────────────────────────────────────────────
def shue(pdyn: float, bz: float) -> Dict:
    r0 = (10.22 + 1.29 * math.tanh(0.184 * (bz + 8.14))) * pdyn ** (-1.0 / 6.6)
    alpha = (0.58 - 0.007 * bz) * (1.0 + 0.024 * math.log(pdyn))
    # sunward angle out to which the geostationary ring lies outside the boundary
    exposed = 0.0
    if r0 < GEO_RE:
        c = 2.0 * (r0 / GEO_RE) ** (1.0 / alpha) - 1.0
        exposed = math.degrees(math.acos(max(-1.0, min(1.0, c))))
    return {"pdyn": round(pdyn, 2), "bz": round(bz, 1), "r0": round(r0, 2), "alpha": round(alpha, 3),
            "geo_re": round(GEO_RE, 2), "gnss_re": round(GNSS_RE, 2),
            "geo_margin": round(r0 - GEO_RE, 2), "geo_exposed_deg": round(exposed, 1)}


def _g_scale(kp: Optional[float]) -> str:
    return "G0" if kp is None or kp < 5 else f"G{min(5, int(kp) - 4)}"


# ─────────────────────────────────────────────────────────────────────────────
# build the payload the browser draws
# ─────────────────────────────────────────────────────────────────────────────
def _oval_stats(cells: List[List[int]]) -> Dict:
    north = [c for c in cells if c[2] >= AURORA_MIN and c[1] > 0]
    south = [c for c in cells if c[2] >= AURORA_MIN and c[1] < 0]
    out = {"n_cells": sum(1 for c in cells if c[2] >= AURORA_MIN), "north_min_mlat": None, "south_max_mlat": None}
    if north:
        out["north_min_mlat"] = round(min(mag_lat(c[1], c[0]) for c in north), 1)
    if south:
        out["south_max_mlat"] = round(max(mag_lat(c[1], c[0]) for c in south), 1)
    return out


def _routes(grid: Optional[Dict[Tuple[int, int], int]]) -> List[Dict]:
    out = []
    for a, b in ROUTES:
        pts, km = great_circle(AIRPORTS[a], AIRPORTS[b])
        vals = [None if grid is None else grid.get((_wrap_lon(lo), int(round(la))), 0) for la, lo in pts]
        under = None if grid is None else round(100.0 * sum(1 for v in vals if v >= AURORA_MIN) / len(vals))
        out.append({"id": f"{a}-{b}", "a": a, "b": b, "km": round(km), "max_mlat": round(max(abs(mag_lat(la, lo)) for la, lo in pts), 1),
                    "under_pct": under, "peak": None if grid is None else max(vals),
                    "pts": [[round(la, 1), round(lo, 1), v] for (la, lo), v in zip(pts, vals)]})
    out.sort(key=lambda r: (-(r["under_pct"] or 0), -r["max_mlat"]))
    return out


def _cities(cells: Optional[List[List[int]]], grid: Optional[Dict[Tuple[int, int], int]]) -> List[Dict]:
    arr = None
    if cells is not None:
        sel = np.array([c for c in cells if c[2] >= AURORA_MIN], dtype=float).reshape(-1, 3)
        arr = sel if len(sel) else None
    out = []
    for name, lat, lon in CITIES:
        row = {"name": name, "lat": lat, "lon": lon, "mlat": round(mag_lat(lat, lon), 1), "v": None, "inside": None, "edge_deg": None}
        if grid is not None:
            v = grid.get((_wrap_lon(lon), int(round(lat))), 0)
            row["v"], row["inside"] = v, v >= AURORA_MIN
            if arr is not None:
                same = arr[np.sign(arr[:, 1]) == np.sign(lat)]
                if len(same):
                    f1, f2 = math.radians(lat), np.radians(same[:, 1])
                    dl = np.radians(same[:, 0] - lon)
                    cosd = np.clip(math.sin(f1) * np.sin(f2) + math.cos(f1) * np.cos(f2) * np.cos(dl), -1, 1)
                    row["edge_deg"] = round(float(np.degrees(np.arccos(cosd)).min()), 1)
        out.append(row)
    out.sort(key=lambda r: (0 if r["inside"] else 1, r["edge_deg"] if r["edge_deg"] is not None else 999))
    return out


def _sat_rows(raw: Dict, mag: Optional[Dict]) -> List[Dict]:
    kp, el, bz = raw.get("kp"), raw.get("electron_flux"), raw.get("bz_gsm")
    leo = {"name": "Low Earth orbit", "re": "1.06-1.3 Re", "detail": (
        f"{LEO_KM[0]}-{LEO_KM[1]} km. Drag rises with geomagnetic activity; Kp is {kp:.1f} ({_g_scale(kp)})." if kp is not None
        else f"{LEO_KM[0]}-{LEO_KM[1]} km. Drag rises with geomagnetic activity; no Kp reading.")}
    gnss = {"name": "Navigation (GPS class)", "re": f"{GNSS_RE:.2f} Re", "detail": (
        f"{mag['r0'] - GNSS_RE:.1f} Re inside the magnetopause. No flux measurement at this altitude in the feeds used." if mag
        else "Magnetopause not computed (needs solar-wind pressure and Bz).")}
    if mag:
        if mag["geo_exposed_deg"] > 0:
            geo_d = (f"Outside the magnetopause on the sunward side, about {2 * mag['geo_exposed_deg'] / 15:.0f} h of local time "
                     f"({100 * mag['geo_exposed_deg'] / 180:.0f}% of the ring).")
        else:
            geo_d = f"{mag['geo_margin']:.1f} Re inside the magnetopause at the nose."
    else:
        geo_d = "Magnetopause not computed (needs solar-wind pressure and Bz)."
    if el is not None:
        geo_d += f" GOES >2 MeV electrons: {el:,.0f} pfu."
    geo = {"name": "Geostationary", "re": f"{GEO_RE:.2f} Re", "detail": geo_d}
    return [leo, gnss, geo]


def build(raw: Dict, now: Optional[datetime] = None, prev: Optional[Dict] = None) -> Dict:
    """Payload for the browser. Never raises; failures are reported inside the payload."""
    now = now or datetime.now(timezone.utc)
    prev = prev or {}
    aurora, err = None, None
    try:
        aurora = fetch_ovation()
        aurora["stale_min"] = 0
    except Exception as exc:
        err = str(exc)[:160]
        pa = prev.get("aurora")
        age = _age_min(pa.get("obs"), now) if pa else None
        if pa and age is not None and age <= STALE_OVAL_MIN:
            aurora = {**pa, "stale_min": round(age)}
    cells = aurora["cells"] if aurora else None
    grid = {(c[0], c[1]): c[2] for c in cells} if cells is not None else None

    mag = None
    n, v, bz = raw.get("density"), raw.get("wind_speed"), raw.get("bz_gsm")
    if n and v and bz is not None:
        pdyn = 1.6726e-6 * n * v * v
        if pdyn > 0:
            mag = shue(pdyn, bz)

    routes = _routes(grid)
    cities = _cities(cells, grid)
    oval = _oval_stats(cells) if cells is not None else None

    # headlines: plain sentences made from the numbers above
    head: Dict[str, str] = {}
    if oval is None:
        head["aviation"] = head["power_grid"] = ("NOAA's aurora feed could not be read just now, so routes and cities are drawn "
                                                 "without oval exposure.")
    else:
        lo = oval["north_min_mlat"]
        hit = [r for r in routes if (r["under_pct"] or 0) > 0]
        top = f" Most exposed: {hit[0]['a']}-{hit[0]['b']}, {hit[0]['under_pct']}% of its length." if hit else ""
        head["aviation"] = ((f"The northern oval reaches down to {lo:.0f}\u00b0 geomagnetic latitude. " if lo is not None else "No northern oval above threshold. ")
                            + f"{len(hit)} of {len(routes)} reference routes pass under it.{top}")
        inside = [c for c in cities if c["inside"]]
        outside = [c for c in cities if c["inside"] is False and c["edge_deg"] is not None]
        near = (f" Closest outside: {outside[0]['name']}, {outside[0]['edge_deg']:.1f}\u00b0 ({outside[0]['edge_deg'] * 111.2:,.0f} km) from the oval."
                if outside else "")
        head["power_grid"] = (f"{len(inside)} of {len(cities)} reference cities sit under the oval right now." + near)
    if mag:
        if mag["geo_exposed_deg"] > 0:
            head["satellite"] = (f"The magnetopause nose is at {mag['r0']:.1f} Earth radii, inside the geostationary ring at {GEO_RE:.1f}. "
                                 f"Geostationary satellites on the sunward side are outside it.")
        else:
            head["satellite"] = (f"The magnetopause nose is at {mag['r0']:.1f} Earth radii. The geostationary ring at {GEO_RE:.1f} "
                                 f"sits {mag['geo_margin']:.1f} inside it.")
    else:
        head["satellite"] = "Magnetopause not computed: it needs live solar-wind density, speed and Bz."

    note_a = ("Aurora: NOAA OVATION nowcast"
              + (f", observation {str(aurora['obs'])[:16].replace('T', ' ')} UTC" if aurora and aurora.get("obs") else "")
              + (f" (reused, {aurora['stale_min']} min old: NOAA unreachable)" if aurora and aurora.get("stale_min") else "")
              + f". A cell counts as under the oval at {AURORA_MIN} or more; that threshold is my choice, not NOAA's. "
              "Geomagnetic latitude uses a dipole pole near 80.7\u00b0N, 72.7\u00b0W. Routes are great circles between airports, "
              "not flight plans, and no aircraft are shown.")
    notes = {"aviation": note_a,
             "power_grid": note_a.split(" Routes")[0] + " Distance is to the nearest oval cell in the same hemisphere. "
                           "Proximity to the oval is one GIC risk factor; this is not a GIC forecast.",
             "satellite": ("Magnetopause: Shue et al. 1998 empirical model from live pressure and Bz; less reliable under extreme driving. "
                           "Field lines are an untilted dipole, drawn for scale. 1 Re = 6,371 km.")}
    return {"ok": True, "built": now.isoformat(), "error": err,
            "aurora": aurora, "aurora_min": AURORA_MIN, "aurora_keep": AURORA_KEEP,
            "pole": list(POLE), "airports": {k: list(v) for k, v in AIRPORTS.items()},
            "routes": routes, "cities": cities, "oval": oval, "mag": mag,
            "speed": raw.get("wind_speed"), "kp": raw.get("kp"),
            "sat_rows": _sat_rows(raw, mag), "headline": head, "notes": notes}


# ─────────────────────────────────────────────────────────────────────────────
# HTML / CSS / JS (drawn on a plain 2D canvas: no libraries, no CDN)
# ─────────────────────────────────────────────────────────────────────────────
_LAND: Optional[str] = None


def _land_json() -> str:
    global _LAND
    if _LAND is None:
        try:
            _LAND = json.dumps(json.loads(LAND_PATH.read_text()), separators=(",", ":"))
        except Exception:
            _LAND = "null"
    return _LAND


def _script_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")


def panel_html(viz: Optional[Dict]) -> str:
    if not viz or not viz.get("ok"):
        return ('<section class="panel geo"><span class="cap">Geomagnetic view</span>'
                '<p class="geo-empty">Not available in this snapshot.</p></section>')
    return f"""
    <section class="panel geo" id="geo">
      <div class="geo-h">
        <span class="cap">Geomagnetic view</span>
        <div class="geo-tabs" role="tablist" aria-label="Choose a sector">
          <button type="button" role="tab" data-v="aviation" aria-selected="true">Aviation</button>
          <button type="button" role="tab" data-v="satellite" aria-selected="false">Satellites</button>
          <button type="button" role="tab" data-v="power_grid" aria-selected="false">Power grid</button>
        </div>
      </div>
      <div class="geo-body">
        <figure class="geo-fig">
          <canvas id="gl" aria-label="Interactive globe. The text list beside it has the same information."></canvas>
          <figcaption id="gl-read">Drag to rotate</figcaption>
          <ul class="geo-legend" id="geo-legend"></ul>
        </figure>
        <div class="geo-side">
          <p class="geo-head" id="geo-head"></p>
          <ul class="geo-rows" id="geo-rows"></ul>
          <p class="geo-note" id="geo-note"></p>
        </div>
      </div>
      <script type="application/json" id="geo-data">{_script_json(viz)}</script>
      <script type="application/json" id="geo-land">{_land_json()}</script>
    </section>"""


CSS = """
<style>
.geo-h{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:18px}
.geo-h .cap{margin:0}
.geo-tabs{display:flex;gap:4px}
.geo-tabs button{font:500 14px/1 var(--sans);color:var(--ink-3);background:none;border:0;border-bottom:2px solid transparent;
  padding:8px 12px;cursor:pointer}
.geo-tabs button:hover{color:var(--ink-2)}
.geo-tabs button[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--cyan)}
.geo-tabs button:focus-visible,.geo-row:focus-visible{outline:2px solid var(--cyan);outline-offset:2px;border-radius:4px}
.geo-body{display:grid;grid-template-columns:minmax(300px,1.05fr) minmax(280px,1fr);gap:26px;align-items:start}
.geo-fig{margin:0;min-width:0}
#gl{display:block;width:100%;aspect-ratio:1/1;max-width:600px;margin:0 auto;border-radius:14px;background:var(--well);
  border:1px solid var(--line);cursor:grab;touch-action:pan-y}
#gl.drag{cursor:grabbing}
#gl-read{margin-top:10px;min-height:16px;text-align:center;font:400 11.5px/1.4 var(--mono);color:var(--ink-3)}
.geo-legend{display:flex;flex-wrap:wrap;justify-content:center;gap:6px 18px;margin:8px 0 0;padding:0;list-style:none;
  font:400 11px/1.3 var(--mono);color:var(--ink-3)}
.geo-legend li{display:inline-flex;align-items:center;gap:7px}
.geo-legend i{display:inline-block;flex:none}
.geo-head{font:400 19px/1.38 var(--sans);color:var(--ink);max-width:46ch;margin:0 0 18px}
.geo-rows{display:grid;gap:2px;margin:0 0 16px;padding:0;list-style:none}
.geo-row{display:grid;grid-template-columns:minmax(92px,auto) 1fr auto;align-items:center;gap:14px;width:100%;
  padding:9px 10px;text-align:left;background:none;border:0;border-radius:8px;color:var(--ink-2);cursor:pointer;
  font:400 12px/1.35 var(--mono)}
.geo-row:hover{background:var(--well)}
.geo-row[aria-pressed="true"]{background:var(--well);box-shadow:inset 2px 0 0 var(--cyan)}
.geo-row b{font-weight:500;color:var(--ink)}
.geo-row .num{text-align:right;color:var(--ink-3);white-space:nowrap}
.geo-row .num em{font-style:normal;color:var(--yellow)}
.strip{display:grid;grid-auto-flow:column;grid-auto-columns:1fr;gap:1px;height:8px}
.strip i{background:var(--line-2);border-radius:1px}
.strip i.on{background:var(--yellow)}
.prox{display:block;max-width:240px;height:4px;border-radius:2px;background:var(--line-2);overflow:hidden}
.prox i{display:block;height:100%;background:var(--ink-3)}
.prox.in i{background:var(--yellow)}
.geo-row.flat{grid-template-columns:minmax(120px,auto) 1fr;cursor:default}
.geo-row.flat:hover{background:none}
.geo-row.flat .num{text-align:left;white-space:normal;color:var(--ink-2);font-family:var(--sans);font-size:13.5px;line-height:1.4}
.geo-note{font:400 10.5px/1.65 var(--mono);color:var(--ink-3);margin:0;max-width:64ch}
.geo-empty{font:400 13px/1.5 var(--mono);color:var(--ink-3);margin:0}
@media (max-width:860px){.geo-body{grid-template-columns:1fr}.geo-head{font-size:17px}}
@media (max-width:520px){.geo-row{grid-template-columns:1fr auto}.geo-row .strip{grid-column:1/-1;order:3}}
</style>
"""

SCRIPT = r"""
<script>
(function () {
  var root = document.getElementById('geo'); if (!root) return;
  var D, LAND = null;
  try { D = JSON.parse(document.getElementById('geo-data').textContent); } catch (e) { return; }
  try { LAND = JSON.parse(document.getElementById('geo-land').textContent); } catch (e) { LAND = null; }
  var $ = function (id) { return document.getElementById(id); };
  var cv = $('gl'), ctx = cv.getContext('2d');
  var cs = getComputedStyle(document.documentElement);
  var tok = function (n, f) { return (cs.getPropertyValue(n) || '').trim() || f; };
  var K = { bg: tok('--well', '#090f19'), line: tok('--line-2', '#22314a'), ink: tok('--ink', '#e8eef8'),
            ink2: tok('--ink-2', '#aab7cb'), ink3: tok('--ink-3', '#6a7a92'), ink4: tok('--ink-4', '#3d4a5f'),
            cyan: tok('--cyan', '#22d3ee'), amber: tok('--yellow', '#f5c542'), red: tok('--red', '#ff4157'),
            sans: tok('--sans', 'sans-serif'), mono: tok('--mono', 'monospace') };
  var RAD = Math.PI / 180, MIN = D.aurora_min, KEEP = D.aurora_keep;
  var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  var CENTER = { aviation: [72, -35], power_grid: [50, -40] };
  var S = { view: 'aviation', lat: 72, lon: -35, yaw: 28, pitch: 24, sel: null, hover: null };
  try { var sv = sessionStorage.getItem('geo.view'); if (sv && D.headline[sv]) S.view = sv; } catch (e) {}

  /* ── land: delta-decoded polylines ───────────────────────────────── */
  var land = [];
  if (LAND && LAND.lines) {
    LAND.lines.forEach(function (a) {
      var x = a[0], y = a[1], pts = [[x / LAND.scale, y / LAND.scale]];
      for (var i = 2; i < a.length; i += 2) { x += a[i]; y += a[i + 1]; pts.push([x / LAND.scale, y / LAND.scale]); }
      land.push(pts);
    });
  }
  var cells = (D.aurora && D.aurora.cells) || [];

  /* ── helpers ─────────────────────────────────────────────────────── */
  function dest(lat, lon, brg, dist) {
    var f = lat * RAD, l = lon * RAD, t = brg * RAD, d = dist * RAD;
    var f2 = Math.asin(Math.sin(f) * Math.cos(d) + Math.cos(f) * Math.sin(d) * Math.cos(t));
    var l2 = l + Math.atan2(Math.sin(t) * Math.sin(d) * Math.cos(f), Math.cos(d) - Math.sin(f) * Math.sin(f2));
    return [f2 / RAD, ((l2 / RAD + 540) % 360) - 180];
  }
  function sun() {
    var d = new Date(), y0 = Date.UTC(d.getUTCFullYear(), 0, 0);
    var doy = (Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()) - y0) / 864e5;
    var decl = -23.44 * Math.cos(2 * Math.PI / 365 * (doy + 10));
    var h = d.getUTCHours() + d.getUTCMinutes() / 60;
    return { lat: decl, lon: ((-15 * (h - 12) + 540) % 360) - 180 };
  }
  function magLat(lat, lon) {
    var p = D.pole[0] * RAD, l0 = D.pole[1] * RAD, f = lat * RAD, l = lon * RAD;
    return Math.asin(Math.max(-1, Math.min(1, Math.sin(f) * Math.sin(p) + Math.cos(f) * Math.cos(p) * Math.cos(l - l0)))) / RAD;
  }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function label(txt, x, y, color, align, font) {          // text with a dark halo so it stays legible over lines
    ctx.font = font || ('10px ' + K.mono); ctx.textAlign = align || 'left';
    ctx.lineJoin = 'round'; ctx.lineWidth = 3.5; ctx.strokeStyle = 'rgba(7,12,21,.92)'; ctx.strokeText(txt, x, y);
    ctx.fillStyle = color; ctx.fillText(txt, x, y);
  }
  function auroraRGBA(v) {                       // fixed scale, so colours mean the same thing all day
    var t = Math.max(0, Math.min(1, (v - KEEP) / (80 - KEEP)));
    var r = t < .6 ? lerp(18, 34, t / .6) : lerp(34, 225, (t - .6) / .4);
    var g = t < .6 ? lerp(120, 211, t / .6) : lerp(211, 252, (t - .6) / .4);
    var b = t < .6 ? lerp(150, 238, t / .6) : lerp(238, 255, (t - .6) / .4);
    return 'rgba(' + (r | 0) + ',' + (g | 0) + ',' + (b | 0) + ',' + (0.22 + 0.62 * t).toFixed(2) + ')';
  }

  /* ── canvas size ─────────────────────────────────────────────────── */
  var W = 0, H = 0, DPR = 1;
  function resize() {
    var w = Math.round(cv.getBoundingClientRect().width); if (!w) return;
    DPR = Math.min(2, window.devicePixelRatio || 1);
    W = H = w; cv.width = w * DPR; cv.height = w * DPR; draw();
  }
  if (window.ResizeObserver) new ResizeObserver(resize).observe(cv); else window.addEventListener('resize', resize);

  /* ── globe (orthographic) ────────────────────────────────────────── */
  var G = null;
  function setG() { G = { sl: Math.sin(S.lat * RAD), cl: Math.cos(S.lat * RAD), cx: W / 2, cy: H / 2, R: W * 0.455 }; }
  function P(lat, lon) {
    var f = lat * RAD, dl = (lon - S.lon) * RAD, cf = Math.cos(f), sf = Math.sin(f);
    var z = G.sl * sf + G.cl * cf * Math.cos(dl);
    return [G.cx + G.R * cf * Math.sin(dl), G.cy - G.R * (G.cl * sf - G.sl * cf * Math.cos(dl)), z];
  }
  function polyline(pts, path) {                 // pts: [[lat,lon]...]; breaks at the horizon
    var pen = false;
    for (var i = 0; i < pts.length; i++) {
      var p = P(pts[i][0], pts[i][1]);
      if (p[2] <= 0) { pen = false; continue; }
      if (pen) path.lineTo(p[0], p[1]); else path.moveTo(p[0], p[1]);
      pen = true;
    }
  }
  function drawGlobe() {
    setG();
    var g = ctx.createRadialGradient(G.cx - G.R * .25, G.cy - G.R * .3, G.R * .1, G.cx, G.cy, G.R);
    g.addColorStop(0, '#0e1727'); g.addColorStop(1, '#070c15');
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(G.cx, G.cy, G.R, 0, 7); ctx.fill();

    // graticule
    var gp = new Path2D(), i, j, pts;
    for (i = -60; i <= 60; i += 30) { pts = []; for (j = -180; j <= 180; j += 5) pts.push([i, j]); polyline(pts, gp); }
    for (i = -180; i < 180; i += 30) { pts = []; for (j = -90; j <= 90; j += 5) pts.push([j, i]); polyline(pts, gp); }
    ctx.strokeStyle = 'rgba(130,170,210,.09)'; ctx.lineWidth = 1; ctx.stroke(gp);

    // geomagnetic latitude circles about both dipole poles
    var poles = [[D.pole[0], D.pole[1]], [-D.pole[0], ((D.pole[1] + 360) % 360) - 180]];
    ctx.setLineDash([2, 4]); ctx.strokeStyle = 'rgba(170,183,203,.32)'; ctx.fillStyle = K.ink3;
    ctx.font = '10px ' + K.mono; ctx.textAlign = 'center';
    [50, 60, 70, 80].forEach(function (m) {
      poles.forEach(function (pl, k) {
        var path = new Path2D(), best = null, bz = -2;
        pts = [];
        for (var b = 0; b <= 360; b += 3) {
          var q = dest(pl[0], pl[1], b, 90 - m); pts.push(q);
          var pp = P(q[0], q[1]); if (pp[2] > bz && b % 30 === 0) { bz = pp[2]; best = pp; }
        }
        polyline(pts, path); ctx.stroke(path);
        if (k === 0 && best && bz > .25) ctx.fillText(m + '°', best[0], best[1] - 4);
      });
    });
    ctx.setLineDash([]);

    // land, shaded by day/night
    var sp = sun(), day = new Path2D(), night = new Path2D();
    land.forEach(function (ring) {
      var prev = null;
      ring.forEach(function (c) {
        var p = P(c[1], c[0]), ok = p[2] > 0;
        if (ok && prev) {
          var f = c[1] * RAD, dl = (c[0] - sp.lon) * RAD;
          var up = Math.sin(f) * Math.sin(sp.lat * RAD) + Math.cos(f) * Math.cos(sp.lat * RAD) * Math.cos(dl) > 0;
          var path = up ? day : night; path.moveTo(prev[0], prev[1]); path.lineTo(p[0], p[1]);
        }
        prev = ok ? p : null;
      });
    });
    ctx.lineWidth = 1; ctx.strokeStyle = 'rgba(150,170,196,.75)'; ctx.stroke(day);
    ctx.strokeStyle = 'rgba(86,104,130,.8)'; ctx.stroke(night);

    // aurora cells: size says "under the oval" (filled) vs faint glow (small), colour says strength
    var cp = (Math.PI / 180) * G.R, big = Math.max(2, cp * .94), sm = Math.max(1.3, cp * .45);
    ctx.globalCompositeOperation = 'lighter';
    for (i = 0; i < cells.length; i++) {
      var c = cells[i], p = P(c[1], c[0]); if (p[2] <= 0.02) continue;
      var s = c[2] >= MIN ? big : sm;
      ctx.fillStyle = auroraRGBA(c[2]); ctx.fillRect(p[0] - s / 2, p[1] - s / 2, s, s);
    }
    ctx.globalCompositeOperation = 'source-over';

    // terminator
    var tp = [], tpath = new Path2D();
    for (var b2 = 0; b2 <= 360; b2 += 3) tp.push(dest(sp.lat, sp.lon, b2, 90));
    polyline(tp, tpath); ctx.setLineDash([6, 5]); ctx.strokeStyle = 'rgba(245,197,66,.28)'; ctx.stroke(tpath); ctx.setLineDash([]);

    if (S.view === 'aviation') drawRoutes(); else drawCities();

    // dipole pole marker
    var pm = P(D.pole[0], D.pole[1]);
    if (pm[2] > 0) {
      ctx.strokeStyle = K.cyan; ctx.lineWidth = 1.4; ctx.beginPath();
      ctx.moveTo(pm[0] - 5, pm[1]); ctx.lineTo(pm[0] + 5, pm[1]); ctx.moveTo(pm[0], pm[1] - 5); ctx.lineTo(pm[0], pm[1] + 5); ctx.stroke();
      label('geomagnetic pole', pm[0] + 8, pm[1] + 3, K.cyan, 'left');
    }
    ctx.strokeStyle = K.line; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(G.cx, G.cy, G.R, 0, 7); ctx.stroke();
  }

  function drawRoutes() {
    var anySel = S.sel !== null, seg;
    for (var pass = 0; pass < 2; pass++) {       // quiet segments first, exposed segments on top
      D.routes.forEach(function (r) {
        var sel = r.id === S.sel, dim = anySel && !sel;
        var path = new Path2D(), pen = false;
        for (var i = 0; i < r.pts.length - 1; i++) {
          var a = r.pts[i], b = r.pts[i + 1], hot = (a[2] !== null && a[2] >= MIN) || (b[2] !== null && b[2] >= MIN);
          if ((pass === 1) !== hot) { pen = false; continue; }
          var pa = P(a[0], a[1]), pb = P(b[0], b[1]);
          if (pa[2] <= 0 || pb[2] <= 0) { pen = false; continue; }
          path.moveTo(pa[0], pa[1]); path.lineTo(pb[0], pb[1]);
        }
        ctx.lineCap = 'round';
        if (pass === 0) { ctx.strokeStyle = dim ? 'rgba(143,163,191,.22)' : 'rgba(170,183,203,.62)'; ctx.lineWidth = sel ? 2.2 : 1.3; }
        else { ctx.strokeStyle = dim ? 'rgba(245,197,66,.35)' : K.amber; ctx.lineWidth = sel ? 3.6 : 2.6; }
        ctx.stroke(path);
      });
    }
    ctx.font = '10px ' + K.mono; ctx.textAlign = 'left';
    var seen = {};
    D.routes.forEach(function (r) {
      [r.a, r.b].forEach(function (code) {
        if (seen[code]) return; seen[code] = 1;
        var ap = D.airports[code], p = P(ap[0], ap[1]); if (p[2] <= 0) return;
        ctx.fillStyle = K.ink; ctx.beginPath(); ctx.arc(p[0], p[1], 2.4, 0, 7); ctx.fill();
        label(code, p[0] + 5, p[1] - 4, K.ink2, 'left');
      });
    });
  }

  function drawCities() {
    var boxes = [];
    function free(x, y, w) {
      for (var i = 0; i < boxes.length; i++) { var b = boxes[i]; if (x < b[0] + b[2] && x + w > b[0] && Math.abs(y - b[1]) < 12) return false; }
      return true;
    }
    ctx.font = '10.5px ' + K.mono;
    D.cities.forEach(function (c) {            // markers first, then labels, so labels never hide a marker
      var p = P(c.lat, c.lon); if (p[2] <= 0.05) return;
      var sel = S.sel === c.name;
      ctx.beginPath(); ctx.arc(p[0], p[1], sel ? 6 : 4.5, 0, 7);
      if (c.inside) { ctx.fillStyle = K.amber; ctx.fill(); } else { ctx.fillStyle = K.bg; ctx.fill(); ctx.strokeStyle = K.ink2; ctx.lineWidth = 1.5; ctx.stroke(); }
      if (sel) { ctx.strokeStyle = K.cyan; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(p[0], p[1], 9, 0, 7); ctx.stroke(); }
    });
    var order = D.cities.slice().sort(function (a, b) { return (S.sel === b.name) - (S.sel === a.name); });
    order.forEach(function (c) {
      var p = P(c.lat, c.lon); if (p[2] <= 0.05) return;
      var w = ctx.measureText(c.name).width, sel = S.sel === c.name;
      if (!sel && !free(p[0] + 9, p[1], w)) return;
      boxes.push([p[0] + 9, p[1], w]); label(c.name, p[0] + 9, p[1] + 3.5, K.ink, 'left', '10.5px ' + K.mono);
    });
  }

  /* ── magnetosphere (3-D wireframe, own projection) ───────────────── */
  function drawMag() {
    var m = D.mag, half = W / 2, s = half * 0.9 / Math.max(m ? m.r0 + 6 : 12, m ? m.r0 * 1.75 : 12), cy = Math.cos(S.yaw * RAD), sy = Math.sin(S.yaw * RAD);
    var cp = Math.cos(S.pitch * RAD), sp = Math.sin(S.pitch * RAD);
    function T(x, y, z) {                        // x to the Sun, z north; returns [px, py, depth]
      var x1 = x * cy - y * sy, y1 = x * sy + y * cy, y2 = y1 * cp - z * sp, z2 = y1 * sp + z * cp;
      return [half + x1 * s, half - z2 * s, y2];
    }
    function hid(p) { var u = (p[0] - half) / s, v = (p[1] - half) / s; return p[2] > 0 && u * u + v * v < 1; }
    function line(pts, style, w, dash) {
      var path = new Path2D(), pen = false;
      pts.forEach(function (q) { var p = T(q[0], q[1], q[2]); if (hid(p)) { pen = false; return; } if (pen) path.lineTo(p[0], p[1]); else path.moveTo(p[0], p[1]); pen = true; });
      ctx.setLineDash(dash || []); ctx.strokeStyle = style; ctx.lineWidth = w || 1; ctx.stroke(path); ctx.setLineDash([]);
    }
    ctx.fillStyle = '#070c15'; ctx.fillRect(0, 0, W, H);
    if (!m) { ctx.fillStyle = K.ink3; ctx.font = '12px ' + K.mono; ctx.textAlign = 'center'; ctx.fillText('Needs solar-wind density, speed and Bz', half, half); return; }
    var i, a, th, pts, r0 = m.r0, al = m.alpha;
    function mp(t) { return r0 * Math.pow(2 / (1 + Math.cos(t)), al); }

    // magnetopause: rings and meridians
    for (var t0 = 25; t0 <= 100; t0 += 25) {
      pts = []; for (a = 0; a <= 360; a += 6) { var r = mp(t0 * RAD); pts.push([r * Math.cos(t0 * RAD), r * Math.sin(t0 * RAD) * Math.cos(a * RAD), r * Math.sin(t0 * RAD) * Math.sin(a * RAD)]); }
      line(pts, 'rgba(34,211,238,.34)', 1);
    }
    for (a = 0; a < 360; a += 30) {
      pts = []; for (th = 0; th <= 100; th += 4) { var rr = mp(th * RAD); pts.push([rr * Math.cos(th * RAD), rr * Math.sin(th * RAD) * Math.cos(a * RAD), rr * Math.sin(th * RAD) * Math.sin(a * RAD)]); }
      line(pts, 'rgba(34,211,238,.34)', 1);
    }
    // dipole shells that fit inside
    [2, 3.5, 5].forEach(function (L) {
      if (L > r0 - 0.4) return;
      var lm = Math.acos(Math.sqrt(1 / L));
      for (a = 0; a < 360; a += 45) {
        pts = []; for (var k = -1; k <= 1.0001; k += 0.04) { var lam = k * lm, rr2 = L * Math.pow(Math.cos(lam), 2);
          pts.push([rr2 * Math.cos(lam) * Math.cos(a * RAD), rr2 * Math.cos(lam) * Math.sin(a * RAD), rr2 * Math.sin(lam)]); }
        line(pts, 'rgba(170,183,203,.22)', 1);
      }
    });
    // orbit rings (equatorial plane); geostationary arc outside the boundary is red
    var ring = function (R, ok) { pts = []; for (a = 0; a <= 360; a += 3) pts.push([R * Math.cos(a * RAD), R * Math.sin(a * RAD), 0]); return pts; };
    line(ring(m.gnss_re), 'rgba(170,183,203,.55)', 1.2, [4, 4]);
    line(ring(m.geo_re), 'rgba(170,183,203,.8)', 1.5);
    if (m.geo_exposed_deg > 0) {
      pts = []; for (a = -m.geo_exposed_deg; a <= m.geo_exposed_deg + 0.01; a += 2) pts.push([m.geo_re * Math.cos(a * RAD), m.geo_re * Math.sin(a * RAD), 0]);
      line(pts, K.red, 3.4);
    }
    // Earth
    var e = T(0, 0, 0), er = s, sx = Math.cos(S.yaw * RAD);
    var eg = ctx.createRadialGradient(e[0] + er * .5 * sx, e[1] - er * .1, er * .1, e[0], e[1], er);
    eg.addColorStop(0, '#46698f'); eg.addColorStop(1, '#10192a');
    ctx.fillStyle = eg; ctx.beginPath(); ctx.arc(e[0], e[1], er, 0, 7); ctx.fill();
    ctx.strokeStyle = 'rgba(170,183,203,.5)'; ctx.lineWidth = 1; ctx.stroke();
    // standoff dimension and solar wind
    line([[1, 0, 0], [r0, 0, 0]], K.ink, 1.2);
    var nose = T(r0, 0, 0), lab = T(r0 / 2 + .5, 0, 0);
    ctx.fillStyle = K.ink; ctx.beginPath(); ctx.arc(nose[0], nose[1], 3, 0, 7); ctx.fill();
    label('nose ' + r0.toFixed(1) + ' Re', lab[0], lab[1] - 9, K.ink, 'center', '11px ' + K.mono);
    [[0, 0], [0, 3], [0, -3], [3, 0], [-3, 0]].forEach(function (o) {
      var a0 = T(r0 + 5.2, o[0], o[1]), a1 = T(r0 + 1.6, o[0], o[1]);
      ctx.strokeStyle = 'rgba(245,197,66,.7)'; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(a0[0], a0[1]); ctx.lineTo(a1[0], a1[1]); ctx.stroke();
      var ang = Math.atan2(a1[1] - a0[1], a1[0] - a0[0]);
      ctx.beginPath(); ctx.moveTo(a1[0], a1[1]); ctx.lineTo(a1[0] - 6 * Math.cos(ang - .4), a1[1] - 6 * Math.sin(ang - .4));
      ctx.moveTo(a1[0], a1[1]); ctx.lineTo(a1[0] - 6 * Math.cos(ang + .4), a1[1] - 6 * Math.sin(ang + .4)); ctx.stroke();
    });
    var sw = T(r0 + 5.2, 0, 3.6);
    label('solar wind' + (D.speed ? ' ' + Math.round(D.speed) + ' km/s' : ''), sw[0], sw[1] - 4, K.amber, 'center', '11px ' + K.mono);
    // ring labels
    var g1 = T(m.geo_re * Math.cos(165 * RAD), m.geo_re * Math.sin(165 * RAD), 0), g2 = T(m.gnss_re * Math.cos(235 * RAD), m.gnss_re * Math.sin(235 * RAD), 0);
    label('geostationary', g1[0], g1[1] - 7, K.ink2, 'center'); label('navigation', g2[0], g2[1] + 14, K.ink2, 'center');
  }

  /* ── frame ───────────────────────────────────────────────────────── */
  function draw() {
    if (!W) return;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0); ctx.clearRect(0, 0, W, H);
    if (S.view === 'satellite') drawMag(); else drawGlobe();
  }
  var raf = 0;
  function redraw() { if (!raf) raf = requestAnimationFrame(function () { raf = 0; draw(); }); }
  function ease(to, ms, done) {
    var from = { lat: S.lat, lon: S.lon, yaw: S.yaw, pitch: S.pitch };
    if (reduce || !ms) { for (var k in to) S[k] = to[k]; draw(); if (done) done(); return; }
    var t0 = performance.now();
    (function step(now) {
      var t = Math.min(1, (now - t0) / ms), e = 1 - Math.pow(1 - t, 3);
      for (var k in to) { var dl = to[k] - from[k]; if (k === 'lon') dl = ((dl + 540) % 360) - 180; S[k] = from[k] + dl * e; }
      draw(); if (t < 1) requestAnimationFrame(step); else if (done) done();
    })(performance.now());
  }

  /* ── drag, hover ─────────────────────────────────────────────────── */
  var drag = null;
  cv.addEventListener('pointerdown', function (e) { drag = { x: e.clientX, y: e.clientY, t: e.pointerType }; cv.setPointerCapture(e.pointerId); cv.classList.add('drag'); });
  cv.addEventListener('pointerup', function () { drag = null; cv.classList.remove('drag'); });
  cv.addEventListener('pointercancel', function () { drag = null; cv.classList.remove('drag'); });
  cv.addEventListener('dblclick', function () { resetView(); });
  cv.addEventListener('pointermove', function (e) {
    if (drag) {
      var dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
      if (S.view === 'satellite') { S.yaw += dx * .4; if (drag.t !== 'touch') S.pitch = Math.max(-80, Math.min(80, S.pitch + dy * .3)); }
      else { S.lon -= dx * 180 / (Math.PI * G.R) * 1.1; if (drag.t !== 'touch') S.lat = Math.max(-90, Math.min(90, S.lat + dy * 180 / (Math.PI * G.R) * 1.1)); }
      redraw(); return;
    }
    if (S.view === 'satellite') return;
    var r = cv.getBoundingClientRect(), x = (e.clientX - r.left) / r.width * W - G.cx, y = G.cy - (e.clientY - r.top) / r.height * H;
    var rho = Math.sqrt(x * x + y * y) / G.R;
    if (rho > 1) { $('gl-read').textContent = 'Drag to rotate'; return; }
    var c = Math.asin(rho), sc = Math.sin(c), cc = Math.cos(c), f0 = S.lat * RAD, xn = x / G.R, yn = y / G.R;
    var lat = rho === 0 ? S.lat : Math.asin(cc * Math.sin(f0) + yn * sc * Math.cos(f0) / rho) / RAD;
    var lon = S.lon + (rho === 0 ? 0 : Math.atan2(xn * sc, rho * Math.cos(f0) * cc - yn * Math.sin(f0) * sc) / RAD);
    lon = ((lon + 540) % 360) - 180;
    var cell = lookup(Math.round(lon), Math.round(lat));
    $('gl-read').textContent = Math.abs(lat).toFixed(1) + '°' + (lat >= 0 ? 'N ' : 'S ') + Math.abs(lon).toFixed(1) + '°' + (lon >= 0 ? 'E' : 'W') +
      '  ·  geomagnetic ' + magLat(lat, lon).toFixed(1) + '°  ·  aurora ' + (D.aurora ? cell : 'n/a');
  });
  var idx = null;
  function lookup(lon, lat) {
    if (!idx) { idx = {}; cells.forEach(function (c) { idx[c[0] + ',' + c[1]] = c[2]; }); }
    lon = ((lon + 540) % 360) - 180; return idx[lon + ',' + lat] || 0;
  }
  cv.addEventListener('pointerleave', function () { if (S.view !== 'satellite') $('gl-read').textContent = 'Drag to rotate · double-click to reset'; });

  /* ── side panel ──────────────────────────────────────────────────── */
  function el(tag, cls, html) { var n = document.createElement(tag); if (cls) n.className = cls; if (html != null) n.innerHTML = html; return n; }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function legend(items) {
    var ul = $('geo-legend'); ul.innerHTML = '';
    items.forEach(function (it) { ul.appendChild(el('li', '', '<i style="' + it[0] + '"></i>' + esc(it[1]))); });
  }
  function rows() {
    var ul = $('geo-rows'); ul.innerHTML = '';
    if (S.view === 'aviation') {
      D.routes.forEach(function (r) {
        var b = el('button', 'geo-row'); b.type = 'button'; b.setAttribute('aria-pressed', S.sel === r.id ? 'true' : 'false');
        var strip = '', n = 36;
        for (var i = 0; i < n; i++) {
          var lo = Math.floor(i * (r.pts.length - 1) / n), hi = Math.max(lo + 1, Math.floor((i + 1) * (r.pts.length - 1) / n)), on = false;
          for (var j = lo; j <= hi && j < r.pts.length; j++) if (r.pts[j][2] !== null && r.pts[j][2] >= MIN) on = true;
          strip += '<i class="' + (on ? 'on' : '') + '"></i>';
        }
        b.innerHTML = '<b>' + r.a + ' → ' + r.b + '</b><span class="strip" title="Along the route: amber where it is under the oval">' + strip + '</span>' +
          '<span class="num">' + (r.under_pct === null ? 'n/a' : '<em>' + r.under_pct + '%</em> under oval') + ' · max ' + r.max_mlat.toFixed(0) + '°</span>';
        b.addEventListener('click', function () { S.sel = S.sel === r.id ? null : r.id; if (S.sel) { var mid = r.pts[Math.floor(r.pts.length / 2)]; ease({ lat: Math.max(35, Math.min(80, mid[0] + 10)), lon: mid[1] }, 500); } rows(); draw(); });
        var li = el('li'); li.appendChild(b); ul.appendChild(li);
      });
    } else if (S.view === 'power_grid') {
      D.cities.forEach(function (c) {
        var b = el('button', 'geo-row'); b.type = 'button'; b.setAttribute('aria-pressed', S.sel === c.name ? 'true' : 'false');
        var state = c.inside === null ? 'n/a' : c.inside ? '<em>under oval</em>' : (c.edge_deg === null ? 'outside' : c.edge_deg.toFixed(0) + '° from oval');
        var prox = c.inside === null ? 0 : c.inside ? 100 : (c.edge_deg === null ? 0 : Math.round(100 * Math.max(0, 1 - c.edge_deg / 30)));
        b.innerHTML = '<b>' + esc(c.name) + '</b><span class="prox' + (c.inside ? ' in' : '') + '" title="Closeness to the oval edge: full bar = under the oval, empty = 30\u00b0 or more away"><i style="width:' + prox + '%"></i></span><span class="num">' + state + ' · geomag ' + Math.abs(c.mlat).toFixed(0) + '°</span>';
        b.addEventListener('click', function () { S.sel = S.sel === c.name ? null : c.name; if (S.sel) ease({ lat: Math.max(-70, Math.min(70, c.lat)), lon: c.lon }, 500); rows(); draw(); });
        var li = el('li'); li.appendChild(b); ul.appendChild(li);
      });
    } else {
      D.sat_rows.forEach(function (r) {
        var b = el('div', 'geo-row flat', '<b>' + esc(r.name) + '<br><span style="color:var(--ink-3);font-weight:400">' + esc(r.re) + '</span></b><span class="num">' + esc(r.detail) + '</span>');
        var li = el('li'); li.appendChild(b); ul.appendChild(li);
      });
    }
  }
  function resetView() {
    if (S.view === 'satellite') ease({ yaw: 28, pitch: 24 }, 400);
    else { S.sel = null; var c = CENTER[S.view]; ease({ lat: c[0], lon: c[1] }, 450); rows(); }
  }
  function show(v, first) {
    S.view = v; S.sel = null;
    try { sessionStorage.setItem('geo.view', v); } catch (e) {}
    Array.prototype.forEach.call(root.querySelectorAll('[role=tab]'), function (t) { t.setAttribute('aria-selected', t.dataset.v === v ? 'true' : 'false'); });
    $('geo-head').textContent = D.headline[v]; $('geo-note').textContent = D.notes[v];
    cv.setAttribute('aria-label', 'Interactive ' + (v === 'satellite' ? '3-D magnetosphere' : 'globe') + '. ' + D.headline[v]);
    $('gl-read').textContent = v === 'satellite' ? 'Drag to rotate · distances in Earth radii' : 'Drag to rotate · double-click to reset';
    var amb = 'width:14px;height:3px;border-radius:2px;background:' + K.amber;
    if (v === 'aviation') legend([['width:7px;height:7px;background:rgba(34,211,238,.9)', 'under the oval (' + MIN + '+)'], ['width:4px;height:4px;background:rgba(18,120,150,.9)', 'faint glow'], [amb, 'route under the oval'], ['width:14px;height:2px;background:rgba(170,183,203,.7)', 'route elsewhere']]);
    else if (v === 'power_grid') legend([['width:7px;height:7px;background:rgba(34,211,238,.9)', 'under the oval (' + MIN + '+)'], ['width:9px;height:9px;border-radius:50%;background:' + K.amber, 'city under the oval'], ['width:9px;height:9px;border-radius:50%;border:1.5px solid ' + K.ink2, 'city outside'], ['width:14px;border-top:1px dashed ' + K.ink3, 'geomagnetic latitude']]);
    else legend([['width:14px;height:2px;background:rgba(34,211,238,.7)', 'magnetopause'], ['width:14px;height:2px;background:rgba(170,183,203,.6)', 'dipole field line'], ['width:14px;height:3px;background:' + K.red, 'orbit outside the boundary'], ['width:14px;height:3px;background:' + K.amber, 'solar wind']]);
    rows();
    if (v !== 'satellite') { var c = CENTER[v]; ease({ lat: c[0], lon: first && !reduce ? c[1] + 55 : c[1] }, 0); if (first && !reduce) ease({ lat: c[0], lon: c[1] }, 1300); else draw(); }
    else { S.yaw = 28; S.pitch = 24; draw(); }
  }
  var tabs = root.querySelectorAll('[role=tab]');
  Array.prototype.forEach.call(tabs, function (t, i) {
    t.addEventListener('click', function () { show(t.dataset.v, false); });
    t.addEventListener('keydown', function (e) {
      var n = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0; if (!n) return;
      var nt = tabs[(i + n + tabs.length) % tabs.length]; nt.focus(); show(nt.dataset.v, false);
    });
  });
  resize(); show(S.view, true);
})();
</script>
"""
