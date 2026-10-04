"""Run on YOUR machine:  python check_sources.py   -> paste the whole output back.
Tests every data source AstroRisk uses (read-only, nothing is written)."""
import sys, time
import requests, pandas as pd
import app

def t(label, fn):
    t0 = time.time()
    try:
        out = fn()
        print(f"[ OK ] {label}: {out}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"[FAIL] {label}: {str(e)[:200]}")

print("== NOAA SWPC live feeds ==")
for name, (paths, merge) in app.ENDPOINTS.items():
    for p in paths:
        t(f"{name:8s} {p}", lambda p=p: f"{len(requests.get(app.SWPC+p, timeout=25, headers=app.UA).json()):,} rows")

print("\n== NASA OMNI2 yearly files (training data) ==")
for url in app.OMNI_URLS:
    t(url.format(year=2023), lambda url=url: (lambda d: f"{len(d):,} rows, speed median {d.speed.median():.0f}, Dst valid {d.dst.notna().mean():.0%}, p10 valid {d.p10.notna().mean():.0%}")(
        app.parse_omni_text(requests.get(url.format(year=2023), timeout=120, headers=app.UA).text)))

print("\n== OMNIWeb CGI fallback (2023) ==")
t("nx1.cgi", lambda: f"{len(app.fetch_omniweb_cgi(2023)):,} rows (sanity-checked)")
try:
    r = requests.get(app.OMNIWEB_CGI, params=[("activity","retrieve"),("res","hour"),("spacecraft","omni2"),
        ("start_date","20230101"),("end_date","20230102"),("vars","22"),("vars","36"),("scale","Linear"),("view","0"),("table","0")], timeout=60)
    print("CGI raw head (send this if the CGI line failed):\n", r.text[:700])
except Exception as e:
    print("CGI raw: ", e)

print("\n== GFZ Potsdam Kp ==")
now = pd.Timestamp.now(tz="UTC")
t("GFZ Kp last 3 days", lambda: (lambda s: f"{len(s)} values, latest {s.iloc[-1]} at {s.index[-1]}")(app.fetch_gfz_kp(now-pd.Timedelta(days=3), now)))
for u in app.GFZ_URLS:
    try:
        r = requests.get(u.format(start=(now-pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), end=now.strftime("%Y-%m-%dT%H:%M:%SZ")), timeout=15)
        print("GFZ raw", r.status_code, r.text[:300])
    except Exception as e:
        print("GFZ raw", e)