"""
AstroRisk - space weather risk operations dashboard (single file)

Every number on the page is observed or derived from observations:
  * live inputs   : NOAA SWPC public JSON (Kp, GOES X-ray / proton / electron,
                    DSCOVR/ACE/SOLAR-1 solar wind). GFZ Potsdam Kp is a fallback
                    and an independent cross-check.
  * training data : NASA OMNI2 hourly record (spdf file, OMNIWeb CGI as fallback).
                    If neither can be downloaded NOTHING is trained and the app
                    says so. There is no synthetic data path anywhere.
  * forecast      : SpaceWeatherLSTM, P(Kp >= 5 within ~9 h). Shown as VALIDATED
                    only if it beats simple baselines on a held-out period.
  * sectors       : level = NOAA-scale nowcast of the live observations (G / S / R
                    scales). The 9 h outlook = Random Forests trained on what
                    actually happened next in OMNI2 (Kp, Dst, AE, proton flux).
No API keys, no paid services.

Run:  streamlit run app.py
"""
from __future__ import annotations

import html
import json
import math
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
import streamlit as st

import geoviz

warnings.filterwarnings("ignore")
torch.set_num_threads(2)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BUILD = "2026-10-04.12"
MODEL_VERSION = 7
WEIGHTS_DIR = Path("weights")
LSTM_PATH = WEIGHTS_DIR / "storm_lstm_v4.pt"
SECTOR_PATH = WEIGHTS_DIR / "sector_rf_v4.joblib"
META_PATH = WEIGHTS_DIR / "meta_v4.json"
LOG_PATH = WEIGHTS_DIR / "train_log.txt"
CACHE_PATH = Path("cache/live_data.json")
WEIGHTS_DIR.mkdir(exist_ok=True)
CACHE_PATH.parent.mkdir(exist_ok=True)

REFRESH_INTERVAL = 300            # seconds between live pulls
RETRAIN_DAYS = 60
TRAIN_YEARS = list(range(2017, 2025))
MIN_TRAIN_YEARS = 4
SEQ_LEN = 8                       # 8 blocks x 3 h = 24 h input window
BLOCK_H = 3
FORECAST_STEPS = 3                # next 3 blocks = ~9 h
STORM_KP = 5.0                    # G1 threshold
MIN_COVERAGE = 0.75               # share of window blocks that must be really observed

FEATURES = ["kp_norm", "wind_speed_norm", "bz_norm", "density_norm", "temperature_norm"]

SWPC = "https://services.swpc.noaa.gov"
OMNI_URLS = [
    "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat",
    "https://omniweb.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat",
]
OMNIWEB_CGI = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"
GFZ_URLS = [
    "https://kp.gfz-potsdam.de/app/json/?start={start}&end={end}&index=Kp",
    "https://kp.gfz.de/app/json/?start={start}&end={end}&index=Kp",
]
LLM_URL = "https://api.groq.com/openai/v1/chat/completions"     # free tier, OpenAI-compatible
LLM_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
LLM_FALLBACK = os.environ.get("GROQ_FALLBACK_MODEL", "llama-3.3-70b-versatile")   # tried if the first gives nothing usable
UA = {"User-Agent": "AstroRisk/4.0 (+student project)"}


class TrainingDataUnavailable(RuntimeError):
    """Raised instead of ever substituting made-up training data."""


# ─────────────────────────────────────────────────────────────────────────────
# 1. NORMALISATION  (scalar and array safe)
# ─────────────────────────────────────────────────────────────────────────────
def _c01(x):
    return np.clip(x, 0.0, 1.0)


def _log(v):
    return math.log10(v) if v and v > 0 else None


def n_kp(v):       return _c01(np.asarray(v, dtype=float) / 9.0)
def n_speed(v):    return _c01((np.asarray(v, dtype=float) - 300.0) / 700.0)
def n_bz(v):       return _c01((np.asarray(v, dtype=float) + 30.0) / 60.0)
def n_bz_south(v): return _c01(-np.asarray(v, dtype=float) / 20.0)
def n_density(v):  return _c01((np.asarray(v, dtype=float) - 1.0) / 99.0)
def n_temp(v):     return _c01((np.asarray(v, dtype=float) - 1e4) / (2e6 - 1e4))


def n_xray(v):      # 1e-7 (quiet) .. 1e-3 (X10+); M1 = 0.50, X1 = 0.75
    l = _log(v)
    return 0.0 if l is None else float(_c01((l + 7.0) / 4.0))


def n_proton(v):    # 0.1 pfu .. 1e4 pfu; S1 (10 pfu) = 0.40
    l = _log(v)
    return 0.0 if l is None else float(_c01((l + 1.0) / 5.0))


def n_electron(v):  # 10 .. ~3e4; 1000 = 0.57
    l = _log(v)
    return 0.0 if l is None else float(_c01((l - 1.0) / 3.5))


def dyn_pressure_npa(density: float, speed: float) -> float:
    return 1.6726e-6 * density * speed * speed


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAINING DATA  (NASA OMNI2 hourly; GFZ Kp cross-check)
# ─────────────────────────────────────────────────────────────────────────────
# 0-based word index in omni2_YYYY.dat, fill value that means "missing"
OMNI_COLS = {"bz": (16, 999.9), "temp": (22, 9999999.0), "density": (23, 999.9),
             "speed": (24, 9999.0), "kp10": (38, 99.0), "dst": (40, 99999.0),
             "ae": (41, 9999.0), "p10": (45, 99999.99)}
OMNI_MIN_WORDS = 46


def parse_omni_text(text: str) -> pd.DataFrame:
    """
    omni2_YYYY.dat: whitespace separated, one row per hour. 1-based words:
    1 year, 2 day-of-year, 3 hour, 17 Bz GSM, 23 proton temperature, 24 proton
    density, 25 flow speed, 39 Kp x10, 41 Dst, 42 AE, 46 proton flux >10 MeV.
    Fill values become NaN - they are never replaced by invented numbers.
    """
    rows = []
    for line in text.splitlines():
        p = line.split()
        if len(p) < OMNI_MIN_WORDS:
            continue
        try:
            rows.append([int(p[0]), int(p[1]), int(p[2])] + [float(p[i]) for i, _ in OMNI_COLS.values()])
        except ValueError:
            continue
    if not rows:
        return pd.DataFrame()
    return _omni_frame(pd.DataFrame(rows, columns=["year", "doy", "hour"] + list(OMNI_COLS)))


def _omni_frame(df: pd.DataFrame) -> pd.DataFrame:
    ts = (pd.to_datetime(df["year"].astype(int).astype(str), format="%Y", utc=True)
          + pd.to_timedelta(df["doy"] - 1, unit="D") + pd.to_timedelta(df["hour"], unit="h"))
    out = pd.DataFrame(index=ts)
    for name, (_, fill) in OMNI_COLS.items():
        col = df[name].to_numpy(dtype=float)
        out[name] = np.where(col >= fill * 0.999, np.nan, col)
    out["kp"] = out.pop("kp10") / 10.0
    return out[~out.index.duplicated()].sort_index()


def fetch_omni_year(year: int) -> pd.DataFrame:
    last = None
    for url in OMNI_URLS:
        try:
            r = requests.get(url.format(year=year), timeout=120, headers=UA)
            r.raise_for_status()
            df = parse_omni_text(r.text)
            if len(df):
                return df
            last = RuntimeError("file downloaded but contained no rows")
        except Exception as exc:
            last = exc
    raise last or RuntimeError("no OMNI2 mirror answered")


def fetch_omniweb_cgi(year: int) -> pd.DataFrame:
    """
    Fallback: same OMNI2 hourly record through the OMNIWeb CGI. CGI variable ids
    are the file's word number minus 1 (checked against one real CGI answer:
    id 22 = proton temperature, id 36 = plasma beta). The result is also
    range-checked and rejected if the columns do not look right.
    """
    ids = [w - 1 for w in (17, 23, 24, 25, 39, 41, 42, 46)]
    params = [("activity", "retrieve"), ("res", "hour"), ("spacecraft", "omni2"),
              ("start_date", f"{year}0101"), ("end_date", f"{year}1231"), ("maxdays", "366")]
    params += [("vars", str(i)) for i in ids] + [("scale", "Linear"), ("view", "0"), ("table", "0")]
    r = requests.get(OMNIWEB_CGI, params=params, timeout=180, headers=UA)
    r.raise_for_status()
    rows = []
    for line in r.text.splitlines():
        p = line.split()
        if len(p) == 3 + len(ids) and p[0].isdigit() and len(p[0]) == 4:
            try:
                rows.append([int(p[0]), int(p[1]), int(p[2])] + [float(x) for x in p[3:]])
            except ValueError:
                continue
    if not rows:
        raise RuntimeError("OMNIWeb CGI returned no parsable rows")
    df = _omni_frame(pd.DataFrame(rows, columns=["year", "doy", "hour"] + list(OMNI_COLS)))
    spd = df["speed"].dropna()
    if len(spd) < 100 or not (250 <= spd.median() <= 700):
        raise RuntimeError("OMNIWeb CGI columns failed the sanity check (speed median "
                           f"{spd.median() if len(spd) else 'n/a'}) - variable ids are wrong")
    return df


def fetch_gfz_kp(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """GFZ Potsdam Kp (definitive / nowcast). Returns a UTC series indexed by block start."""
    last = None
    for url in GFZ_URLS:
        try:
            r = requests.get(url.format(start=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                        end=end.strftime("%Y-%m-%dT%H:%M:%SZ")),
                             timeout=15, headers=UA)
            r.raise_for_status()
            j = r.json()
            s = pd.Series(pd.to_numeric(pd.Series(j["Kp"]), errors="coerce").values,
                          index=pd.to_datetime(j["datetime"], utc=True)).dropna()
            if len(s):
                return s[~s.index.duplicated()].sort_index()
            last = RuntimeError("empty answer")
        except Exception as exc:
            last = exc
    raise last or RuntimeError("GFZ unreachable")


def gfz_crosscheck(noaa_kp: pd.Series, now: pd.Timestamp) -> Optional[Dict]:
    """Compare NOAA's real-time Kp against GFZ's for the same blocks (last 3 days)."""
    try:
        g = fetch_gfz_kp(now - pd.Timedelta(days=3), now)
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:120]}
    j = pd.concat([noaa_kp.rename("noaa"), g.rename("gfz")], axis=1, join="inner").dropna()
    if len(j) < 4:
        return {"ok": False, "error": "no overlapping blocks"}
    d = (j["noaa"] - j["gfz"]).abs()
    return {"ok": True, "n": int(len(j)), "mean_abs_diff": round(float(d.mean()), 2),
            "within_1": round(float((d <= 1.0).mean()), 2)}


BLOCK_AGG = {"kp": "max", "bz": "mean", "speed": "mean", "density": "mean", "temp": "mean",
             "dst": "min", "ae": "max", "p10": "max"}


def to_blocks(hourly: pd.DataFrame) -> pd.DataFrame:
    """3-hour blocks; a block needs >= 2 real hourly values, otherwise it stays NaN."""
    out = {}
    for c, how in BLOCK_AGG.items():
        r = hourly[c].resample("3h")
        v = getattr(r, how)()
        out[c] = v.where(r.count() >= 2)
    return pd.DataFrame(out)


def load_training_frame(years: List[int], log: Callable[[str], None]) -> Tuple[pd.DataFrame, str, List[int]]:
    def one(y):
        try:
            return y, fetch_omni_year(y), "omni2 file"
        except Exception as exc1:
            try:
                return y, fetch_omniweb_cgi(y), "OMNIWeb CGI"
            except Exception as exc2:
                return y, RuntimeError(f"file: {str(exc1)[:90]} | cgi: {str(exc2)[:90]}"), ""

    got, ok_years, via = [], [], set()
    with ThreadPoolExecutor(max_workers=4) as ex:
        for y, res, how in ex.map(one, years):
            if isinstance(res, pd.DataFrame) and len(res):
                got.append(res)
                ok_years.append(y)
                via.add(how)
                log(f"OMNI2 {y}: {len(res):,} hourly rows via {how}")
            else:
                log(f"OMNI2 {y}: FAILED - {res}")
    if len(ok_years) < MIN_TRAIN_YEARS:
        raise TrainingDataUnavailable(
            f"only {len(ok_years)} of {len(years)} OMNI2 years could be downloaded "
            f"(need {MIN_TRAIN_YEARS}). No model was trained and no substitute data is used.")
    return to_blocks(pd.concat(got).sort_index()), " + ".join(sorted(via)), sorted(ok_years)


def make_windows(df3h: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """X [N, SEQ_LEN, 5], y [N] (Kp>=5 in next 3 blocks), cur_kp [N], pos [N] (block index)."""
    g = df3h.asfreq("3h").copy()
    for c in ("bz", "speed", "density", "temp"):        # gaps of <= 6 h only, inside real data
        g[c] = g[c].interpolate(limit=2, limit_area="inside")
    feats = np.column_stack([n_kp(g["kp"].values), n_speed(g["speed"].values),
                             n_bz(g["bz"].values), n_density(g["density"].values),
                             n_temp(g["temp"].values)]).astype(np.float32)
    kp = g["kp"].values
    valid = np.isfinite(feats).all(axis=1) & np.isfinite(kp)
    cs = np.concatenate([[0], np.cumsum(~valid)])
    X, y, cur, pos = [], [], [], []
    for i in range(SEQ_LEN - 1, len(g) - FORECAST_STEPS):
        if cs[i + FORECAST_STEPS + 1] - cs[i - SEQ_LEN + 1] != 0:
            continue
        X.append(feats[i - SEQ_LEN + 1: i + 1])
        y.append(float(kp[i + 1: i + 1 + FORECAST_STEPS].max() >= STORM_KP))
        cur.append(kp[i])
        pos.append(i)
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32),
            np.asarray(cur, dtype=np.float32), np.asarray(pos))


def _dist_to_sorted(p: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Distance from every p to the nearest value of the sorted array ref."""
    if len(ref) == 0:
        return np.full(len(p), 10 ** 9)
    k = np.searchsorted(ref, p)
    left = np.abs(p - ref[np.clip(k - 1, 0, len(ref) - 1)])
    right = np.abs(p - ref[np.clip(k, 0, len(ref) - 1)])
    return np.minimum(left, right)


def split_windows(pos: np.ndarray) -> Dict[str, np.ndarray]:
    """
    test       : last 15 % of the record (a period the model never sees)
    validation : every 6th week of the rest, spread over all years (early stopping
                 sees the same mix of solar-cycle phases as training)
    train      : everything else, purged around validation weeks and the test start
    """
    gap = SEQ_LEN + FORECAST_STEPS
    cut = pos[int(len(pos) * 0.85)]
    pool = pos <= cut
    test = pos > cut + gap
    val = pool & ((pos // 56) % 6 == 3)
    train = pool & ~val & (_dist_to_sorted(pos, pos[val]) > gap) & (pos < cut - gap)
    return {"train": train, "val": val, "test": test}


# ─────────────────────────────────────────────────────────────────────────────
# 3. LSTM
# ─────────────────────────────────────────────────────────────────────────────
class SpaceWeatherLSTM(nn.Module):
    """2-layer LSTM + temporal attention + linear skip from the latest block.
    forward() returns (logit, attention)."""

    def __init__(self, input_dim: int = len(FEATURES), hidden: int = 64,
                 layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, dropout=dropout)
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(32, 1))
        self.skip = nn.Linear(input_dim, 1)             # persistence-style path

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x)
        a = torch.softmax(self.attn(out), dim=1)
        ctx = (out * a).sum(dim=1)
        return (self.head(ctx) + self.skip(x[:, -1, :])).squeeze(-1), a.squeeze(-1)


def _probs(model: SpaceWeatherLSTM, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logit, _ = model(torch.from_numpy(X))
    return torch.sigmoid(logit).numpy()


def _safe_auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def train_lstm(frame: pd.DataFrame, log: Callable[[str], None] = print) -> Tuple[SpaceWeatherLSTM, Dict]:
    torch.manual_seed(7)
    np.random.seed(7)
    X, y, cur, pos = make_windows(frame)
    if len(X) < 2000:
        raise TrainingDataUnavailable(f"only {len(X)} usable windows after removing gaps")
    sp = split_windows(pos)
    X_tr, y_tr = X[sp["train"]], y[sp["train"]]
    X_va, y_va = X[sp["val"]], y[sp["val"]]
    X_te, y_te, cur_te = X[sp["test"]], y[sp["test"]], cur[sp["test"]]
    log(f"{len(X):,} windows | train {len(X_tr):,} ({y_tr.mean() * 100:.1f}% pos) | "
        f"val {len(X_va):,} ({y_va.mean() * 100:.1f}%) | test {len(X_te):,} ({y_te.mean() * 100:.1f}%)")

    model = SpaceWeatherLSTM()
    loss_fn = nn.BCEWithLogitsLoss()                       # plain BCE: calibrated probabilities
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

    best, best_state, wait = float("inf"), None, 0
    for epoch in range(1, 61):
        model.train()
        idx = np.random.permutation(len(X_tr))
        tl = 0.0
        for k in range(0, len(idx), 256):
            j = idx[k:k + 256]
            xb, yb = torch.from_numpy(X_tr[j]), torch.from_numpy(y_tr[j])
            opt.zero_grad()
            loss = loss_fn(model(xb)[0], yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(j)
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(torch.from_numpy(X_va))[0], torch.from_numpy(y_va)).item()
        sched.step(vl)
        if vl < best - 1e-5:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if epoch == 1 or epoch % 5 == 0:
            log(f"epoch {epoch:2d}  train {tl / len(X_tr):.4f}  val {vl:.4f}  "
                f"val-AUC {_safe_auc(y_va, _probs(model, X_va))}")
        if wait >= 8:
            log(f"early stop at epoch {epoch}")
            break
    model.load_state_dict(best_state)
    model.eval()

    p_te = _probs(model, X_te)
    onset = cur_te < STORM_KP
    flat = lambda a: a.reshape(len(a), -1)
    auc_lr = None
    try:
        lr = LogisticRegression(max_iter=600).fit(flat(X_tr), y_tr)
        auc_lr = _safe_auc(y_te, lr.predict_proba(flat(X_te))[:, 1])
    except Exception as exc:
        log(f"logistic baseline failed: {exc}")
    brier = float(brier_score_loss(y_te, p_te))
    brier_clim = float(np.mean((y_te - y_tr.mean()) ** 2))
    auc, auc_p = _safe_auc(y_te, p_te), _safe_auc(y_te, X_te[:, -1, 0])
    validated = bool(auc is not None and auc >= 0.75 and auc >= (auc_p or 0) - 0.03 and brier < brier_clim)
    meta = {
        "n_windows": int(len(X)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "base_rate_train": float(y_tr.mean()), "base_rate_test": float(y_te.mean()),
        "test_mean_pred": float(p_te.mean()),
        "auc": auc, "auc_onset": _safe_auc(y_te[onset], p_te[onset]),
        "auc_persistence": auc_p,
        "auc_persistence_onset": _safe_auc(y_te[onset], X_te[onset][:, -1, 0]),
        "auc_logistic": auc_lr, "brier": brier, "brier_climatology": brier_clim,
        "validated": validated,
        "baseline": np.median(X_tr[:, -1, :], axis=0).astype(float).tolist(),
    }
    torch.save(model.state_dict(), LSTM_PATH)
    log(f"TEST ROC-AUC {auc}  persistence {auc_p}  logistic {auc_lr}  | onset-only {meta['auc_onset']} "
        f"(persistence {meta['auc_persistence_onset']})  | brier {brier:.4f} vs climatology {brier_clim:.4f}  "
        f"=> {'VALIDATED' if validated else 'NOT validated'}")
    return model, meta


# ─────────────────────────────────────────────────────────────────────────────
# 4. SECTOR OUTLOOK  (Random Forests on real next-9-hour outcomes)
# ─────────────────────────────────────────────────────────────────────────────
def _future(s: pd.Series, how: str) -> pd.Series:
    """max / min over the next FORECAST_STEPS blocks (i+1 .. i+3)."""
    r = s[::-1].shift(1).rolling(FORECAST_STEPS, min_periods=2)
    return getattr(r, how)()[::-1]


def sector_labels(f: pd.DataFrame) -> Dict[str, Dict[str, pd.Series]]:
    """
    What actually happened in the next ~9 h, per sector. Only variables with a full
    OMNI2 record are used (Kp, Dst, AE). The OMNI2 proton column is observed for 2017-2019
    only (one radiation storm), too thin to learn from, so radiation risk is covered by the
    live S-scale nowcast, not by a model.
      power grid : elevated Kp>=5 or Dst<=-50 | severe Kp>=7 or Dst<=-100
      satellite  : elevated AE>=1000 (substorm charging) or Dst<=-50 | severe Dst<=-100
      aviation   : elevated Kp>=5 (GNSS / HF disturbance)            | severe Kp>=7
    A block is labelled only if every variable behind its rule was observed.
    """
    kp, dst, ae = _future(f["kp"], "max"), _future(f["dst"], "min"), _future(f["ae"], "max")

    def lab(conds) -> pd.Series:
        hit = np.zeros(len(f), bool)
        known = np.ones(len(f), bool)
        for c, v in conds:
            hit |= c.fillna(False).to_numpy(bool)
            known &= v.notna().to_numpy()                       # symmetric: unknown is never "no event"
        return pd.Series(np.where(~known, np.nan, np.where(hit, 1.0, 0.0)), index=f.index)

    return {
        "power_grid": {"elev": lab([(kp >= 5, kp), (dst <= -50, dst)]),
                       "severe": lab([(kp >= 7, kp), (dst <= -100, dst)])},
        "satellite": {"elev": lab([(ae >= 1000, ae), (dst <= -50, dst)]),
                      "severe": lab([(dst <= -100, dst)])},
        "aviation": {"elev": lab([(kp >= 5, kp)]),
                     "severe": lab([(kp >= 7, kp)])},
    }


def rf_matrix(f: pd.DataFrame, use_proton: bool) -> Tuple[pd.DataFrame, List[str]]:
    """Features known at the end of block i (same definitions as live_rf_features)."""
    X = pd.DataFrame(index=f.index)
    X["kp"] = f["kp"]
    X["kp_max24"] = f["kp"].rolling(8, min_periods=6).max()
    X["speed"] = f["speed"]
    X["bz_south"] = (-f["bz"]).clip(lower=0)
    X["density"] = f["density"]
    X["temp_log"] = np.log10(f["temp"].where(f["temp"] > 0))
    X["pressure"] = dyn_pressure_npa(f["density"], f["speed"])
    if use_proton:
        X["proton_log"] = np.log10(f["p10"].clip(lower=0.01))
    return X, list(X.columns)


def train_sector_models(frame: pd.DataFrame, log: Callable[[str], None] = print) -> Tuple[Dict, Dict]:
    f = frame.asfreq("3h")
    use_proton = bool(f["p10"].notna().mean() >= 0.6)
    X, feats = rf_matrix(f, use_proton)
    labels = sector_labels(f)
    ok_x = X.notna().all(axis=1).to_numpy()
    log(f"sector RF: {int(ok_x.sum()):,} usable blocks, features {feats}")
    models, meta = {}, {"features": feats, "sectors": {}}
    for sec, tg in labels.items():
        models[sec], meta["sectors"][sec] = {}, {}
        for lvl, y in tg.items():
            yv = y.to_numpy()
            m = ok_x & np.isfinite(yv)
            at = np.flatnonzero(m)
            cut = at[int(len(at) * 0.8)] if len(at) else 0   # last 20 % of the rows that HAVE this label
            tr = m.copy(); tr[cut - 12:] = False
            te = m.copy(); te[:cut] = False
            mk = lambda: RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=20,
                                                random_state=42, n_jobs=1)
            auc, kp_auc = None, None
            if tr.sum() > 500 and len(np.unique(yv[tr])) == 2:
                rf = mk().fit(X.to_numpy()[tr], yv[tr])
                if te.sum() > 100:
                    pt = rf.predict_proba(X.to_numpy()[te])[:, 1]
                    auc = _safe_auc(yv[te], pt)
                    kp_auc = _safe_auc(yv[te], X["kp"].to_numpy()[te])
                    bins = np.digitize(pt, [0.05, 0.2, 0.5])
                    cal = " | ".join(f"{lo}-{hi}: pred {pt[bins == b].mean() * 100:.0f}% obs {yv[te][bins == b].mean() * 100:.0f}% (n={int((bins == b).sum())})"
                                     for b, (lo, hi) in enumerate([("0", ".05"), (".05", ".2"), (".2", ".5"), (".5", "1")])
                                     if (bins == b).sum() >= 20)
                    log(f"  {sec}/{lvl}: Kp-alone AUC {kp_auc}  |  calibration {cal}")
            if m.sum() < 500 or len(np.unique(yv[m])) < 2:
                models[sec][lvl] = None
                meta["sectors"][sec][lvl] = {"auc": None, "base_rate": None, "validated": False}
                log(f"  {sec}/{lvl}: not enough positive or negative outcomes - no model")
                continue
            models[sec][lvl] = mk().fit(X.to_numpy()[m], yv[m])
            # a sector model is only used if it beats "just look at Kp" on held-out rows
            meta["sectors"][sec][lvl] = {"auc": auc, "kp_auc": kp_auc, "base_rate": float(yv[m].mean()),
                                         "validated": bool(auc is not None and kp_auc is not None
                                                           and auc >= 0.70 and auc >= kp_auc + 0.01)}
            log(f"  {sec}/{lvl}: base rate {yv[m].mean() * 100:.1f}%  held-out ROC-AUC {auc}")
    joblib.dump({"models": models, "features": feats}, SECTOR_PATH)
    return {"models": models, "features": feats}, meta


class SectorModels:
    def __init__(self, bundle: Dict, meta: Dict):
        self.models, self.features, self.meta = bundle["models"], bundle["features"], meta

    def outlook(self, fv: Dict[str, float]) -> Dict[str, Dict]:
        vec = np.array([[fv.get(k, np.nan) for k in self.features]], dtype=float)
        out = {}
        for sec, lv in self.models.items():
            if not np.isfinite(vec).all():
                out[sec] = None
                continue
            r = {}
            for k in ("elev", "severe"):
                rf = lv.get(k)
                r[k] = None if rf is None else float(rf.predict_proba(vec)[0][list(rf.classes_).index(1.0)])
                r[k + "_ok"] = bool(self.meta["sectors"].get(sec, {}).get(k, {}).get("validated"))
            out[sec] = r
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN / LOAD
# ─────────────────────────────────────────────────────────────────────────────
def train_all(log: Callable[[str], None] = print) -> Dict:
    lines: List[str] = []

    def both(msg: str):
        lines.append(msg)
        log(msg)

    frame, source, years = load_training_frame(TRAIN_YEARS, both)
    model, lmeta = train_lstm(frame, both)
    _, rmeta = train_sector_models(frame, both)
    meta = {"version": MODEL_VERSION, "features": FEATURES, "source": source, "years": years,
            "trained_at": datetime.now(timezone.utc).isoformat(), "lstm": lmeta, "sectors": rmeta}
    META_PATH.write_text(json.dumps(meta, indent=2))
    LOG_PATH.write_text("\n".join(lines))
    return meta


def weights_are_fresh(meta: Optional[Dict], allow_stale: bool = False) -> bool:
    if not (meta and LSTM_PATH.exists() and SECTOR_PATH.exists()):
        return False
    if meta.get("version") != MODEL_VERSION or meta.get("features") != FEATURES:
        return False
    if allow_stale:
        return True
    try:
        return datetime.now(timezone.utc) - datetime.fromisoformat(meta["trained_at"]) < timedelta(days=RETRAIN_DAYS)
    except Exception:
        return False


def load_meta() -> Optional[Dict]:
    try:
        return json.loads(META_PATH.read_text()) if META_PATH.exists() else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
DRIVER_NAMES = {"kp_norm": "Kp", "wind_speed_norm": "wind speed", "bz_norm": "Bz",
                "density_norm": "density", "temperature_norm": "temperature"}


def detect_pattern(raw: Dict[str, Optional[float]]) -> Tuple[str, bool]:
    """Rule-based label from the observations that exist. Returns (text, is_quiet)."""
    g = lambda k: raw.get(k)
    kp, bz, spd, xr, pr, el = g("kp"), g("bz_gsm"), g("wind_speed"), g("xray_flux"), g("proton_flux"), g("electron_flux")
    pdyn = dyn_pressure_npa(g("density"), spd) if g("density") and spd else None
    if pr is not None and pr >= 10:            return "Solar radiation storm (S1+) in progress", False
    if xr is not None and xr >= 1e-5:          return "Flare-driven HF blackout risk (M-class+)", False
    if bz is not None and spd and bz <= -8 and spd >= 500:
        return "Fast solar wind + sustained southward Bz", False
    if bz is not None and bz <= -8:            return "Southward Bz - strong reconnection coupling", False
    if kp is not None and kp >= 5:             return "Geomagnetic storm in progress (G1+)", False
    if pdyn is not None and pdyn >= 10:        return "Strong solar-wind pressure - magnetopause compression", False
    if spd is not None and spd >= 600:         return "High-speed solar wind stream", False
    if el is not None and el >= 1000:          return "Elevated outer-belt electron flux", False
    if kp is not None and kp >= 4:             return "Unsettled geomagnetic field (Kp 4)", False
    if all(v is None for v in (kp, bz, spd, xr, pr, el)):
        return "No observations available", True
    return "Quiet - no organised driver", True


class LSTMPredictor:
    def __init__(self, model: SpaceWeatherLSTM, meta: Dict):
        self.model = model.eval()
        self.meta = meta
        self.baseline = np.asarray(meta["lstm"]["baseline"], dtype=np.float32)

    def predict(self, window: np.ndarray, coverage: float) -> Dict:
        X = torch.from_numpy(window[None].astype(np.float32))
        with torch.no_grad():
            logit, attn = self.model(X)
        p = float(torch.sigmoid(logit).item())
        peak = int(attn.squeeze(0).numpy().argmax())
        drivers = []                         # occlusion: P drop when one input is set to its typical (median) value
        for j, name in enumerate(FEATURES):
            Xo = X.clone()
            Xo[:, :, j] = float(self.baseline[j])
            with torch.no_grad():
                drivers.append((name, p - float(torch.sigmoid(self.model(Xo)[0]).item())))
        floor = max(0.02, 0.15 * p)
        top = [DRIVER_NAMES[n] for n, d in sorted(drivers, key=lambda t: -t[1]) if d >= floor][:3]
        return {"storm_probability": round(p, 4), "coverage": round(float(coverage), 3),
                "peak_attention_hours_ago": (SEQ_LEN - 1 - peak) * BLOCK_H, "drivers": top,
                "horizon_h": FORECAST_STEPS * BLOCK_H, "window_h": SEQ_LEN * BLOCK_H}


# ─────────────────────────────────────────────────────────────────────────────
# 7. LIVE INGEST  (NOAA SWPC)
# ─────────────────────────────────────────────────────────────────────────────
# name -> (paths, merge). merge=True: read every path and stitch them (primary wins).
ENDPOINTS: Dict[str, Tuple[List[str], bool]] = {
    "kp":       (["/products/noaa-planetary-k-index.json"], False),
    "xray":     (["/json/goes/primary/xrays-3-day.json", "/json/goes/primary/xrays-1-day.json"], False),
    "proton":   (["/json/goes/primary/integral-protons-3-day.json", "/json/goes/primary/integral-protons-1-day.json"], False),
    "electron": (["/json/goes/primary/integral-electrons-3-day.json", "/json/goes/primary/integral-electrons-1-day.json"], False),
    "mag":      (["/json/rtsw/rtsw_mag_1m.json"], True),     # ~2.6 days of 1-min data
    "wind":     (["/json/rtsw/rtsw_wind_1m.json"], True),
}
STALE_MIN = {"kp": 240, "xray": 45, "proton": 45, "electron": 120,
             "bz": 45, "speed": 45, "density": 45, "temp": 45}
WIND_KEYS = ("bz", "speed", "density", "temp")     # 1-minute feeds: shown and scored as 10-minute means
FIELD_SERIES = {"kp": "kp", "xray_flux": "xray", "proton_flux": "proton",
                "electron_flux": "electron", "bz_gsm": "bz", "wind_speed": "speed",
                "density": "density", "temperature": "temp"}


def _to_df(data) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    if isinstance(data[0], list):                  # legacy [header, rows...]
        return pd.DataFrame(data[1:], columns=data[0])
    return pd.DataFrame(data)


def parse_series(data, columns: List[str], *, filters: Optional[Dict[str, str]] = None,
                 active_only: bool = False, positive: bool = False) -> pd.Series:
    """Time-indexed (UTC) numeric series from either NOAA JSON layout."""
    df = _to_df(data)
    if df.empty or "time_tag" not in df:
        return pd.Series(dtype=float)
    col = next((c for c in columns if c in df.columns), None)
    if col is None:
        return pd.Series(dtype=float)
    for k, val in (filters or {}).items():
        if k in df:
            df = df[df[k] == val]
    if active_only and "active" in df:
        act = df[df["active"].astype(str).str.lower().isin(["true", "1"])]
        if len(act):
            df = act
    t = pd.to_datetime(df["time_tag"], utc=True, errors="coerce")
    v = pd.to_numeric(df[col], errors="coerce")
    s = pd.Series(v.values, index=t).dropna()
    s = s[s.index.notna()]
    if positive:
        s = s[s > 0]
    return s[~s.index.duplicated(keep="last")].sort_index()


def _fetch(paths: List[str], merge: bool = False) -> Tuple[Optional[list], Optional[str]]:
    """Returns (list of payloads, error). First success only unless merge=True."""
    got, last = [], None
    for p in paths:
        try:
            r = requests.get(SWPC + p, timeout=25, headers=UA)
            r.raise_for_status()
            got.append(r.json())
            if not merge:
                break
        except Exception as exc:
            last = f"{p}: {exc}"
    return (got or None), (None if got else last)


def _merge(parts: List[pd.Series]) -> pd.Series:
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.Series(dtype=float)
    s = pd.concat(parts)                           # earlier parts (higher quality) win on duplicates
    return s[~s.index.duplicated(keep="first")].sort_index()


def build_series(payloads: Dict[str, Optional[List[list]]]) -> Dict[str, pd.Series]:
    def over(key, fn):
        return _merge([fn(p) for p in (payloads.get(key) or [])])

    s = {
        "kp":       over("kp", lambda d: parse_series(d, ["Kp", "kp_index"])),
        "xray":     over("xray", lambda d: parse_series(d, ["flux"], filters={"energy": "0.1-0.8nm"}, positive=True)),
        "proton":   over("proton", lambda d: parse_series(d, ["flux"], filters={"energy": ">=10 MeV"}, positive=True)),
        "electron": over("electron", lambda d: parse_series(d, ["flux"], filters={"energy": ">=2 MeV"}, positive=True)),
        "bz":       over("mag", lambda d: parse_series(d, ["bz_gsm"], active_only=True)),
        "speed":    over("wind", lambda d: parse_series(d, ["proton_speed", "speed"], active_only=True, positive=True)),
        "density":  over("wind", lambda d: parse_series(d, ["proton_density", "density"], active_only=True, positive=True)),
        "temp":     over("wind", lambda d: parse_series(d, ["proton_temperature", "temperature"], active_only=True, positive=True)),
    }
    s["bz"] = s["bz"][s["bz"].abs() < 900]
    return s


def collect_series() -> Tuple[Dict[str, pd.Series], Dict[str, str]]:
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {k: ex.submit(_fetch, paths, merge) for k, (paths, merge) in ENDPOINTS.items()}
        results = {k: f.result() for k, f in futs.items()}
    payloads = {k: r[0] for k, r in results.items()}
    errors = {k: r[1] for k, r in results.items() if r[0] is None}
    return build_series(payloads), errors


def build_window(series: Dict[str, pd.Series], now: pd.Timestamp) -> Tuple[Optional[np.ndarray], float, str]:
    """
    Last SEQ_LEN *completed* 3-hour blocks (matches training: block means).
    Only real observations are used: gaps of up to 2 blocks inside the window are
    interpolated (same rule as training); anything else -> no forecast.
    Returns (window | None, coverage, reason).
    """
    end = now.floor("3h") - pd.Timedelta(hours=BLOCK_H)
    blocks = pd.date_range(end=end, periods=SEQ_LEN, freq="3h", tz="UTC")
    cols = {"kp": series["kp"].reindex(blocks)}
    for k in ("bz", "speed", "density", "temp"):
        s = series[k]
        cols[k] = s.resample("3h").mean().reindex(blocks) if len(s) else pd.Series(np.nan, index=blocks)
    f = pd.DataFrame(cols)
    coverage = float(f.notna().all(axis=1).mean())
    if coverage < MIN_COVERAGE:
        missing = [k for k in cols if f[k].notna().mean() < MIN_COVERAGE]
        return None, coverage, "not enough observed blocks (" + ", ".join(missing or ["all inputs"]) + ")"
    for k in ("bz", "speed", "density", "temp"):
        f[k] = f[k].interpolate(limit=2, limit_area="inside")
    if f.isna().any().any():
        return None, coverage, "gaps at the edge of the 24 h window"
    W = np.column_stack([n_kp(f["kp"].values), n_speed(f["speed"].values),
                         n_bz(f["bz"].values), n_density(f["density"].values),
                         n_temp(f["temp"].values)]).astype(np.float32)
    return W, coverage, ""


def live_rf_features(series: Dict[str, pd.Series], now: pd.Timestamp, feats: List[str]) -> Dict[str, float]:
    """Same definitions as rf_matrix, from the most recent 3 h of live data. NaN if unobserved."""
    cut = now - pd.Timedelta(hours=BLOCK_H)
    last = lambda k: series[k][series[k].index > cut]
    kp = series["kp"]
    kp24 = kp[kp.index > now - pd.Timedelta(hours=24)]
    bz, spd, den, tmp, p = (last("bz"), last("speed"), last("density"), last("temp"), last("proton"))
    mean = lambda s: float(s.mean()) if len(s) else float("nan")
    v = {"kp": float(kp.iloc[-1]) if len(kp) else float("nan"),
         "kp_max24": float(kp24.max()) if len(kp24) else float("nan"),
         "speed": mean(spd), "bz_south": max(-mean(bz), 0.0) if len(bz) else float("nan"),
         "density": mean(den),
         "temp_log": math.log10(mean(tmp)) if len(tmp) and mean(tmp) > 0 else float("nan"),
         "proton_log": math.log10(max(float(p.max()), 0.01)) if len(p) else float("nan")}
    v["pressure"] = dyn_pressure_npa(v["density"], v["speed"])
    return {k: v[k] for k in feats}


def build_trend(series: Dict[str, pd.Series], now: pd.Timestamp, hours: int = 48) -> List[Dict]:
    idx = pd.date_range(end=now.floor("1h"), periods=hours, freq="1h", tz="UTC")

    def hourly(s: pd.Series, how: str = "mean") -> pd.Series:
        if not len(s):
            return pd.Series(np.nan, index=idx)
        return getattr(s.resample("1h"), how)().reindex(idx)

    kp = series["kp"]
    kp_h = (kp.reindex(idx, method="ffill", tolerance=pd.Timedelta("3h"))
            if len(kp) else pd.Series(np.nan, index=idx))
    df = pd.DataFrame({"kp": kp_h, "xray": hourly(series["xray"], "median"),
                       "proton": hourly(series["proton"]), "bz": hourly(series["bz"]),
                       "wind": hourly(series["speed"])})
    return [{"t": t.isoformat(), **{k: (None if pd.isna(v) else float(v)) for k, v in r.items()}}
            for t, r in df.iterrows()]


# ─────────────────────────────────────────────────────────────────────────────
# 8. SECTOR NOWCAST  (NOAA scales applied to the live observations)
# ─────────────────────────────────────────────────────────────────────────────
def kp_score(kp):          # G-scale: Kp5 = G1 = 40, Kp7 = G3 = 70, Kp9 = G5 = 100
    return None if kp is None else float(np.interp(kp, [0, 4, 5, 6, 7, 8, 9], [0, 25, 40, 55, 70, 85, 100]))


def proton_score(pfu):     # S-scale: 10 pfu = S1 = 40, 100 = S2, 1e3 = S3, 1e4 = S4
    return None if pfu is None or pfu <= 0 else float(np.interp(math.log10(pfu), [-1, 0, 1, 2, 3, 4], [0, 10, 40, 60, 80, 100]))


def xray_score(flux):      # R-scale: M1 = 1e-5 = R1 = 40, X1 = 1e-4 = R3, X10 = R4
    return None if flux is None or flux <= 0 else float(np.interp(math.log10(flux), [-8, -6, -5, -4.3, -4, -3], [0, 10, 40, 55, 75, 100]))


def electron_score(flux):  # >2 MeV; ~1e3 pfu is the usual charging-alert level
    return None if flux is None or flux <= 0 else float(np.interp(math.log10(flux), [1, 2, 3, 4, 5], [0, 10, 40, 75, 100]))


def bz_score(bz):
    return None if bz is None else float(np.interp(max(-bz, 0), [0, 5, 10, 20], [0, 10, 40, 80]))


def pressure_score(p):
    return None if p is None else float(np.interp(p, [2, 5, 10, 20], [0, 15, 40, 80]))


def driver_scores(raw: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    pdyn = None
    if raw.get("density") and raw.get("wind_speed"):
        pdyn = dyn_pressure_npa(raw["density"], raw["wind_speed"])
    return {"kp": kp_score(raw.get("kp")), "proton": proton_score(raw.get("proton_flux")),
            "xray": xray_score(raw.get("xray_flux")), "electron": electron_score(raw.get("electron_flux")),
            "bz": bz_score(raw.get("bz_gsm")), "pressure": pressure_score(pdyn)}


# sector -> {driver: weight}. score = strongest weighted driver (a sector is as exposed as its worst threat)
SECTORS = {
    "satellite": {
        "label": "Satellite operations",
        "drivers": {"electron": 1.0, "proton": 1.0, "kp": 0.7},
        "threats": [("electron", 34, "Deep dielectric charging risk"),
                    ("proton", 34, "Single-event upsets"),
                    ("kp", 25, "Increased LEO atmospheric drag"),
                    ("proton", 55, "Star-tracker and sensor noise")],
        "advice": {"low": "Routine operations. No charging or upset concerns.",
                   "med": "Elevated charging environment - defer non-critical maneuvers.",
                   "high": "Severe radiation environment - safe-mode sensitive payloads, hold maneuvers."},
    },
    "aviation": {
        "label": "Aviation & communications",
        "drivers": {"xray": 1.0, "proton": 1.0, "kp": 0.6},
        "threats": [("xray", 34, "HF blackout on the sunlit side"),
                    ("proton", 34, "Polar HF absorption"),
                    ("kp", 34, "GNSS position error"),
                    ("proton", 55, "Elevated crew radiation dose")],
        "advice": {"low": "Routine ops nominal; monitor polar HF on high-latitude tracks.",
                   "med": "HF fade likely on polar routes - brief crews, cross-check GNSS.",
                   "high": "HF blackout and GNSS errors likely - reroute polar flights."},
    },
    "power_grid": {
        "label": "Power grid stability",
        "drivers": {"kp": 1.0, "bz": 0.6, "pressure": 0.6},
        "threats": [("kp", 40, "Geomagnetically induced currents"),
                    ("kp", 70, "Voltage instability, mid-latitude corridors"),
                    ("bz", 34, "Sustained southward Bz coupling"),
                    ("pressure", 34, "Magnetopause compression")],
        "advice": {"low": "Grid conditions stable. Routine monitoring.",
                   "med": "GIC possible - raise monitoring cadence, coordinate with TSOs.",
                   "high": "GIC exceedance likely - stage reactive reserves and defer maintenance."},
    },
}
LEVEL_FLOOR = {"low": 0, "med": 34, "high": 67}


def _level(score: float) -> str:
    return "high" if score >= 67 else "med" if score >= 34 else "low"


def sector_nowcast(key: str, scores: Dict[str, Optional[float]], outlook: Optional[Dict]) -> Dict:
    """Level and score = NOAA-scale nowcast of what is observed NOW. The 9 h outlook is reported
    separately (never mixed into the score) so the two can be read independently."""
    cfg = SECTORS[key]
    parts = {d: scores[d] * w for d, w in cfg["drivers"].items() if scores.get(d) is not None}
    base = {"sector": cfg["label"], "outlook": None}
    if not parts:
        return {**base, "level": "nodata", "risk_score": None, "threats": [],
                "advisory": "No observations available for this sector's inputs."}
    score = int(round(max(parts.values())))
    lv = _level(score)
    ol = None
    if outlook:
        pe, ps = outlook.get("elev"), outlook.get("severe")
        ok_e, ok_s = bool(outlook.get("elev_ok")), bool(outlook.get("severe_ok"))
        ol = {"p_elev": pe, "p_severe": ps, "validated": ok_e, "severe_validated": ok_s, "level": None}
        if ok_s and ps is not None and ps >= 0.5:
            ol["level"] = "high"
        elif ok_e and pe is not None and pe >= 0.5:
            ol["level"] = "med"
    threats = []
    for d, thr, text in cfg["threats"]:
        if scores.get(d) is not None and scores[d] >= thr and text not in threats:
            threats.append(text)
    advisory = cfg["advice"][lv]
    if ol and ol["level"]:
        word = "severe" if ol["level"] == "high" else "elevated"
        if lv == "low":
            advisory = f"Nominal now, but {word} conditions are likely within 9 h - raise monitoring."
    return {**base, "outlook": ol, "level": lv, "risk_score": score,
            "advisory": advisory, "threats": threats[:4]}


def _pct_txt(p: Optional[float]) -> str:
    return "-" if p is None else ("<1%" if p < 0.01 else f"{p * 100:.0f}%")


def overall_status(sectors: Dict[str, Dict]) -> Dict:
    """Banner = what is observed now; the 9 h outlook is stated separately so 'all LOW' cards never read 'elevated'."""
    live = [s for s in sectors.values() if s["level"] != "nodata"]
    if not live:
        return {"level": "nodata", "label": "NO DATA", "text": "No live observations could be read."}
    gap = "" if len(live) == len(sectors) else " Some sectors have no data."
    hi = [s["sector"] for s in live if s["level"] == "high"]
    md = [s["sector"] for s in live if s["level"] == "med"]
    soon = [s["sector"] for s in live if s["level"] == "low" and (s.get("outlook") or {}).get("level")]
    tail = ""
    if soon and (hi or md):
        tail = f" Outlook: elevated conditions likely within 9 h for {', '.join(soon)}."
    if hi:
        return {"level": "high", "label": "ALERT", "text": f"{', '.join(hi)} at critical risk - follow sector advisories." + tail + gap}
    if md:
        return {"level": "med", "label": "WATCH",
                "text": (f"{', '.join(md)} elevated now." + tail + gap)}
    if soon:
        who = "all sectors" if len(soon) == len(live) else ", ".join(soon)
        return {"level": "med", "label": "WATCH",
                "text": f"Nominal now. Elevated conditions likely within 9 h for {who} - monitor." + gap}
    return {"level": "low", "label": "NOMINAL", "text": "All sectors within normal bounds." + gap}


# ─────────────────────────────────────────────────────────────────────────────
# 8b. ADVISORY TEXT  (optional free LLM; templates only if it is unavailable)
# ─────────────────────────────────────────────────────────────────────────────
OBS_UNITS = {"kp": "", "xray_flux": "W/m2", "proton_flux": "pfu >10 MeV", "electron_flux": "pfu >2 MeV",
             "wind_speed": "km/s", "bz_gsm": "nT", "density": "p/cm3", "temperature": "K"}


def _band(p: Optional[float]) -> Optional[str]:
    return None if p is None else "likely" if p >= 0.5 else "possible" if p >= 0.05 else "unlikely"


def trend_word(s: pd.Series, now: pd.Timestamp, kind: str) -> Optional[str]:
    """How a quantity moved over ~3 h, from the real series. kind: kp | bz | rel | log."""
    if s is None or len(s) < 2:
        return None
    if kind == "kp":
        d = float(s.iloc[-1] - s.iloc[-2])
        return "rising" if d >= 0.5 else "falling" if d <= -0.5 else "steady"
    a = s[s.index > now - pd.Timedelta(hours=1)]
    b = s[(s.index > now - pd.Timedelta(hours=4)) & (s.index <= now - pd.Timedelta(hours=3))]
    if a.empty or b.empty:
        return None
    a, b = float(a.mean()), float(b.mean())
    if kind == "bz":
        return "turning more southward" if a - b <= -2 else "turning northward" if a - b >= 2 else "steady"
    if kind == "log":
        r = a / b if b > 0 else 1.0
        return "rising" if r >= 1.5 else "falling" if r <= 1 / 1.5 else "steady"
    r = a / b if b > 0 else 1.0
    return "rising" if r >= 1.15 else "falling" if r <= 1 / 1.15 else "steady"


def compute_trends(series: Dict[str, pd.Series], now: pd.Timestamp) -> Dict[str, Optional[str]]:
    return {"kp": trend_word(series["kp"], now, "kp"), "bz_gsm": trend_word(series["bz"], now, "bz"),
            "wind_speed": trend_word(series["speed"], now, "rel"), "density": trend_word(series["density"], now, "rel"),
            "xray_flux": trend_word(series["xray"], now, "log"), "proton_flux": trend_word(series["proton"], now, "log"),
            "electron_flux": trend_word(series["electron"], now, "log")}


def xray_class(f: float) -> str:
    for lim, letter in ((1e-4, "X"), (1e-5, "M"), (1e-6, "C"), (1e-7, "B"), (0, "A")):
        if f >= lim:
            return f"{letter}{f / lim:.1f}" if lim else f"A{f / 1e-8:.1f}"
    return "A"


def driver_reading(name: str, raw: Dict[str, Optional[float]], tr: Dict[str, Optional[str]]) -> Optional[str]:
    """One plain-language line per driver: the value, where it sits on its scale, and its 3 h movement."""
    g = raw.get(name)
    if name == "pressure":
        if not (raw.get("density") and raw.get("wind_speed")):
            return None
        pr = dyn_pressure_npa(raw["density"], raw["wind_speed"])
        return f"solar-wind dynamic pressure {pr:.1f} nPa ({'strong, above 10 nPa' if pr >= 10 else 'within the normal range'}), density {tr.get('density') or 'trend unknown'}"
    if g is None:
        return None
    mv = f", {tr[name]} vs 3 h ago" if tr.get(name) else ""
    if name == "kp":
        return f"Kp {g:.1f} ({'at or above' if g >= 5 else 'below'} storm level G1 at Kp 5){mv}"
    if name == "proton_flux":
        return f"protons >10 MeV {g:.2g} pfu ({'above' if g >= 10 else 'below'} the S1 radiation-storm threshold of 10 pfu){mv}"
    if name == "xray_flux":
        return f"X-ray class {xray_class(g)} ({'M-class or above: radio-blackout level' if g >= 1e-5 else 'below M-class flare level'}){mv}"
    if name == "electron_flux":
        return f"electrons >2 MeV {g:.0f} pfu ({'above' if g >= 1000 else 'below'} the ~1000 pfu charging-alert level){mv}"
    if name == "bz_gsm":
        word = "southward" if g < -1 else "northward" if g > 1 else "near zero"
        return f"Bz {g:+.1f} nT ({word}; below -10 nT couples strongly to Earth){mv}"
    if name == "wind_speed":
        return f"solar wind {g:.0f} km/s ({'fast, above 500' if g >= 500 else 'ordinary speed'}){mv}"
    return None


SECTOR_FACTS = {   # what each sector is sensitive to (domain grounding for the LLM; it may mention only these effects)
    "satellite": (["kp", "electron_flux", "proton_flux"],
                  "Geomagnetic activity (Kp) raises upper-atmosphere drag on low-orbit satellites; >2 MeV electrons "
                  "drive deep dielectric charging; >10 MeV protons cause single-event upsets."),
    "aviation": (["xray_flux", "proton_flux", "kp"],
                 "X-ray flares cause HF radio blackouts on the sunlit side; >10 MeV protons cause polar HF absorption "
                 "and higher crew radiation dose; geomagnetic activity degrades GNSS accuracy."),
    "power_grid": (["kp", "bz_gsm", "pressure", "wind_speed"],
                   "Geomagnetic storms (high Kp) drive geomagnetically induced currents in power lines; sustained "
                   "southward Bz and strong solar-wind pressure increase the coupling and compress the magnetosphere."),
}


def advisory_facts(raw: Dict[str, Optional[float]], sectors: Dict[str, Dict],
                   trends: Optional[Dict[str, Optional[str]]] = None) -> Dict:
    tr = trends or {}
    sec = {}
    for k, s in sectors.items():
        ol = s.get("outlook") or {}
        names, mech = SECTOR_FACTS[k]
        sec[k] = {"name": s["sector"], "level_now": s["level"], "score_now_of_100": s["risk_score"],
                  "elevated_conditions_next_9h": _band(ol.get("p_elev")) if ol.get("validated") else None,
                  "readings": [x for x in (driver_reading(n, raw, tr) for n in names) if x],
                  "mechanisms_you_may_mention": mech}
    return {"sectors": sec}


ADVISORY_SYSTEM = ("You write operational advisories for a space-weather risk dashboard, one per sector. Use ONLY the "
                   "facts given. Be specific: name the readings that matter for that sector (quote the values exactly as "
                   "given, with their trend), say what they mean for that sector, and say what to do. Mention only "
                   "effects listed under mechanisms_you_may_mention; do not invent causes, events or numbers, and do "
                   "not call a reading high unless its text says it is above a threshold. Lead with the most "
                   "important reading. If it is all quiet, say which readings are quiet. Match level_now (low = "
                   "routine, med = elevated, high = severe) and mention elevated_conditions_next_9h when it is "
                   "likely or possible. Never recommend more than level_now justifies: low = routine monitoring only, "
                   "med = raise monitoring and prepare, high = act. Say a reading 'may' or 'could' affect something "
                   "unless it is at or above its threshold, and mention an effect only if one of that sector's "
                   "readings is actually relevant to it (for example HF blackout only if the X-ray reading is at "
                   "M class or above). Plain text, no markdown, no organisation names. Each advisory: two sentences, "
                   "at most 50 words. Reply with a single JSON object and nothing else: "
                   '{"satellite": "...", "aviation": "...", "power_grid": "..."}')


def _llm_call(model: str, facts: Dict, key: str) -> str:
    """One chat-completion call. Returns the message content; raises with diagnostics if there is none."""
    body = {"model": model, "temperature": 0.2, "max_completion_tokens": 4000,
            "messages": [{"role": "system", "content": ADVISORY_SYSTEM},
                         {"role": "user", "content": json.dumps(facts)}]}
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"          # reasoning models can spend the whole budget thinking
    attempts = [{**body, "response_format": {"type": "json_object"}}, body]
    last: Optional[Exception] = None
    for payload in attempts:                      # 1st: JSON mode; if the API rejects the option, plain
        try:
            r = requests.post(LLM_URL, timeout=20, headers={"Authorization": f"Bearer {key}",
                                                            "Content-Type": "application/json"}, json=payload)
            if r.status_code == 400 and payload is not attempts[-1]:
                last = RuntimeError(f"400 {r.text[:120]}")
                continue
            r.raise_for_status()
            ch = r.json()["choices"][0]
            content = ch["message"].get("content") or ""
            if not content.strip():
                raise ValueError(f"empty reply from {model} (finish_reason={ch.get('finish_reason')})")
            return content
        except requests.HTTPError as exc:
            raise RuntimeError(f"{model}: HTTP {exc.response.status_code} {exc.response.text[:100]}") from None
    raise last or RuntimeError("no attempt made")


def groq_key() -> str:
    """Key from the environment (local shell, HF/Docker) or from Streamlit secrets (Community Cloud)."""
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    try:
        return str(st.secrets.get("GROQ_API_KEY", "") or "")
    except Exception:                                  # no secrets file configured
        return ""


def llm_advisories(facts: Dict) -> Dict[str, str]:
    """One call, three advisories, written only from the computed facts. Raises on any problem."""
    key = groq_key()
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    errors, out = [], None
    for model in dict.fromkeys([LLM_MODEL, LLM_FALLBACK]):          # primary, then fallback (deduplicated)
        try:
            content = _llm_call(model, facts, key)
            m = re.search(r"\{.*\}", content, re.S)
            if not m:
                raise ValueError(f"{model}: no JSON in reply (starts: {content[:60]!r})")
            out = json.loads(m.group(0))
            allowed = set(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", json.dumps(facts))) | {"9"}
            allowed |= {str(round(float(n))) for n in list(allowed)}          # integer roundings of given values
            clean = {}
            for k in facts["sectors"]:
                txt = str(out.get(k, "")).strip()
                if not txt or len(txt) > 380:
                    raise ValueError(f"{model}: bad advisory for {k}")
                bad = [n for n in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", txt) if n not in allowed]
                if bad:                                                        # the model may not introduce numbers
                    raise ValueError(f"{model}: advisory for {k} has numbers not in the data: {bad}")
                band = facts["sectors"][k].get("elevated_conditions_next_9h")
                if band in ("likely", "possible") and band not in txt.lower():
                    txt += f" Elevated conditions {band} within 9 h."          # outlook must never be left out
                clean[k] = txt
            clean["_model"] = model
            return clean
        except Exception as exc:
            errors.append(str(exc)[:110])
    raise RuntimeError(" | ".join(errors))


def apply_advisories(raw, sectors: Dict[str, Dict], prev: Optional[Dict], trends: Optional[Dict] = None,
                     force: bool = False) -> Dict:
    """
    Fills sectors[*]['advisory']. Free-tier budget (Groq: ~1000 requests and 200k tokens a day) is protected:
    text is regenerated at once when a sector's level or outlook band changes, otherwise at most every
    15 min when its threat list changes, and after a failure the API is not retried for 10 min.
    """
    facts = advisory_facts(raw, sectors, trends)
    state = {k: (v["level_now"], v["elevated_conditions_next_9h"]) for k, v in facts["sectors"].items()}
    detail = {k: [s["threats"], trends and [trends.get(n) for n in SECTOR_FACTS[k][0] if n in trends]]
              for k, s in sectors.items()}
    key, dkey = json.dumps(state, sort_keys=True), json.dumps(detail, sort_keys=True)
    now = pd.Timestamp.now(tz="UTC")
    old = (prev or {}).get("advisory_meta") or {}
    try:
        age = (now - pd.Timestamp(old["ts"])).total_seconds() / 60 if old.get("ts") else 1e9
    except Exception:
        age = 1e9
    if all(s["level"] == "nodata" for s in sectors.values()):
        for s in sectors.values():
            s["advisory_source"] = "rules"
        return {"key": key, "dkey": dkey, "ts": now.isoformat(), "source": "rules", "texts": {}, "error": "no observations"}
    same_state = old.get("key") == key and old.get("build") == BUILD
    if old.get("source") == "llm" and old.get("texts") and same_state and (old.get("dkey") == dkey or age < 15):
        texts, meta = old["texts"], {**old}
    elif old.get("source") == "rules" and same_state and age < 5 and not force:
        texts, meta = {}, {**old}                                   # recent failure: do not hammer the API
    else:
        try:
            texts = llm_advisories(facts)
            used = texts.pop("_model", LLM_MODEL)
            meta = {"key": key, "dkey": dkey, "ts": now.isoformat(), "build": BUILD, "source": "llm",
                    "model": used, "texts": texts}
        except Exception as exc:
            texts = {}
            meta = {"key": key, "dkey": dkey, "ts": now.isoformat(), "build": BUILD, "source": "rules", "texts": {},
                    "error": str(exc)[:300].replace(groq_key() or "\0", "***")}
    if meta.get("ts") == now.isoformat():                             # only log when a call was actually made
        print(f"[advisory] {meta['source']}" + (f" ({meta['model']})" if meta["source"] == "llm" else f" - {meta.get('error')}"))
    for k, s in sectors.items():
        if s["level"] != "nodata" and texts.get(k):
            s["advisory"], s["advisory_source"] = texts[k], "llm"
        else:
            s["advisory_source"] = "rules"
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# 9. PIPELINE + JSON DOCUMENT STORE
# ─────────────────────────────────────────────────────────────────────────────
def _jsonable(o):
    if isinstance(o, (np.floating,)):  return float(o)
    if isinstance(o, (np.integer,)):   return int(o)
    if isinstance(o, (np.bool_,)):     return bool(o)
    if isinstance(o, np.ndarray):      return o.tolist()
    return str(o)


def load_snapshot() -> Optional[Dict]:
    try:
        return json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else None
    except Exception:
        return None


def save_snapshot(snap: Dict) -> None:
    try:
        tmp = CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(snap, indent=1, default=_jsonable))
        tmp.replace(CACHE_PATH)
    except Exception as exc:
        print(f"snapshot write failed: {exc}")


def run_pipeline(models: Dict, prev: Optional[Dict] = None, force: bool = False) -> Dict:
    now = pd.Timestamp.now(tz="UTC")
    series, errors = collect_series()
    prev = prev or {}
    prev_raw, prev_stat = prev.get("raw_data", {}), prev.get("data_status", {})
    kp_source = "NOAA SWPC"

    if series["kp"].empty:                               # NOAA Kp unreachable -> GFZ Kp (real, independent)
        try:
            series["kp"] = fetch_gfz_kp(now - pd.Timedelta(days=3), now)
            kp_source = "GFZ Potsdam"
        except Exception as exc:
            errors["kp_gfz"] = str(exc)[:120]

    raw: Dict[str, Optional[float]] = {}
    status: Dict[str, Dict] = {}
    for field, key in FIELD_SERIES.items():
        s = series.get(key)
        if s is not None and len(s):
            ts = s.index[-1]
            age = (now - ts).total_seconds() / 60.0
            if key == "kp":                              # Kp stamps mark block START; it is current until block end
                age = max(0.0, age - 180.0)
            raw[field] = float(s[s.index > ts - pd.Timedelta(minutes=10)].mean()) if key in WIND_KEYS else float(s.iloc[-1])
            status[field] = {"ts": ts.isoformat(), "age_min": round(age, 1),
                             "status": "live" if age <= STALE_MIN[key] else "stale"}
        elif prev_raw.get(field) is not None:            # last REAL reading, clearly marked stale
            ts = pd.Timestamp(prev_stat.get(field, {}).get("ts") or prev.get("timestamp") or now.isoformat())
            raw[field] = prev_raw[field]
            status[field] = {"ts": ts.isoformat(), "age_min": round((now - ts).total_seconds() / 60.0, 1),
                             "status": "stale"}
        else:
            raw[field] = None
            status[field] = {"ts": None, "age_min": None, "status": "missing"}

    # ── LSTM ───────────────────────────────────────────────────────────────
    lstm: Optional[LSTMPredictor] = models.get("lstm")
    horizon = {"horizon_h": FORECAST_STEPS * BLOCK_H, "window_h": SEQ_LEN * BLOCK_H}
    ml: Dict = {"storm_probability": None, "coverage": 0.0, "peak_attention_hours_ago": None,
                "drivers": [], "status": "unavailable", **horizon}
    if lstm is None:
        ml["reason"] = models.get("error") or "model not trained"
    else:
        window, coverage, why = build_window(series, now)
        ml["coverage"] = round(coverage, 3)
        if window is None:
            ml["reason"] = why
        else:
            ml.update(lstm.predict(window, coverage))
            ml["status"] = "ok" if lstm.meta["lstm"]["validated"] else "unvalidated"
    ml["pattern"], ml["pattern_quiet"] = detect_pattern(raw)

    # ── sectors ────────────────────────────────────────────────────────────
    scores = driver_scores(raw)
    outlook: Dict[str, Optional[Dict]] = {k: None for k in SECTORS}
    rf: Optional[SectorModels] = models.get("rf")
    if rf is not None:
        outlook = {**outlook, **rf.outlook(live_rf_features(series, now, rf.features))}
    # Coherence with the LSTM, which forecasts exactly "Kp >= 5 within 9 h": aviation's outlook IS that event,
    # and the grid event (Kp>=5 or Dst<=-50) contains it, so it can never be less likely than the LSTM says.
    pl = ml.get("storm_probability") if ml.get("status") == "ok" else None
    if pl is not None:
        av = outlook.get("aviation") or {"severe": None, "severe_ok": False}
        outlook["aviation"] = {**av, "elev": pl, "elev_ok": True}
        gr = outlook.get("power_grid")
        if gr and gr.get("elev") is not None and gr.get("elev_ok"):
            gr["elev"] = max(gr["elev"], pl)
    sectors = {k: sector_nowcast(k, scores, outlook.get(k)) for k in SECTORS}

    advisory_meta = apply_advisories(raw, sectors, prev, compute_trends(series, now), force)

    gfz = prev.get("gfz")
    try:
        t_prev = pd.Timestamp(gfz["checked"]) if gfz else None
    except Exception:
        t_prev = None
    if t_prev is None or (now - t_prev) > pd.Timedelta(hours=6):
        gfz = gfz_crosscheck(series["kp"], now) if kp_source == "NOAA SWPC" else None
        if gfz is not None:
            gfz["checked"] = now.isoformat()

    try:                                                   # aurora oval, routes, magnetopause: never blocks the dashboard
        viz = geoviz.build(raw, now.to_pydatetime(), prev.get("viz"))
    except Exception as exc:
        viz = {"ok": False, "error": str(exc)[:160]}

    mdl = models.get("meta") or {}
    snap = {
        "timestamp": now.isoformat(),
        "raw_data": raw,
        "data_status": status,
        "driver_scores": scores,
        "ml_forecast": ml,
        "sectors": sectors,
        "overall": overall_status(sectors),
        "derived": {"dynamic_pressure_npa": (round(dyn_pressure_npa(raw["density"], raw["wind_speed"]), 2)
                                             if raw.get("density") and raw.get("wind_speed") else None)},
        "trend": build_trend(series, now),
        "model": {k: v for k, v in mdl.items() if k not in ("sectors",)} | {
            "lstm": {k: v for k, v in (mdl.get("lstm") or {}).items() if k != "baseline"}},
        "advisory_meta": advisory_meta,
        "viz": viz,
        "build": BUILD,
        "kp_source": kp_source,
        "gfz": gfz,
        "errors": errors,
        "train_error": models.get("error"),
        "error": "No live data could be fetched from NOAA SWPC." if len(errors) >= len(ENDPOINTS) else None,
    }
    save_snapshot(snap)
    return snap


# ─────────────────────────────────────────────────────────────────────────────
# 8. VIEW  (pure functions: snapshot -> HTML string)
# ─────────────────────────────────────────────────────────────────────────────
e = html.escape

# Page chrome (outside the iframe): background grid, hidden Streamlit furniture.
PAGE_CSS = """
<style>
.stApp{
  background-color:#060a12;
  background-image:
    linear-gradient(rgba(130,170,210,.035) 1px,transparent 1px),
    linear-gradient(90deg,rgba(130,170,210,.035) 1px,transparent 1px);
  background-size:32px 32px;
}
header[data-testid="stHeader"],#MainMenu,footer{visibility:hidden;height:0}
.block-container{max-width:1360px !important;padding:22px 24px 40px !important}
section[data-testid="stSidebar"]{background:#080d16;border-right:1px solid #172234}
iframe{border:0 !important;color-scheme:normal}
</style>
"""

# Dashboard styles (inside the iframe, so inline SVG, fonts and hover all work).
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;600&display=swap');

:root{
  --bg:#060a12; --panel:#0b121e; --well:#090f19; --line:#172234; --line-2:#22314a;
  --ink:#e8eef8; --ink-2:#aab7cb; --ink-3:#6a7a92; --ink-4:#3d4a5f;
  --cyan:#22d3ee; --green:#34d399; --yellow:#f5c542; --orange:#fb7c3c; --red:#ff4157;
  --sans:'Space Grotesk','Inter',system-ui,sans-serif;
  --mono:'JetBrains Mono',ui-monospace,'SF Mono',Menlo,Consolas,monospace;
}
html,body{margin:0;background:transparent}
body{color:var(--ink)}

.ar{font-family:var(--sans);color:var(--ink);font-variant-numeric:tabular-nums}
.ar *{box-sizing:border-box}
:where(.ar) :is(h1,h2,h3,p,ul){margin:0;padding:0}
:where(.ar) ul{list-style:none}
.cap{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-3)}

/* header */
.top{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:16px;padding:6px 4px 18px}
.logo{font-family:var(--mono);font-weight:700;font-size:26px;letter-spacing:.34em;color:var(--cyan);
  text-shadow:0 0 22px rgba(34,211,238,.35)}
.logo small{display:block;margin-top:4px;font-size:10px;font-weight:500;letter-spacing:.28em;color:var(--ink-3);text-shadow:none}
.feed{justify-self:center;display:inline-flex;align-items:center;gap:9px;padding:6px 14px;border-radius:999px;
  font-family:var(--mono);font-size:11px;letter-spacing:.12em;border:1px solid}
.feed i{width:7px;height:7px;border-radius:50%;background:currentColor}
.feed.ok{color:var(--green);border-color:rgba(52,211,153,.4);background:rgba(52,211,153,.07)}
.feed.ok i{animation:pulse 2.4s ease-in-out infinite}
.feed.warn{color:var(--yellow);border-color:rgba(245,197,66,.4);background:rgba(245,197,66,.07)}
.feed.bad{color:var(--red);border-color:rgba(255,65,87,.45);background:rgba(255,65,87,.08)}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(52,211,153,.5)}50%{opacity:.5;box-shadow:0 0 0 5px rgba(52,211,153,0)}}
.clock{justify-self:end;text-align:right;font-family:var(--mono);font-size:12px;color:var(--ink-2);letter-spacing:.04em}
.clock #age{display:block;margin-top:3px;font-size:10px;color:var(--ink-3)}

/* status banner */
.status{display:flex;align-items:center;gap:14px;padding:10px 20px;border-radius:999px;border:1px solid;
  font-size:14px;margin-bottom:18px}
.status b{font-family:var(--mono);font-size:10.5px;font-weight:500;letter-spacing:.16em;padding-right:14px;
  border-right:1px solid currentColor;opacity:.85;white-space:nowrap}
.status.low{color:var(--green);border-color:rgba(52,211,153,.35);background:rgba(52,211,153,.06)}
.status.med{color:var(--yellow);border-color:rgba(245,197,66,.4);background:rgba(245,197,66,.07)}
.status.high{color:var(--red);border-color:rgba(255,65,87,.5);background:rgba(255,65,87,.09)}
.notice{padding:10px 16px;border-radius:12px;border:1px solid rgba(255,65,87,.4);background:rgba(255,65,87,.07);
  color:#ff9aa6;font-size:13px;margin-bottom:16px}

/* panels */
.panel{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:22px 24px}
.panel > .cap{display:block;margin-bottom:18px}

/* sectors */
.sectors{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-bottom:18px}
.sec{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:22px 24px;display:flex;flex-direction:column}
.sec.high{border-color:rgba(255,65,87,.55);box-shadow:0 0 0 1px rgba(255,65,87,.12),0 0 44px rgba(255,65,87,.16)}
.sec header{display:flex;justify-content:space-between;align-items:center;gap:10px}
.sec h3{font-family:var(--mono);font-weight:500;font-size:12px;letter-spacing:.13em;text-transform:uppercase;color:var(--ink-2)}
.chip{font-family:var(--mono);font-size:10px;letter-spacing:.14em;padding:3px 10px;border-radius:999px;border:1px solid;white-space:nowrap}
.low .chip{color:var(--green);border-color:rgba(52,211,153,.5)}
.med .chip{color:var(--yellow);border-color:rgba(245,197,66,.55)}
.high .chip{color:var(--red);border-color:rgba(255,65,87,.6)}
.score{display:flex;align-items:baseline;gap:8px;margin:14px 0 10px}
.score b{font-size:60px;line-height:1;font-weight:500;letter-spacing:-.02em}
.score span{font-size:15px;color:var(--ink-3)}
.score svg{margin-left:auto;align-self:center}
.low .score b{color:var(--green)} .med .score b{color:var(--yellow)} .high .score b{color:var(--red)}
.meter{height:3px;background:var(--line-2);border-radius:2px;margin-bottom:16px;overflow:hidden}
.meter i{display:block;height:100%;border-radius:2px;background:currentColor}
.low .meter{color:var(--green)} .med .meter{color:var(--yellow)} .high .meter{color:var(--red)}
.adv{font-size:15px;line-height:1.45;color:#d3dcea;margin-bottom:18px}
.threats{margin-top:auto;padding-top:14px;border-top:1px solid var(--line);display:grid;gap:7px}
.threats li{display:flex;gap:10px;align-items:baseline;font-family:var(--mono);font-size:11.5px;color:var(--ink-2)}
.threats li::before{content:"";flex:none;width:6px;height:6px;border-radius:50%;background:currentColor;transform:translateY(-1px)}
.low .threats li::before{color:var(--green)} .med .threats li::before{color:var(--yellow)} .high .threats li::before{color:var(--red)}
.threats li.none{color:var(--ink-3)} .threats li.none::before{color:var(--ink-4)}
.src{margin:-10px 0 12px;font-family:var(--mono);font-size:9.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-4)}
.src.ai{color:var(--cyan);opacity:.8}
.outlook{margin:-6px 0 16px;font-family:var(--mono);font-size:11px;line-height:1.5;color:var(--ink-3)}
.outlook b{font-weight:500;color:var(--ink-2)}
.outlook.med,.outlook.high{padding-left:10px;border-left:2px solid var(--yellow);color:var(--ink-2)}
.outlook.high{border-left-color:var(--red)}
.nodata .chip{color:var(--ink-3);border-color:var(--line-2)}
.nodata .score b{color:var(--ink-4)} .nodata .meter{color:var(--ink-4)}
.status.nodata{color:var(--ink-3);border-color:var(--line-2);background:var(--well)}
.flag{display:inline-block;margin-left:8px;padding:1px 8px;border-radius:999px;border:1px solid rgba(245,197,66,.5);
  color:var(--yellow);font-family:var(--mono);font-size:9.5px;letter-spacing:.14em;vertical-align:middle}

/* forecast + telemetry */
.mid{display:grid;grid-template-columns:minmax(300px,5fr) 8fr;gap:18px;margin-bottom:18px}
.gauge{display:flex;flex-direction:column;height:100%}
.gauge svg{width:100%;max-width:360px;height:auto;margin:0 auto;display:block}
.g-num{font-family:var(--sans);font-weight:500;font-size:50px;fill:var(--ink);letter-spacing:-.02em}
.g-sub{font-family:var(--mono);font-size:10px;letter-spacing:.14em;fill:var(--ink-3)}
.g-lbl{font-family:var(--mono);font-size:9.5px;fill:var(--ink-3)}
.needle{transform-origin:0 0;animation:sweep 1.1s cubic-bezier(.2,.8,.2,1) both}
@keyframes sweep{from{transform:rotate(135deg)}}
.pattern{margin-top:auto;border:1px solid var(--line-2);border-radius:14px;padding:12px 16px;text-align:center;background:var(--well)}
.pattern .cap{display:block;font-size:10px;margin-bottom:5px}
.pattern p{font-size:14px;font-weight:500}
.pattern p.hot{color:var(--orange)} .pattern p.calm{color:var(--ink-2)}
.pattern small{display:block;margin-top:7px;font-family:var(--mono);font-size:10.5px;color:var(--ink-3);line-height:1.5}
.telem{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.tile{position:relative;background:var(--well);border:1px solid var(--line);border-radius:12px;padding:14px 16px 14px}
.tile .cap{font-size:10px;display:block}
.tile .val{margin-top:8px;display:flex;align-items:baseline;gap:7px}
.tile .val b{font-size:30px;font-weight:500;letter-spacing:-.01em}
.tile .val.warn b{color:var(--yellow)} .tile .val.crit b{color:var(--red)}
.tile .val span{font-family:var(--mono);font-size:10.5px;color:var(--ink-3)}
.tile .bar{height:4px;margin-top:12px;border-radius:2px;background:var(--line)}
.tile .bar i{display:block;height:100%;border-radius:2px;background:var(--c)}
.tile .age{position:absolute;top:12px;right:14px;font-family:var(--mono);font-size:9.5px;letter-spacing:.08em;color:var(--yellow)}
.tile .age.missing{color:var(--ink-3)}

/* trend */
.trend-h{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:10px}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-family:var(--mono);font-size:11px;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:7px}
.legend i{width:16px;height:3px;border-radius:2px;background:var(--c)}
.scroll{overflow-x:auto}
.scroll svg{width:100%;min-width:640px;height:auto;display:block}
.t-ax{font-family:var(--mono);font-size:10px;fill:var(--ink-3)}
.t-title{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;text-transform:uppercase}
.hit{fill:transparent}.hit:hover{fill:rgba(34,211,238,.07)}
.foot{margin-top:18px;padding:0 4px;font-family:var(--mono);font-size:10.5px;line-height:1.7;color:var(--ink-3)}
.foot b{font-weight:500;color:var(--ink-2)} .foot .syn{color:var(--yellow)}

@media (max-width:980px){.sectors{grid-template-columns:1fr}.mid{grid-template-columns:1fr}
  .top{grid-template-columns:1fr;justify-items:start}.feed,.clock{justify-self:start;text-align:left}}
@media (max-width:520px){.telem{grid-template-columns:1fr}.score b{font-size:52px}.logo{font-size:21px}}
@media (prefers-reduced-motion:reduce){.needle,.feed.ok i{animation:none}}
</style>
"""


def fmt_sci(v: float, d: int = 1) -> str:
    if v == 0:
        return "0"
    ex = int(math.floor(math.log10(abs(v))))
    return f"{v / 10 ** ex:.{d}f}e{ex}"


def fmt_val(key: str, v: Optional[float]) -> str:
    if v is None:
        return "-"
    if key == "kp":             return f"{v:.1f}"
    if key == "xray_flux":      return fmt_sci(v)
    if key == "proton_flux":    return f"{v:.0f}" if v >= 100 else f"{v:.1f}" if v >= 10 else f"{v:.2f}"
    if key == "electron_flux":  return fmt_sci(v) if v >= 1000 else f"{v:.0f}"
    if key == "wind_speed":     return f"{v:.0f}"
    if key == "bz_gsm":         return f"{v:+.1f}"
    if key == "density":        return f"{v:.1f}"
    if key == "temperature":    return fmt_sci(v)
    return f"{v:.3g}"


# key, label, unit, colour, bar fraction, (warn, crit) severity test
TELEM = [
    ("kp",            "Kp index",         "/ 9",            "var(--cyan)"),
    ("xray_flux",     "X-ray flux",       "W/m²",           "var(--orange)"),
    ("proton_flux",   "Proton flux",      "pfu >10 MeV",    "var(--green)"),
    ("electron_flux", "Electron flux",    "e/cm²·s·sr >2 MeV", "var(--yellow)"),
    ("wind_speed",    "Solar wind speed", "km/s",           "var(--cyan)"),
    ("bz_gsm",        "Bz component",     "nT",             "var(--red)"),
    ("density",       "Density",          "p/cm³",          "var(--green)"),
    ("temperature",   "Temperature",      "K",              "var(--orange)"),
]


def bar_fraction(key: str, v: Optional[float]) -> float:
    if v is None:
        return 0.0
    if key == "kp":            return float(n_kp(v))
    if key == "xray_flux":     return n_xray(v)
    if key == "proton_flux":   return n_proton(v)
    if key == "electron_flux": return n_electron(v)
    if key == "wind_speed":    return float(_c01((v - 250) / 650))
    if key == "bz_gsm":        return max(float(n_bz_south(v)), 0.02)
    if key == "density":       return float(_c01(v / 30))
    if key == "temperature":   return float(_c01((math.log10(max(v, 1)) - 4) / 2.5))
    return 0.0


def severity(key: str, v: Optional[float]) -> str:
    if v is None:
        return ""
    t = {"kp": (4, 5), "xray_flux": (1e-5, 1e-4), "proton_flux": (10, 1000),
         "electron_flux": (1000, 10000), "wind_speed": (500, 700), "density": (15, 30)}
    if key == "bz_gsm":
        return "crit" if v <= -15 else "warn" if v <= -8 else ""
    if key in t:
        w, c = t[key]
        return "crit" if v >= c else "warn" if v >= w else ""
    return ""


def age_label(m: Optional[float]) -> str:
    if m is None:
        return ""
    if m < 90:    return f"{m:.0f}m"
    if m < 2880:  return f"{m / 60:.0f}h"
    return f"{m / 1440:.0f}d"


WARN_ICON = ('<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#ff4157" stroke-width="1.8" '
             'stroke-linecap="round" stroke-linejoin="round" aria-label="critical"><path d="M12 3 2.5 20h19L12 3z"/>'
             '<path d="M12 10v5M12 17.6v.1"/></svg>')


def _pct(p: Optional[float]) -> str:
    return "-" if p is None else ("<1%" if p < 0.01 else f"{p * 100:.0f}%")


def sector_html(s: Dict) -> str:
    lv = s["level"]
    chip = {"low": "LOW", "med": "MEDIUM", "high": "HIGH", "nodata": "NO DATA"}[lv]
    items = "".join(f"<li>{e(t)}</li>" for t in s["threats"]) or (
        '<li class="none">No observations</li>' if lv == "nodata" else '<li class="none">No active threats</li>')
    score = "-" if s["risk_score"] is None else s["risk_score"]
    width = 0 if s["risk_score"] is None else s["risk_score"]
    ol = s.get("outlook")
    outlook = ""
    if ol and ol.get("validated") and ol.get("p_elev") is not None:
        pe = ol["p_elev"]
        band = "likely" if pe >= 0.5 else "possible" if pe >= 0.05 else "unlikely"
        extra = ""
        if ol.get("severe_validated") and ol.get("p_severe") is not None and ol["p_severe"] >= 0.2:
            extra = " · severe <b>possible</b>" if ol["p_severe"] < 0.5 else " · severe <b>likely</b>"
        tip = f"model output {_pct(pe)} (rank-reliable, not calibrated across years)"
        outlook = (f'<p class="outlook {ol.get("level") or ""}" title="{e(tip)}">9 h outlook: elevated conditions '
                   f'<b>{band}</b>{extra}</p>')
    src = ('<p class="src ai">AI-written from live data</p>' if s.get("advisory_source") == "llm"
           else '<p class="src">rule-based text</p>')
    return f"""
    <article class="sec {lv}">
      <header><h3>{e(s['sector'])}</h3><span class="chip">{chip}</span></header>
      <div class="score"><b>{score}</b><span>/100</span>{WARN_ICON if lv == 'high' else ''}</div>
      <div class="meter"><i style="width:{width}%"></i></div>
      <p class="adv">{e(s['advisory'])}</p>{src}{outlook}
      <ul class="threats">{items}</ul>
    </article>"""


def gauge_html(ml: Dict) -> str:
    p = ml.get("storm_probability")
    cx, cy, r = 180, 152, 108
    start = 135.0

    def pt(rad, ang):
        a = math.radians(ang)
        return cx + rad * math.cos(a), cy + rad * math.sin(a)

    def arc(rad, a0, a1):
        x0, y0 = pt(rad, a0)
        x1, y1 = pt(rad, a1)
        return f"M{x0:.2f} {y0:.2f}A{rad} {rad} 0 {1 if a1 - a0 > 180 else 0} 1 {x1:.2f} {y1:.2f}"

    frac = 0.0 if p is None else max(0.004, min(1.0, p))
    end = start + 270 * frac
    colour = "#22d3ee" if frac < .35 else "#f5c542" if frac < .65 else "#ff4157"
    ticks, labels = [], []
    for i in range(21):
        ang = start + 270 * i / 20
        major = i % 5 == 0
        x0, y0 = pt(r + 11, ang)
        x1, y1 = pt(r + (21 if major else 16), ang)
        ticks.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                     f'stroke="{"#4a5d7c" if major else "#26354d"}" stroke-width="{1.6 if major else 1}"/>')
        if major:
            lx, ly = pt(r + 36, ang)
            labels.append(f'<text class="g-lbl" x="{lx:.1f}" y="{ly + 3:.1f}" text-anchor="middle">{i * 5}</text>')
    value_arc = "" if p is None else (
        f'<path d="{arc(r, start, end)}" stroke="{colour}" stroke-width="13" fill="none" stroke-linecap="round" '
        f'opacity=".35" filter="url(#blur)"/>'
        f'<path d="{arc(r, start, end)}" stroke="{colour}" stroke-width="13" fill="none" stroke-linecap="round"/>')
    needle = "" if p is None else (
        f'<g transform="translate({cx} {cy})"><g class="needle" style="transform:rotate({end:.1f}deg)">'
        f'<line x1="-16" y1="0" x2="{r - 22}" y2="0" stroke="#fb7c3c" stroke-width="2.6" stroke-linecap="round"/>'
        f'</g><circle r="7" fill="#0b121e" stroke="#fb7c3c" stroke-width="2.4"/></g>')
    shown = "-" if p is None else ("&lt;1%" if p < 0.01 else f"{p * 100:.0f}%")
    sub = ("FORECAST UNAVAILABLE" if p is None else
           f"P(Kp ≥ 5) · NEXT {ml.get('horizon_h', 9)} H" + ("" if ml.get("status") == "ok" else " · UNVALIDATED"))
    return f"""
    <svg viewBox="0 0 360 292" role="img" aria-label="Storm probability {shown}">
      <defs><filter id="blur" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="6"/></filter></defs>
      <path d="{arc(r, start, start + 270)}" stroke="#16213a" stroke-width="13" fill="none" stroke-linecap="round"/>
      {value_arc}{''.join(ticks)}{''.join(labels)}{needle}
      <text class="g-num" x="{cx}" y="{cy + 104}" text-anchor="middle">{shown}</text>
      <text class="g-sub" x="{cx}" y="{cy + 130}" text-anchor="middle">{sub}</text>
    </svg>"""


def telemetry_html(raw: Dict, status: Dict) -> str:
    tiles = []
    for key, label, unit, colour in TELEM:
        v = raw.get(key)
        st_ = status.get(key, {})
        sev = severity(key, v)
        tag = ""
        if st_.get("status") == "stale":
            tag = f'<span class="age">STALE {age_label(st_.get("age_min"))}</span>'
        elif st_.get("status") == "missing":
            tag = '<span class="age missing">NO DATA</span>'
        tiles.append(f"""
        <div class="tile" style="--c:{colour}">
          <span class="cap">{label}</span>{tag}
          <div class="val {sev}"><b>{fmt_val(key, v)}</b><span>{e(unit)}</span></div>
          <div class="bar"><i style="width:{bar_fraction(key, v) * 100:.0f}%"></i></div>
        </div>""")
    return f'<div class="telem">{"".join(tiles)}</div>'


def _smooth(points: List[Tuple[float, float]]) -> str:
    """Catmull-Rom spline through points as an SVG path (control points clamped: no overshoot)."""
    if len(points) == 1:
        x, y = points[0]
        return f"M{x:.1f} {y:.1f}h.1"
    d = [f"M{points[0][0]:.1f} {points[0][1]:.1f}"]
    for i in range(len(points) - 1):
        p0 = points[i - 1] if i > 0 else points[i]
        p1, p2 = points[i], points[i + 1]
        p3 = points[i + 2] if i + 2 < len(points) else p2
        lo_y, hi_y = min(p1[1], p2[1]), max(p1[1], p2[1])
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, min(max(p1[1] + (p2[1] - p0[1]) / 6, lo_y), hi_y))
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, min(max(p2[1] - (p3[1] - p1[1]) / 6, lo_y), hi_y))
        d.append(f"C{c1[0]:.1f} {c1[1]:.1f} {c2[0]:.1f} {c2[1]:.1f} {p2[0]:.1f} {p2[1]:.1f}")
    return "".join(d)


def _kp_colour(v: float) -> str:          # NOAA G-scale colours
    return "#ff4157" if v >= 8 else "#fb7c3c" if v >= 7 else "#f5c542" if v >= 5 else "#22d3ee"


# key, title, unit, colour, kind, (y_lo, y_hi), grid lines [(value, label)], formatter
TREND_PANELS = [
    ("kp", "Kp index", "", "#22d3ee", "bars", (0, 9), [(5, "G1")], "{:.1f}"),
    ("wind", "Solar wind speed", "km/s", "#f5c542", "line", (250, 800), [(400, ""), (600, "")], "{:.0f}"),
    ("bz", "Bz (GSM)", "nT", "#ff4157", "bz", (-20, 20), [(0, "")], "{:+.1f}"),
    ("xray", "X-ray flux 0.1-0.8 nm", "W/m²", "#fb7c3c", "log", (-8, -3),
     [(-8, "A"), (-7, "B"), (-6, "C"), (-5, "M"), (-4, "X")], "{:.1e}"),
    ("proton", "Protons >10 MeV", "pfu", "#34d399", "log", (-1, 3), [(1, "S1"), (2, "S2")], "{:.2f}"),
]


def trend_html(trend: List[Dict]) -> str:
    n = len(trend)
    if n < 2:
        return '<p class="cap">No trend data yet</p>'
    W, L, R, PH, GAP, T0, AX = 1000, 58, 16, 82, 16, 6, 28
    H = T0 + len(TREND_PANELS) * PH + (len(TREND_PANELS) - 1) * GAP + AX
    xs = [L + (W - L - R) * i / (n - 1) for i in range(n)]
    dx = (W - L - R) / (n - 1)
    out, y0 = [], T0
    for key, title, unit, colour, kind, (lo, hi), lines, fmt in TREND_PANELS:
        vals = [r.get(key) for r in trend]
        real = [v for v in vals if v is not None]
        if kind in ("line", "bz") and real:                  # widen the range if data leaves it
            lo, hi = min(lo, math.floor(min(real) / 5) * 5), max(hi, math.ceil(max(real) / 5) * 5)
        if kind == "bz" and real:
            m = max(abs(lo), abs(hi)); lo, hi = -m, m
        tf = (lambda v: math.log10(v)) if kind == "log" else (lambda v: v)
        Y = lambda v, y0=y0, lo=lo, hi=hi, tf=tf: y0 + PH * (1 - (min(max(tf(v), lo), hi) - lo) / (hi - lo))
        g = [f'<rect x="{L}" y="{y0}" width="{W - L - R}" height="{PH}" fill="#090f19" stroke="#172234" rx="6"/>']
        for gv, gl in lines:
            gy = Y(gv if kind != "log" else 10 ** gv)
            g.append(f'<line x1="{L}" x2="{W - R}" y1="{gy:.1f}" y2="{gy:.1f}" stroke="#1d2b42" stroke-dasharray="3 4"/>')
            if gl:
                lx, anc = (L + 8, "start") if kind == "bars" else (W - R - 6, "end")
                g.append(f'<text class="t-ax" x="{lx}" y="{gy - 3:.1f}" text-anchor="{anc}">{gl}</text>')
        ticks = ([0, 5, 9] if kind == "bars" else [lo, lo + (hi - lo) // 2, hi] if kind == "log"
                 else [lo, (lo + hi) / 2, hi])
        for tv in ticks:
            label = (f"1e{int(round(tv))}" if kind == "log" and abs(tv - round(tv)) < 1e-9 else
                     f"{10 ** tv:.0e}" if kind == "log" else f"{tv:g}")
            ty = y0 + PH * (1 - (tv - lo) / (hi - lo))
            g.append(f'<text class="t-ax" x="{L - 8}" y="{ty + 3:.1f}" text-anchor="end">{label}</text>')
        g.append(f'<text class="t-title" x="{L + 10}" y="{y0 + 15}" fill="{colour}">{e(title)}'
                 f'{f" · {unit}" if unit else ""}</text>')
        segs, seg = [], []
        for x, v in zip(xs, vals):
            if v is None:
                if seg: segs.append(seg)
                seg = []
            else:
                seg.append((x, v))
        if seg: segs.append(seg)
        if kind == "bars":
            for x, v in [s for sg in segs for s in sg]:
                h = max(PH * v / hi, 1.5)
                bx = min(max(x - dx / 2, L), W - R - dx)
                g.append(f'<rect x="{bx:.1f}" y="{y0 + PH - h:.1f}" width="{dx + .6:.1f}" height="{h:.1f}" fill="{_kp_colour(v)}" opacity=".85"/>')
        else:
            for sg in segs:
                pts = [(x, Y(v)) for x, v in sg]
                if kind == "bz":
                    z = Y(0)
                    neg = [(x, min(Y(v), 10 ** 9) if v < 0 else z) for x, v in sg]
                    g.append(f'<path d="M{neg[0][0]:.1f} {z:.1f}' + "".join(f"L{x:.1f} {y:.1f}" for x, y in neg)
                             + f'L{neg[-1][0]:.1f} {z:.1f}Z" fill="#ff4157" opacity=".16"/>')
                g.append(f'<path d="{_smooth(pts)}" fill="none" stroke="{colour}" stroke-width="1.8" '
                         f'stroke-linejoin="round" stroke-linecap="round"/>')
        first = next((i for i, v in enumerate(vals) if v is not None), None)
        if first is None:
            g.append(f'<text class="t-ax" x="{(L + W - R) / 2:.0f}" y="{y0 + PH / 2 + 3:.0f}" text-anchor="middle">no data in feed</text>')
        elif first > 3:
            g.append(f'<line x1="{xs[first]:.1f}" x2="{xs[first]:.1f}" y1="{y0 + 4}" y2="{y0 + PH - 4}" stroke="#3d4a5f" stroke-dasharray="2 3"/>'
                     f'<text class="t-ax" x="{xs[first] - 6:.1f}" y="{y0 + PH / 2 + 3:.0f}" text-anchor="end">feed starts here ({n - 1 - first} h of history)</text>')
        out.append("".join(g))
        y0 += PH + GAP
    xl = []
    for h in (48, 36, 24, 12, 0):
        i = min(n - 1, max(0, n - 1 - h))
        xl.append(f'<text class="t-ax" x="{xs[i]:.1f}" y="{H - 8}" text-anchor="{"end" if h == 0 else "middle" if h else "start"}">'
                  f'{"now" if h == 0 else f"-{h}h"}</text>')
    hits = []
    for i, r in enumerate(trend):
        tm = pd.Timestamp(r["t"]).strftime("%b %d %H:%M UTC")
        f = lambda k, fm: "no data" if r.get(k) is None else fm.format(r[k])
        tip = (f"{tm}\nKp {f('kp', '{:.1f}')}\nWind {f('wind', '{:.0f}')} km/s\nBz {f('bz', '{:+.1f}')} nT\n"
               f"X-ray {f('xray', '{:.1e}')} W/m²\nProton {f('proton', '{:.2f}')} pfu")
        hits.append(f'<rect class="hit" x="{xs[i] - dx / 2:.1f}" y="{T0}" width="{dx:.1f}" height="{H - T0 - AX}"><title>{e(tip)}</title></rect>')
    return f"""
    <div class="trend-h"><span class="cap">48-hour parameter trend</span>
      <span class="legend"><span>Kp bar colour = NOAA G-scale</span><span>dashed = scale thresholds</span></span></div>
    <div class="scroll"><svg viewBox="0 0 {W} {H}" role="img" aria-label="48 hour trend: Kp, solar wind speed, Bz, X-ray flux and proton flux, each on its own scale">
      {''.join(out)}{''.join(xl)}{''.join(hits)}
    </svg></div>"""


def footer_html(snap: Dict) -> str:
    m = snap.get("model") or {}
    L = m.get("lstm") or {}
    f = lambda x: "n/a" if x is None else f"{x:.2f}"
    if L:
        yrs = m.get("years") or []
        span = f"{min(yrs)}-{max(yrs)}" if yrs else "?"
        verdict = ('<b>validated</b>: beats the persistence baseline on held-out data' if L.get("validated") else
                   '<span class="syn">NOT validated - it does not beat the simple baselines on held-out data, '
                   'so treat the gauge as indicative only</span>')
        train = (f"<b>SpaceWeatherLSTM</b> trained on NASA OMNI2 {span} ({L.get('n_train', 0):,} windows, {e(m.get('source', ''))}). "
                 f"Held-out ROC-AUC <b>{f(L.get('auc'))}</b> · persistence {f(L.get('auc_persistence'))} · "
                 f"logistic {f(L.get('auc_logistic'))}; {verdict}.")
    else:
        why = snap.get("train_error") or "no trained weights"
        train = f'<span class="syn">Forecast model unavailable: {e(str(why))}</span>'
    g = snap.get("gfz")
    cross = ""
    if g and g.get("ok"):
        cross = (f" Kp cross-check vs GFZ Potsdam: mean difference {g['mean_abs_diff']} over {g['n']} blocks "
                 f"({g['within_1'] * 100:.0f}% within 1.0).")
    elif g:
        cross = f" GFZ cross-check unavailable ({e(g.get('error', ''))})."
    if snap.get("kp_source") == "GFZ Potsdam":
        cross += " Kp shown comes from GFZ because NOAA Kp was unreachable."
    am = snap.get("advisory_meta") or {}
    adv = (f" Advisory sentences are written by {e(am.get('model', 'an LLM'))} (free tier) from the computed facts only; numbers "
           f"it invents are rejected." if am.get("source") == "llm" else
           " Advisory sentences are rule-based text (no LLM key configured or the free API was unavailable"
           + (f": {e(am['error'])}" if am.get("error") else "") + ").")
    ts = snap.get("timestamp", "")[:19].replace("T", " ")
    return (f'<div class="foot">{train}<br>Live data: NOAA SWPC (DSCOVR/ACE/SOLAR-1 solar wind as 10-minute means, GOES X-ray, proton and '
            f'electron) fetched {ts} UTC.{cross} Sector level = nowcast of those observations: NOAA G/S/R scales for Kp, protons and X-ray; '
            f'the electron, Bz and pressure thresholds are my heuristics. The 9 h outlook comes from Random Forests trained on '
            f'what followed in OMNI2 (Kp, Dst and AE outcomes; radiation risk is nowcast only); it is shown as bands because the probabilities are not calibrated across years, and only '
            f'models that beat a Kp-only baseline on held-out data are used. Geomagnetic view: aurora from NOAA OVATION, coastlines from '
            f'Natural Earth, magnetopause from the Shue 1998 model; routes are great circles, not flights.{adv}<br>build {BUILD}</div>')


def dashboard_html(snap: Dict) -> str:
    raw, status = snap.get("raw_data", {}), snap.get("data_status", {})
    ml, ov = snap.get("ml_forecast", {}), snap.get("overall", {"level": "nodata", "label": "NO DATA", "text": ""})
    sectors = snap.get("sectors", {})

    have_any = any(v is not None for v in raw.values())
    states = [d.get("status") for d in status.values()]
    stale = sum(1 for s in states if s != "live")
    if snap.get("error"):
        feed = ("bad", "OFFLINE · SHOWING LAST SNAPSHOT" if have_any else "OFFLINE · NO LIVE DATA")
    elif stale:
        feed = ("warn", f"PARTIAL FEED · {stale} STALE")
    else:
        feed = ("ok", "LIVE FEED · NOAA SWPC")

    ts = pd.Timestamp(snap.get("timestamp") or pd.Timestamp.now(tz="UTC"))
    age = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 60
    notice = (f'<div class="notice">{e(snap["error"])}'
              + (" Values below are the last real readings, marked stale." if have_any else " Nothing is shown rather than guessing.")
              + "</div>") if snap.get("error") else ""

    cards = "".join(sector_html(sectors[k]) for k in ("satellite", "aviation", "power_grid") if k in sectors)
    if snap.get("kp_source") == "GFZ Potsdam":
        feed = ("warn", "KP FROM GFZ · NOAA KP DOWN") if feed[0] == "ok" else feed

    pat = ml.get("pattern", "-")
    hot = "calm" if ml.get("pattern_quiet", True) else "hot"
    drivers = ml.get("drivers") or []
    pa = ml.get("peak_attention_hours_ago")
    cov = ml.get("coverage", 0.0)
    detail = []
    if ml.get("storm_probability") is None:
        detail.append(ml.get("reason") or "Forecast unavailable")
    else:
        detail.append("Forecast driven by " + ", ".join(drivers) if drivers else "No single input above its typical level")
    detail.append(f"input {ml.get('window_h', 24)} h · observed coverage {cov * 100:.0f}%"
                  + ("" if pa is None else f" · top attention weight {'at T-0' if pa == 0 else f'at T-{pa}h'}"))

    return f"""
    <div class="ar">
      <div class="top">
        <div class="logo">ASTRORISK<small>SPACE WEATHER RISK OPS</small></div>
        <div class="feed {feed[0]}"><i></i>{feed[1]}</div>
        <div class="clock"><span id="utc">{pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S')} UTC</span><span id="age" data-ts="{e(ts.isoformat())}">updated {age:.0f} min ago</span></div>
      </div>
      {notice}
      <div class="status {ov.get('level', 'nodata')}"><b>OVERALL STATUS</b>{e(ov.get('label', ''))}: {e(ov.get('text', ''))}</div>
      <section class="sectors">{cards}</section>
      <section class="mid">
        <div class="panel gauge">
          <span class="cap">Storm probability</span>
          {gauge_html(ml)}
          <div class="pattern"><span class="cap">Detected pattern</span><p class="{hot}">{e(pat)}</p>
            <small>{e(detail[0])}<br>{e(detail[1])}</small></div>
        </div>
        <div class="panel"><span class="cap">Live telemetry</span>{telemetry_html(raw, status)}</div>
      </section>
      {geoviz.panel_html(snap.get('viz'))}
      <section class="panel" style="margin-top:18px">{trend_html(snap.get('trend', []))}</section>
      {footer_html(snap)}
    </div>"""


SCRIPT = """
<script>
(function () {
  function fit() {                       // size the iframe to its content
    try {
      var h = Math.ceil(document.body.getBoundingClientRect().height);
      if (window.frameElement && h > 0) { window.frameElement.style.height = h + 'px'; }
    } catch (e) {}
  }
  if (window.ResizeObserver) { new ResizeObserver(fit).observe(document.body); }
  window.addEventListener('load', fit); fit();
  var utc = document.getElementById('utc'), age = document.getElementById('age');
  function tick() {
    var now = new Date();
    if (utc) { utc.textContent = now.toISOString().slice(0, 19).replace('T', ' ') + ' UTC'; }
    if (age) {
      var m = Math.max(0, Math.round((now - new Date(age.dataset.ts)) / 60000));
      age.textContent = 'updated ' + (m < 1 ? 'just now' : m + ' min ago');
    }
  }
  tick(); setInterval(tick, 1000);
})();
</script>
"""


def dashboard_doc(snap: Dict) -> str:
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'{CSS}{geoviz.CSS}</head><body>{dashboard_html(snap)}{SCRIPT}{geoviz.SCRIPT}</body></html>')


def render_iframe(doc: str) -> None:
    """st.iframe sizes itself (new Streamlit); older versions use components.html."""
    if hasattr(st, "iframe"):
        st.iframe(doc, height="content")
    else:
        import streamlit.components.v1 as components
        components.html(doc, height=1500, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
# 9. APP
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False, ttl=600)
def get_models() -> Dict:
    """Trained models from disk, or trained now from real OMNI2 data. Never synthetic."""
    meta = load_meta()
    err = None
    if not weights_are_fresh(meta):
        try:
            with st.status("Training on NASA OMNI2 (first run, a few minutes)", expanded=True) as status:
                meta = train_all(log=st.write)
                status.update(label="Models trained", state="complete", expanded=False)
        except TrainingDataUnavailable as exc:
            err = str(exc)
        except Exception as exc:                       # keep the dashboard alive, say what happened
            err = f"training failed: {exc}"
        if err:
            meta = load_meta()
            if not weights_are_fresh(meta, allow_stale=True):
                return {"lstm": None, "rf": None, "meta": None, "error": err}
    try:
        model = SpaceWeatherLSTM()
        model.load_state_dict(torch.load(LSTM_PATH, map_location="cpu"))
        bundle = joblib.load(SECTOR_PATH)
        return {"lstm": LSTMPredictor(model, meta), "rf": SectorModels(bundle, meta["sectors"]),
                "meta": meta, "error": err}
    except Exception as exc:
        return {"lstm": None, "rf": None, "meta": None, "error": f"could not load weights: {exc}"}


def _advisory_cache_ok(cached: Dict) -> bool:
    """A cached snapshot is stale for our purposes if it predates the advisory step, or was made
    without an LLM key while one is configured now."""
    am = cached.get("advisory_meta")
    if not am:
        return False
    return not (am.get("source") == "rules" and groq_key()
                and str(am.get("error", "")).startswith("GROQ_API_KEY not set"))


def get_snapshot(models: Dict, force: bool) -> Dict:
    cached = load_snapshot()
    if cached and not force:
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(cached["timestamp"])).total_seconds()
            if age < REFRESH_INTERVAL and cached.get("model", {}).get("version", 0) == (models.get("meta") or {}).get("version", 0) \
                    and "driver_scores" in cached and "trend" in cached and cached.get("build") == BUILD \
                    and _advisory_cache_ok(cached):
                return cached
        except Exception:
            pass
    return run_pipeline(models, cached, force)


@st.fragment(run_every=REFRESH_INTERVAL)
def dashboard(models: Dict) -> None:
    force = st.session_state.pop("force_refresh", False)
    with st.spinner("Fetching NOAA SWPC feeds"):
        snap = get_snapshot(models, force)
    render_iframe(dashboard_doc(snap))
    if st.session_state.get("show_raw"):
        st.json(snap, expanded=False)


def main() -> None:
    st.set_page_config(page_title="AstroRisk", page_icon="🛰", layout="wide",
                       initial_sidebar_state="collapsed")
    st.html(PAGE_CSS)
    with st.sidebar:
        st.button("Fetch live data now",
                  on_click=lambda: st.session_state.update(force_refresh=True))
        st.checkbox("Show raw snapshot JSON", key="show_raw")
        st.caption(f"Auto-refresh every {REFRESH_INTERVAL // 60} min. Snapshot: {CACHE_PATH}")
        st.caption(f"Build {BUILD}")
    dashboard(get_models())


if __name__ == "__main__":
    main()