# AstroRisk

Live space-weather risk dashboard for three sectors: **satellite operations**, **aviation and communications**,
and **power grids**. It reads public NOAA feeds, scores current conditions on NOAA's G/S/R scales, estimates the
chance of a geomagnetic storm in the next ~9 hours, and writes a short advisory for each sector.

Live demo: https://astroriskapp.streamlit.app/

## What you see

- **Overall status banner**: nominal, watch or alert, built from the sector states and outlooks.
- **Three sector cards**: a 0-100 score for conditions *now* (the strongest weighted driver), the drivers behind it,
  a 9-hour outlook band (unlikely / possible / likely), and an advisory paragraph.
- **Storm probability gauge**: LSTM estimate of P(Kp >= 5 within the next 9 h), plus a rule-based pattern label.
- **Live telemetry**: Kp, X-ray flux, proton flux, electron flux, solar-wind speed, Bz, density, temperature.
- **48-hour trends**: Kp, wind speed, Bz, X-ray, protons, with NOAA scale thresholds. Wind and Bz only have about
  24 h of live history; the charts mark where the feed starts instead of padding the gap.

## How it works

1. The server fetches NOAA SWPC feeds (planetary Kp, GOES X-ray / proton / electron, real-time solar wind and
   interplanetary magnetic field). If the NOAA Kp feed is stale it falls back to GFZ Potsdam.
2. Observations are turned into scores using NOAA's published G, S and R scale thresholds. Electron flux, Bz and
   dynamic-pressure thresholds are heuristics, not NOAA scales, and are labelled as such in the footer.
3. An LSTM with temporal attention predicts storm probability. Per-sector random forests give sector outlooks;
   the aviation outlook uses the LSTM probability, and the grid outlook takes the higher of forest and LSTM.
4. An LLM (Groq free tier) writes the three advisories from the computed numbers only. A guard rejects replies that
   contain numbers not present in the input. If the key is missing or the call fails, rule-based text is used and the
   footer says which source wrote it.
5. Everything refreshes every 5 minutes while a page is open. Nothing runs in the background when nobody is looking.

## Models and what they actually achieve

Trained on NASA OMNI2 hourly data (2017-2024) aggregated to 3-hour blocks. Test set is the last 15% of the
period; validation is interleaved weeks, with purging around splits.

| Model | Target | Test result (last training run) |
| --- | --- | --- |
| LSTM + attention | Kp >= 5 within next 9 h | ROC-AUC about 0.915, vs 0.900 for persistence and 0.911 for logistic regression; Brier about 0.038 vs 0.058 for climatology |
| Sector random forests | Elevated-impact outlook per sector | ROC-AUC about 0.90-0.93, 0.01-0.04 above a Kp-only baseline |

Exact figures for the committed weights are in `weights/train_log.txt` and `weights/meta_v4.json`.

A model is shown only if it beats its baseline on held-out data. Severe-impact sector models did not beat Kp alone,
so they are hidden. The LSTM's gain over persistence is small (about 0.015 AUC).

## Limitations

- Not an operational forecast. Use NOAA SWPC for anything that matters.
- Radiation risk is not modelled; only the current NOAA S-scale is reported.
- Sector outlooks are shown as bands, not percentages: the forest probabilities are not calibrated across years.
- Proton flux (>10 MeV) is observed in OMNI2 only for part of the training period, so it is not a model input.
- The training period ends in 2024. Weights older than 60 days trigger a retrain on startup.
- Advisory wording comes from an LLM and can vary. The number guard catches invented numbers, not invented claims.

## Run locally

```
python -m venv venv
venv\Scripts\activate          # Windows; on Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Optional: set `GROQ_API_KEY` (and `GROQ_MODEL`) to enable LLM-written advisories. Without it the rule-based text is used.

If `weights/` is missing or stale, the first start trains from NASA OMNI2, which takes a few minutes. No synthetic
data is ever substituted: if OMNI2 cannot be downloaded, training fails and says so.

Helpers:

- `python check_sources.py` tests every data source and reports which respond.
- `python train_check.py` trains from scratch and prints metrics, baselines and calibration.

## Files

- `app.py`: data fetching, models, scoring, advisory generation and the dashboard (single file).
- `weights/`: committed trained models, so the hosted app starts without retraining.
- `.streamlit/config.toml`: theme and server settings.

## Data sources

NOAA Space Weather Prediction Center, GFZ Potsdam (Kp), NASA OMNI2 (training), Groq (advisory text).
