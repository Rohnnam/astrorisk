"""Run on YOUR machine:  python train_check.py   -> paste the whole output back.
Trains on real OMNI2 and prints split sizes, per-feature signal, baselines and the verdict.
Takes a few minutes. Writes weights/ (the app will reuse them)."""
import numpy as np
import pandas as pd
import app
from sklearn.metrics import roc_auc_score

frame, source, years = app.load_training_frame(app.TRAIN_YEARS, print)
print("\nsource:", source, "| years:", years)
print("coverage per column (share of 3h blocks observed):")
print(frame.notna().mean().round(3).to_string())
print("\nproton (p10) coverage and outcome counts by year  <- explains the satellite/aviation test regime:")
fa = frame.asfreq("3h"); lab = app.sector_labels(fa)
yr = pd.DataFrame({"p10_observed": fa["p10"].notna().groupby(fa.index.year).mean().round(2),
                   "kp>=5 blocks": (fa["kp"] >= 5).groupby(fa.index.year).sum(),
                   "p10>=10 blocks": (fa["p10"] >= 10).groupby(fa.index.year).sum(),
                   "aviation_elev=1": (lab["aviation"]["elev"] == 1).groupby(fa.index.year).sum()})
print(yr.to_string())
X, y, cur, pos = app.make_windows(frame)
sp = app.split_windows(pos)
for k, m in sp.items():
    print(f"{k:5s}: {m.sum():6d} windows, {y[m].mean()*100:5.1f}% positive")
print("\nsingle-feature AUC on TEST (last block; 0.5 = no signal, <0.5 = inverted):")
for j, n in enumerate(app.FEATURES):
    print(f"  {n:18s} {roc_auc_score(y[sp['test']], X[sp['test']][:, -1, j]):.3f}")
print()
meta = app.train_all(print)
print("\nFINAL:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in meta["lstm"].items() if k != "baseline"})