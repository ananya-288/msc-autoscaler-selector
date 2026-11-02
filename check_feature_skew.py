#!/usr/bin/env python3
"""
check_feature_skew.py - are the features the live selector computed similar to the training features?

    python check_feature_skew.py --root .

Training (train_selector.py) computes burst and peak_ratio over the 15-s samples of ALL runs of a shape
pooled together (3 mechanisms x 2 repeats = 24 samples per minute). The live selector computes them from
ONE run (4 samples per minute). This script quantifies that difference in three ways:
  A. training features as used           (pooled over runs)
  B. training data, computed per run      (exactly how the live selector computes them), then averaged
  C. live values from the selector runs   (results/mech/selector_eval_heldout_run*_decisions.csv)
A vs B shows the size of the train/live skew on the SAME data; C shows what the live selector saw.
"""
import argparse, glob, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--shapes", nargs="+", default=["burst", "rise", "gentle", "decline"])
ap.add_argument("--runs", nargs="+", type=int, default=[1, 2])
a = ap.parse_args()
P = lambda *x: os.path.join(a.root, "results", *x)
MECH = ["hpa", "keda", "vertical"]

def feats(d):
    d = d[d.service != "redis-cart"].copy(); d["win"] = (d.t_sec // 60).astype(int)
    g = d.groupby(["service", "win"]).rps
    w = pd.DataFrame({"rps": g.mean(), "sd": g.std(), "mx": g.max()}).reset_index()
    w["burst"] = (w.sd / w.rps.replace(0, np.nan)).fillna(0)
    w["peak_ratio"] = (w.mx / w.rps.replace(0, np.nan)).fillna(1)
    return w[["service", "win", "rps", "burst", "peak_ratio"]]

pooled, per_run = [], []
for s in a.shapes:
    files = [P("mech", f"{m}_{s}_run{r}_metrics.csv") for m in MECH for r in a.runs]
    files = [f for f in files if os.path.exists(f)]
    pooled.append(feats(pd.concat([pd.read_csv(f) for f in files])))
    pr = pd.concat([feats(pd.read_csv(f)) for f in files])
    per_run.append(pr.groupby(["service", "win"])[["rps", "burst", "peak_ratio"]].mean().reset_index())
A, B = pd.concat(pooled), pd.concat(per_run)

live = [pd.read_csv(f) for f in sorted(glob.glob(P("mech", "selector_eval_heldout_run[0-9]_decisions.csv")))]
C = pd.concat(live) if live else pd.DataFrame(columns=["rps", "burst", "peak_ratio"])

rows = []
for col in ["rps", "burst", "peak_ratio"]:
    for name, df in [("A training (pooled, as used)", A), ("B training (per run, live-style)", B),
                     ("C live selector runs", C)]:
        if col in df and len(df):
            x = df[col].astype(float)
            rows.append({"feature": col, "source": name, "mean": x.mean(), "median": x.median(),
                         "p90": x.quantile(0.9), "n": len(x)})
T = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(T.round(4).to_string(index=False))
for col in ["burst", "peak_ratio"]:
    a_, b_ = A[col].mean(), B[col].mean()
    print(f"\n{col}: pooled training mean is {100 * (a_ / b_ - 1):+.0f}% vs the live-style per-run value on the same data")
os.makedirs(P("selector"), exist_ok=True)
T.to_csv(P("selector", "feature_skew_check.csv"), index=False)
print("\nsaved results/selector/feature_skew_check.csv")
