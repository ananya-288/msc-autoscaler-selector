#!/usr/bin/env python3
"""
build_final_model.py - fix the FINAL selector (tuned soft-voting ensemble) before the live test.

    python build_final_model.py --data results/selector/training_dataset_floor50_avg12_binary_lags.csv

Same procedure as one outer fold of tune_selector.py, but now using ALL four training shapes:
  1. for MLP, ExtraTrees and kNN: pick the hyper-parameters with the best leave-one-shape-out
     macro-F1 (same grids as tune_selector.py);
  2. pick the decision threshold for "horizontal" on the leave-one-shape-out probabilities of the
     soft-voting ensemble (average of the three tuned models).
The settings are written to results/selector/final_selector.json. live_selector.py (mode
"ensemble") trains the three models on the same CSV with these settings at start-up, so no
binary model file has to be copied between machines.
NOTE: the macro-F1 printed here is the score used to CHOOSE the settings (optimistic);
the unbiased estimate is the nested result from tune_selector.py.
"""
import argparse, itertools, json, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="results/selector/training_dataset_floor50_avg12_binary_lags.csv")
ap.add_argument("--out", default="results/selector/final_selector.json")
a = ap.parse_args()

FEATS_ALL = ["rps", "slope", "burst", "peak_ratio", "fe_rps", "fe_slope", "forecast",
             "rps_lag1", "rps_lag2", "slope_lag1", "rps_roll3", "accel"]
D = pd.read_csv(a.data)
FEATS = [c for c in FEATS_ALL if c in D]
X = pd.concat([D[FEATS].astype(float), pd.get_dummies(D.service, prefix="svc").astype(float)], axis=1)
COLS = list(X.columns); X = X.values
y = (D.label == "horizontal").astype(int).values
shape = D["shape"].values

def build(name, g):
    if name == "MLP":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=tuple(g["h"]), alpha=g["alpha"],
                                                             max_iter=2000, random_state=0))
    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=300, max_depth=g["depth"], min_samples_leaf=g["leaf"],
                                    max_features=g["feat"], class_weight="balanced", random_state=0, n_jobs=-1)
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=g["k"], weights=g["w"]))

GRIDS = {
    "MLP": [dict(h=list(h), alpha=al) for h, al in itertools.product([(16,), (32,), (64,), (32, 16)], [1e-4, 1e-3, 1e-2])],
    "ExtraTrees": [dict(depth=d, leaf=l, feat=f) for d, l, f in itertools.product([None, 6, 10], [1, 2, 5], ["sqrt", 0.5])],
    "kNN": [dict(k=k, w=w) for k, w in itertools.product([5, 7, 11, 15], ["uniform", "distance"])],
}
THRS = np.round(np.arange(0.20, 0.81, 0.05), 2)

def oof(model):
    p = np.zeros(len(y))
    for s in dict.fromkeys(shape):
        te = shape == s
        p[te] = clone(model).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
    return p

final = {"features": FEATS, "columns": COLS, "positive_class": "horizontal", "horizontal_acts_as": "keda",
         "training_data": a.data, "models": {}}
probs = []
for name, grid in GRIDS.items():
    scores = [f1_score(y, (oof(build(name, g)) >= 0.5).astype(int), average="macro") for g in grid]
    g = grid[int(np.argmax(scores))]
    final["models"][name] = g
    probs.append(oof(build(name, g)))
    print(f"{name:10s} chosen {g}  (selection macro-F1 {max(scores):.3f})")
ens = np.mean(probs, axis=0)
thr = float(max(THRS, key=lambda t: f1_score(y, (ens >= t).astype(int), average="macro")))
final["threshold"] = thr
print(f"Ensemble threshold for 'horizontal': {thr}  (selection macro-F1 "
      f"{f1_score(y, (ens >= thr).astype(int), average='macro'):.3f}; unbiased estimate = nested result 0.779)")
json.dump(final, open(a.out, "w"), indent=1)
print(f"saved {a.out}")
