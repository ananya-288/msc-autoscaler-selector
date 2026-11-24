#!/usr/bin/env python3
"""
posthoc_frontend.py - POST-HOC, OFFLINE: can the frontend decisions be improved?

    python training/posthoc_frontend.py --data results/selector/training_dataset_floor50_avg12_binary_lags.csv

Done after the live evaluation, so it cannot change the frozen selector or the live results.
Same nested leave-one-shape-out procedure and grids as training/tune_selector.py (soft-voting
ensemble of tuned MLP + ExtraTrees + kNN; hyper-parameters and thresholds chosen on the three
training shapes only). Four variants:
  base        the final feature set (reproduces the tuned ensemble of tune_selector.py)
  downstream  + traffic behind the frontend in the same (previous) minute: total request rate of
                all non-frontend services and its change from the minute before (traffic-only)
  fe_thr      base features, but a separate decision threshold for frontend rows
  both        downstream features + separate frontend threshold
Reported: macro-F1 (all rows), macro-F1 on frontend rows, mean regret, and a paired comparison of
each variant with base (per-row regret difference: bootstrap 95% CI and Wilcoxon).
The frontend has only 56 rows (19 horizontal), so frontend results are indicative only.
Output: results/selector/posthoc_frontend.csv and posthoc_frontend_per_service.csv
"""
import argparse, itertools, os, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
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
ap.add_argument("--out", default="results/selector/posthoc_frontend.csv")
a = ap.parse_args()

D = pd.read_csv(a.data)
BASE = [c for c in ["rps", "slope", "burst", "peak_ratio", "fe_rps", "fe_slope", "forecast",
                    "rps_lag1", "rps_lag2", "slope_lag1", "rps_roll3", "accel"] if c in D]
# traffic behind the frontend in the same window (all features already describe the previous minute)
down = (D[D.service != "frontend"].groupby(["shape", "minute"])
        .agg(down_rps=("rps", "sum"), down_slope=("slope", "sum")).reset_index())
D = D.merge(down, on=["shape", "minute"], how="left")
y = (D.label == "horizontal").astype(int).values
shape = D["shape"].values
is_fe = (D.service == "frontend").values
SC = D[["score_hpa", "score_keda", "score_vertical"]].values
SHAPES = list(dict.fromkeys(shape))

def matrix(feats):
    return pd.concat([D[feats].astype(float), pd.get_dummies(D.service, prefix="svc").astype(float)], axis=1).values

def mlp(h=(32,), alpha=1e-4):
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=h, alpha=alpha, max_iter=2000, random_state=0))
def et(depth=None, leaf=2, feat="sqrt"):
    return ExtraTreesClassifier(n_estimators=300, max_depth=depth, min_samples_leaf=leaf, max_features=feat,
                                class_weight="balanced", random_state=0, n_jobs=-1)
def knn(k=7, w="uniform"):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, weights=w))
GRIDS = {   # identical to tune_selector.py
    "MLP": [dict(h=h, alpha=al) for h, al in itertools.product([(16,), (32,), (64,), (32, 16)], [1e-4, 1e-3, 1e-2])],
    "ExtraTrees": [dict(depth=d, leaf=l, feat=f) for d, l, f in itertools.product([None, 6, 10], [1, 2, 5], ["sqrt", 0.5])],
    "kNN": [dict(k=k, w=w) for k, w in itertools.product([5, 7, 11, 15], ["uniform", "distance"])],
}
BUILD = {"MLP": mlp, "ExtraTrees": et, "kNN": knn}
THRS = np.round(np.arange(0.20, 0.81, 0.05), 2)

def oof(model, X, idx):
    p = np.zeros(len(idx))
    for s in dict.fromkeys(shape[idx]):
        te = shape[idx] == s
        p[te] = clone(model).fit(X[idx][~te], y[idx][~te]).predict_proba(X[idx][te])[:, 1]
    return p

def best_thr(p, yy):
    return max(THRS, key=lambda t: f1_score(yy, (p >= t).astype(int), average="macro"))

def run(feats, fe_threshold):
    X = matrix(feats); P = np.zeros(len(y), int)
    for s in SHAPES:                                         # outer loop
        te = shape == s; tr = ~te; tri = np.where(tr)[0]
        models = []
        for name in ["MLP", "ExtraTrees", "kNN"]:            # inner loop: hyper-parameters
            sc = [f1_score(y[tri], (oof(BUILD[name](**g), X, tri) >= 0.5).astype(int), average="macro")
                  for g in GRIDS[name]]
            models.append(BUILD[name](**GRIDS[name][int(np.argmax(sc))]))
        inner = np.mean([oof(m, X, tri) for m in models], axis=0)
        thr = best_thr(inner, y[tri])
        thr_fe = best_thr(inner[is_fe[tri]], y[tri][is_fe[tri]]) if fe_threshold else thr
        prob = np.mean([clone(m).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1] for m in models], axis=0)
        t = np.where(is_fe[te], thr_fe, thr)
        P[te] = (prob >= t).astype(int)
    return P

def regret(P):
    chosen = np.where(P == 1, SC[:, 1], SC[:, 2])           # horizontal is executed as KEDA
    return chosen - SC.min(axis=1)

VARIANTS = {"base": (BASE, False), "downstream": (BASE + ["down_rps", "down_slope"], False),
            "fe_thr": (BASE, True), "both": (BASE + ["down_rps", "down_slope"], True)}
preds = {}
for k, (feats, fet) in VARIANTS.items():
    print(f"running {k} ...", flush=True)
    preds[k] = run(feats, fet)

rng = np.random.default_rng(0); rb = regret(preds["base"]); rows = []
for k, P in preds.items():
    r = regret(P); d = r - rb
    boots = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    try:
        pw = wilcoxon(r, rb).pvalue if k != "base" else np.nan
    except ValueError:
        pw = np.nan
    rows.append({"variant": k, "macro_F1": f1_score(y, P, average="macro"),
                 "frontend_macro_F1": f1_score(y[is_fe], P[is_fe], average="macro"),
                 "mean_regret": r.mean(), "frontend_regret": r[is_fe].mean(),
                 "regret_diff_vs_base": d.mean(), "ci_low": lo, "ci_high": hi, "wilcoxon_p": pw})
T = pd.DataFrame(rows)
per = pd.DataFrame({k: D.assign(p=P).groupby("service")[["label", "p"]].apply(
        lambda g: f1_score((g.label == "horizontal").astype(int), g.p, average="macro")) for k, P in preds.items()})
os.makedirs(os.path.dirname(a.out), exist_ok=True)
T.to_csv(a.out, index=False); per.to_csv(a.out.replace(".csv", "_per_service.csv"))
pd.set_option("display.width", 200)
print("\nPOST-HOC offline comparison (nested leave-one-shape-out):")
print(T.round(4).to_string(index=False))
print("\nmacro-F1 per service:"); print(per.round(3).to_string())
print(f"\nsaved {a.out}")
