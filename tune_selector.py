#!/usr/bin/env python3
"""
tune_selector.py - one honest round of tuning for the binary selector (scale up vs scale out).

    python tune_selector.py --data results/selector/training_dataset_floor50_avg12_binary_lags.csv

Input: the exact training table written by
    python train_selector.py --root . --floor 50 --runs 1 2 --avg --task binary --lags

NESTED leave-one-shape-out (nothing is tuned on the shape being tested):
  outer loop : hold out 1 traffic shape for testing, keep 3 for training
  inner loop : on the 3 training shapes only, leave-one-shape-out again to
               (1) pick the hyper-parameters (best inner macro-F1) and
               (2) pick the decision threshold for "horizontal" (best inner macro-F1),
               then refit on all 3 training shapes and predict the held-out shape.
Models: MLP, ExtraTrees (class_weight balanced), kNN, and a soft-voting ensemble of the three
tuned models. Default (untuned) versions are reported next to them for comparison.
Regret: "horizontal" is executed as KEDA (as in the live selector).
"""
import argparse, itertools, os, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="results/selector/training_dataset_floor50_avg12_binary_lags.csv")
ap.add_argument("--out", default="results/selector/rq2_tuned_binary_lags.csv")
a = ap.parse_args()

D = pd.read_csv(a.data)
FEATS = [c for c in ["rps", "slope", "burst", "peak_ratio", "fe_rps", "fe_slope", "forecast",
                     "rps_lag1", "rps_lag2", "slope_lag1", "rps_roll3", "accel"] if c in D]
X = pd.concat([D[FEATS].astype(float), pd.get_dummies(D.service, prefix="svc").astype(float)], axis=1).values
y = (D.label == "horizontal").astype(int).values            # 1 = horizontal (scale out), 0 = vertical
shape = D["shape"].values
SC = D[["score_hpa", "score_keda", "score_vertical"]].values
SHAPES = list(dict.fromkeys(shape))
print(f"{len(D)} rows, {len(FEATS)} features + service; horizontal share {y.mean()*100:.1f}%")

def regret(pred, idx):
    chosen = np.where(pred == 1, SC[idx, 1], SC[idx, 2])      # horizontal -> KEDA score, vertical -> vertical
    return chosen - SC[idx].min(axis=1)

def mlp(h=(32,), alpha=1e-4):
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=h, alpha=alpha, max_iter=2000, random_state=0))
def et(depth=None, leaf=2, feat="sqrt"):
    return ExtraTreesClassifier(n_estimators=300, max_depth=depth, min_samples_leaf=leaf, max_features=feat,
                                class_weight="balanced", random_state=0, n_jobs=-1)
def knn(k=7, w="uniform"):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, weights=w))

GRIDS = {
    "MLP": [dict(h=h, alpha=al) for h, al in itertools.product([(16,), (32,), (64,), (32, 16)], [1e-4, 1e-3, 1e-2])],
    "ExtraTrees": [dict(depth=d, leaf=l, feat=f) for d, l, f in itertools.product([None, 6, 10], [1, 2, 5], ["sqrt", 0.5])],
    "kNN": [dict(k=k, w=w) for k, w in itertools.product([5, 7, 11, 15], ["uniform", "distance"])],
}
BUILD = {"MLP": mlp, "ExtraTrees": et, "kNN": knn}
THRS = np.round(np.arange(0.20, 0.81, 0.05), 2)

def oof_proba(model, idx):
    """inner leave-one-shape-out out-of-fold P(horizontal) on rows idx"""
    p = np.zeros(len(idx))
    for s in dict.fromkeys(shape[idx]):
        te = shape[idx] == s
        m = clone(model).fit(X[idx][~te], y[idx][~te])
        p[te] = m.predict_proba(X[idx][te])[:, 1]
    return p

def best_threshold(p, yy):
    return max(THRS, key=lambda t: f1_score(yy, (p >= t).astype(int), average="macro"))

results, preds, chosen = {}, {}, []
for name in ["MLP", "ExtraTrees", "kNN"]:
    for variant in ["default", "tuned"]:
        P = np.zeros(len(y), int)
        for s in SHAPES:                                     # OUTER loop
            te = shape == s; tr = ~te; tri = np.where(tr)[0]
            if variant == "default":
                model, thr = BUILD[name](), 0.5
            else:                                            # INNER loop on training shapes only
                scores = []
                for g in GRIDS[name]:
                    p = oof_proba(BUILD[name](**g), tri)
                    scores.append(f1_score(y[tri], (p >= 0.5).astype(int), average="macro"))
                g = GRIDS[name][int(np.argmax(scores))]
                model = BUILD[name](**g)
                thr = best_threshold(oof_proba(model, tri), y[tri])
                chosen.append({"model": name, "test_shape": s, "params": g, "threshold": thr,
                               "inner_macro_F1": max(scores)})
            m = clone(model).fit(X[tr], y[tr])
            P[te] = (m.predict_proba(X[te])[:, 1] >= thr).astype(int)
        preds[f"{name}_{variant}"] = P

# soft-voting ensemble of the three tuned models (tuning again done inside each outer fold)
P = np.zeros(len(y), int)
for s in SHAPES:
    te = shape == s; tr = ~te; tri = np.where(tr)[0]
    models = []
    for name in ["MLP", "ExtraTrees", "kNN"]:
        g = next(c["params"] for c in chosen if c["model"] == name and c["test_shape"] == s)
        models.append(BUILD[name](**g))
    inner = np.mean([oof_proba(m, tri) for m in models], axis=0)
    thr = best_threshold(inner, y[tri])
    chosen.append({"model": "Ensemble", "test_shape": s, "params": "soft vote of tuned MLP+ExtraTrees+kNN",
                   "threshold": thr, "inner_macro_F1": f1_score(y[tri], (inner >= thr).astype(int), average="macro")})
    prob = np.mean([clone(m).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1] for m in models], axis=0)
    P[te] = (prob >= thr).astype(int)
preds["Ensemble_tuned"] = P
preds["always_vertical"] = np.zeros(len(y), int)

idx = np.arange(len(y)); base = regret(preds["always_vertical"], idx)
rng = np.random.default_rng(0); rows = []
for k, P in preds.items():
    r = regret(P, idx); diff = r - base
    boots = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    try:
        pw = wilcoxon(r, base).pvalue if k != "always_vertical" else np.nan
    except ValueError:
        pw = np.nan
    rows.append({"model": k, "accuracy": accuracy_score(y, P), "macro_F1": f1_score(y, P, average="macro"),
                 "mean_regret": r.mean(), "diff_vs_vertical": diff.mean(), "ci_low": lo, "ci_high": hi,
                 "wilcoxon_p": pw, "horizontal_%": 100 * P.mean()})
T = pd.DataFrame(rows).sort_values("macro_F1", ascending=False)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
T.to_csv(a.out, index=False); pd.DataFrame(chosen).to_csv(a.out.replace(".csv", "_choices.csv"), index=False)
pd.set_option("display.width", 200)
print("\nNested leave-one-shape-out (tuning and thresholds chosen on training shapes only):")
print(T.round(4).to_string(index=False))
print("\nChosen settings per held-out shape:")
print(pd.DataFrame(chosen)[["model", "test_shape", "params", "threshold"]].to_string(index=False))
print(f"\nsaved {a.out}")
