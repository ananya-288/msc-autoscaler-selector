#!/usr/bin/env python3
"""
train_selector.py - RQ2: can a classifier predict, one minute ahead, which mechanism
(HPA, KEDA, vertical) will be best for a service?

    python train_selector.py --root C:\\shpa\\check\\shpa --floor 50

Inputs (from the backup):
  results/mech/<mech>_<shape>_run1_metrics.csv   (collector, 15-s samples)
  results/labels/<shape>_run1_w60_floor<F>_labels.csv   (make_labels.py output)

Features for window w use ONLY windows w-1 and w-2 (what a controller knows when it decides),
and ONLY traffic signals that are the same whatever mechanism ran (averaged over the 3 runs):
  rps, rps slope, burstiness (std/mean inside the minute), peak/mean, frontend rps + slope,
  a naive next-minute forecast (rps + slope), and the service (one-hot).
Label: the best mechanism in window w.
Evaluation: leave-one-shape-out (train on 3 shapes, test on the 4th), macro-F1, and REGRET =
score(chosen) - score(best), compared with always choosing one mechanism.
"""
import argparse, glob, os, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--floor", type=int, default=50)
ap.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8])
ap.add_argument("--runs", nargs="+", type=int, default=[1])
ap.add_argument("--avg", action="store_true", help="one label per service-minute from the AVERAGE score over --runs")
ap.add_argument("--task", choices=["3class", "binary"], default="3class",
                help="binary = vertical (scale up) vs horizontal (scale out = HPA or KEDA)")
ap.add_argument("--lags", action="store_true", help="add time-series features (previous minutes, rolling mean)")
ap.add_argument("--clean", action="store_true",
                help="train only on reliable windows (runs agree on the winner); test on ALL windows")
ap.add_argument("--shapes", nargs="+", default=["burst", "rise", "gentle", "decline"])
a = ap.parse_args()
MECH = ["hpa", "keda", "vertical"]
R = a.root

def window_traffic(shape, run):
    runs = run if isinstance(run, (list, tuple)) else [run]
    d = pd.concat([pd.read_csv(os.path.join(R, "results", "mech", f"{m}_{shape}_run{r}_metrics.csv")) for m in MECH for r in runs])
    d = d[d.service != "redis-cart"].copy()
    d["win"] = (d.t_sec // 60).astype(int)
    g = d.groupby(["service", "win"]).rps
    w = pd.DataFrame({"rps": g.mean(), "rps_std": g.std(), "rps_max": g.max()}).reset_index()
    w["burst"] = (w.rps_std / w.rps.replace(0, np.nan)).fillna(0)
    w["peak_ratio"] = (w.rps_max / w.rps.replace(0, np.nan)).fillna(1)
    w = w.sort_values(["service", "win"])
    w["slope"] = w.groupby("service").rps.diff().fillna(0)
    if a.lags:                                             # time-series history (earlier minutes)
        g2 = w.groupby("service")
        w["rps_lag1"] = g2.rps.shift(1).fillna(w.rps)
        w["rps_lag2"] = g2.rps.shift(2).fillna(w.rps)
        w["slope_lag1"] = g2.slope.shift(1).fillna(0)
        w["rps_roll3"] = g2.rps.transform(lambda x: x.rolling(3, min_periods=1).mean())
        w["accel"] = w.slope - w.slope_lag1                # is the rise speeding up?
    fe = w[w.service == "frontend"][["win", "rps", "slope"]].rename(columns={"rps": "fe_rps", "slope": "fe_slope"})
    w = w.merge(fe, on="win", how="left")
    w["forecast"] = (w.rps + w.slope).clip(lower=0)
    w["shape"] = shape; w["run"] = "avg" if isinstance(run, (list, tuple)) else run
    return w

rows = []
def agree_flags(shape):
    tag = f"_floor{a.floor}" if a.floor else ""
    ls = [pd.read_csv(os.path.join(R, "results", "labels", f"{shape}_run{r}_w60{tag}_labels.csv"))[["service", "win", "label"]]
          .rename(columns={"label": f"l{r}"}) for r in a.runs]
    m = ls[0]
    for x in ls[1:]:
        m = m.merge(x, on=["service", "win"])
    labs = m[[c for c in m.columns if c.startswith("l")]]
    if a.task == "binary":
        labs = labs.replace({"hpa": "horizontal", "keda": "horizontal"})
    m["reliable"] = labs.nunique(axis=1) == 1
    return m[["service", "win", "reliable"]]

def avg_labels(shape):
    tag = f"_floor{a.floor}" if a.floor else ""
    det = pd.concat([pd.read_csv(os.path.join(R, "results", "labels", f"{shape}_run{r}_w60{tag}_window_details.csv"))
                     for r in a.runs])
    sc = det.groupby(["mechanism", "service", "win"]).score.mean().reset_index()
    wide = sc.pivot_table(index=["service", "win"], columns="mechanism", values="score").dropna().reset_index()
    wide["label"] = wide[MECH].idxmin(axis=1)
    return wide

for s in a.shapes:
    if a.avg:
        tr = window_traffic(s, a.runs)
        lab = avg_labels(s).merge(agree_flags(s), on=["service", "win"], how="left")
        feat = tr.copy(); feat["win"] = feat.win + 1
        rows.append(lab.merge(feat, on=["service", "win"], how="inner"))
        continue
    for run in a.runs:
        tr = window_traffic(s, run)
        tag = f"_floor{a.floor}" if a.floor else ""
        lab_path = os.path.join(R, "results", "labels", f"{s}_run{run}_w60{tag}_labels.csv")
        lab = pd.read_csv(lab_path)
        feat = tr.copy(); feat["win"] = feat.win + 1          # features of w-1 describe decision for w
        m = lab.merge(feat, on=["service", "win"], how="inner")
        rows.append(m)
D = pd.concat(rows, ignore_index=True)
FEATS = ["rps", "slope", "burst", "peak_ratio", "fe_rps", "fe_slope", "forecast"]
if a.lags:
    FEATS += ["rps_lag1", "rps_lag2", "slope_lag1", "rps_roll3", "accel"]
if "reliable" not in D:
    D["reliable"] = True
D["reliable"] = D["reliable"].fillna(False).astype(bool)
if a.task == "binary":
    D["label"] = D["label"].replace({"hpa": "horizontal", "keda": "horizontal"})
CL = ["horizontal", "vertical"] if a.task == "binary" else MECH
SCORECOL = {"hpa": "hpa", "keda": "keda", "vertical": "vertical", "horizontal": "keda"}  # horizontal acted as KEDA
os.makedirs(os.path.join(R, "results", "selector"), exist_ok=True)
runs_tag = ("_avg" if a.avg else "_runs") + "".join(map(str, a.runs)) + \
    ("_binary" if a.task == "binary" else "") + ("_lags" if a.lags else "") + ("_clean" if a.clean else "")
ds_path = os.path.join(R, "results", "selector", f"training_dataset_floor{a.floor}{runs_tag}.csv")
D[["shape", "run", "service", "win"] + FEATS + MECH + ["label", "reliable"]].rename(
    columns={"win": "minute", "hpa": "score_hpa", "keda": "score_keda", "vertical": "score_vertical"}
).to_csv(ds_path, index=False)
print(f"Saved the exact model input (features + label) to {ds_path}")
X_num = D[FEATS].astype(float)
X = pd.concat([X_num, pd.get_dummies(D.service, prefix="svc").astype(float)], axis=1)
y = D.label.values
print(f"Task: {a.task}; time-series features: {a.lags}; train on reliable windows only: {a.clean} "
      f"({D.reliable.mean()*100:.0f}% of windows reliable)")
print(f"Dataset: {len(D)} service-windows from {len(a.shapes)} shapes x runs {a.runs}; label share:",
      (pd.Series(y).value_counts(normalize=True) * 100).round(1).to_dict(), "\n")

MODELS = {
 "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
 "DecisionTree": DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=0),
 "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=0),
 "ExtraTrees": ExtraTreesClassifier(n_estimators=500, class_weight="balanced", min_samples_leaf=2, random_state=0),
 "GradientBoosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1, class_weight="balanced", random_state=0),
 "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
 "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=0)),
}
try:
    from xgboost import XGBClassifier
    MODELS["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1, random_state=0, verbosity=0)
except Exception:
    print("(xgboost not installed - using sklearn GradientBoosting instead)\n")

def regret_vec(pred, idx):
    sc = D.loc[idx, MECH].values
    chosen = sc[np.arange(len(pred)), [MECH.index(SCORECOL[p]) for p in pred]]
    return chosen - sc.min(axis=1)

def regret(pred, idx):
    return float(np.mean(regret_vec(pred, idx)))

PER = {}

res = []
CHANCE = {"chance_uniform": DummyClassifier(strategy="uniform", random_state=0),
          "chance_stratified": DummyClassifier(strategy="stratified", random_state=0)}
for name, model in list(MODELS.items()) + list(CHANCE.items()) + [(f"always_{m}", None) for m in CL]:
    preds, idxs = [], []
    for test_shape in a.shapes:
        te = D.index[D["shape"] == test_shape]; trn = D.index[D["shape"] != test_shape]
        if a.clean:
            trn = trn[D.loc[trn, "reliable"].values]
        if model is None:
            p = np.array([name.split("_", 1)[1]] * len(te))
        else:
            if name == "XGBoost":
                enc = {m: i for i, m in enumerate(CL)}
                model.fit(X.loc[trn], [enc[v] for v in y[trn]])
                p = np.array([CL[i] for i in model.predict(X.loc[te])])
            else:
                model.fit(X.loc[trn], y[trn]); p = model.predict(X.loc[te])
        preds.append(p); idxs.append(te)
    P = np.concatenate(preds); I = np.concatenate(idxs)
    PER[name] = pd.Series(regret_vec(P, I), index=I).sort_index()
    rel = D.loc[I, "reliable"].values
    res.append({"model": name, "accuracy": accuracy_score(y[I], P),
                "macro_F1": f1_score(y[I], P, average="macro", labels=CL),
                "macro_F1_reliable": f1_score(y[I][rel], P[rel], average="macro", labels=CL) if rel.any() else np.nan,
                "mean_regret": regret(P, I),
                **{f"picks_{m}_%": 100 * np.mean(P == m) for m in CL}})
R_ = pd.DataFrame(res).sort_values("mean_regret")
os.makedirs(os.path.join(R, "results", "selector"), exist_ok=True)
out = os.path.join(R, "results", "selector", f"rq2_leave_one_shape_out_floor{a.floor}{runs_tag}.csv")
R_.to_csv(out, index=False)
print("Leave-one-shape-out results (lower regret = better; compare with always_vertical):")
print(R_.round(3).to_string(index=False))
print(f"\nsaved {out}")

# noise ceiling: how well do the labels of one repeat "predict" the other repeat?
if len(a.runs) >= 2:
    tag = f"_floor{a.floor}" if a.floor else ""
    L = []
    for s in a.shapes:
        l1 = pd.read_csv(os.path.join(R, "results", "labels", f"{s}_run{a.runs[0]}_w60{tag}_labels.csv"))[["service", "win", "label"]]
        l2 = pd.read_csv(os.path.join(R, "results", "labels", f"{s}_run{a.runs[1]}_w60{tag}_labels.csv"))[["service", "win", "label"]]
        L.append(l1.merge(l2, on=["service", "win"], suffixes=("_a", "_b")))
    L = pd.concat(L)
    if a.task == "binary":
        L = L.replace({"hpa": "horizontal", "keda": "horizontal"})
    print(f"\nNoise ceiling: repeat {a.runs[1]} vs repeat {a.runs[0]} labels -> accuracy "
          f"{accuracy_score(L.label_a, L.label_b):.3f}, macro-F1 {f1_score(L.label_a, L.label_b, average='macro'):.3f}"
          "  (identical experiments; no model can be expected to exceed this by much)")

# paired comparison with the strongest single mechanism (same service-windows)
from scipy.stats import wilcoxon
base = PER["always_vertical"]
rng = np.random.default_rng(0)
print("\nPaired comparison with always_vertical (per service-window regret):")
for name in [m for m in R_.model if not m.startswith(("always_", "chance_"))]:
    diff = (PER[name] - base).values
    boots = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    try:
        p = wilcoxon(PER[name].values, base.values).pvalue
    except ValueError:
        p = float("nan")
    print(f"  {name:20s} mean difference {diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  Wilcoxon p={p:.3g}")
print("  (negative = lower regret than always_vertical)")

# confidence-gated selectors: stay on vertical unless the model is confident about HPA/KEDA
print("\nConfidence-gated selector (switch away from vertical only if P(best) >= threshold):")
print(f"  {'model':20s} {'thr':>4s} {'regret':>7s} {'switch%':>8s}  diff vs always_vertical [95% CI]   Wilcoxon p")
GATED = []
for name, model in MODELS.items():
    if name in ("DecisionTree",):
        continue
    probs, idxs = [], []
    for test_shape in a.shapes:
        te = D.index[D["shape"] == test_shape]; trn = D.index[D["shape"] != test_shape]
        if a.clean:
            trn = trn[D.loc[trn, "reliable"].values]
        if name == "XGBoost":
            enc = {m: i for i, m in enumerate(CL)}
            model.fit(X.loc[trn], [enc[v] for v in y[trn]]); pr = model.predict_proba(X.loc[te]); cls = CL
        else:
            model.fit(X.loc[trn], y[trn]); pr = model.predict_proba(X.loc[te]); cls = list(model.classes_)
        pr = pd.DataFrame(pr, columns=cls).reindex(columns=CL, fill_value=0.0).values
        probs.append(pr); idxs.append(te)
    PR = np.vstack(probs); I = np.concatenate(idxs)
    for thr in a.thresholds:
        best = PR.argmax(axis=1)
        pick = np.where((PR.max(axis=1) >= thr) & (best != CL.index("vertical")), best, CL.index("vertical"))
        P = np.array([CL[i] for i in pick])
        rv = pd.Series(regret_vec(P, I), index=I).sort_index()
        diff = (rv - base).values
        boots = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        try:
            pw = wilcoxon(rv.values, base.values).pvalue
        except ValueError:
            pw = float("nan")
        GATED.append({"model": name, "threshold": thr, "regret": rv.mean(), "switch_%": 100 * np.mean(P != "vertical"),
                      "macro_F1": f1_score(y[I], P, average="macro", labels=CL),
                      "diff": diff.mean(), "ci_low": lo, "ci_high": hi, "wilcoxon_p": pw})
        flag = "  *" if hi < 0 else ""
        print(f"  {name:20s} {thr:4.1f} {rv.mean():7.4f} {100*np.mean(P!='vertical'):7.1f}%  "
              f"{diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]   {pw:.3g}{flag}")
print(f"  always_vertical           {base.mean():7.4f}     0.0%   (* = 95% CI entirely below zero)")
pd.DataFrame(GATED).to_csv(out.replace(".csv", "_gated.csv"), index=False)

# feature importance from a Random Forest trained on all data (for interpretation)
rf = RandomForestClassifier(n_estimators=500, max_depth=6, class_weight="balanced", random_state=0).fit(X, y)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
print("\nTop features (Random Forest importance):"); print(imp.round(3).to_string())
