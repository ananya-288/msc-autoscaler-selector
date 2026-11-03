#!/usr/bin/env python3
"""
make_figures.py - report figures, made only from the saved result files (no hand-typed numbers).

    python make_figures.py --root .

Writes PNG (300 dpi) and PDF (for LaTeX) to results/figures/:
  fig_tradeoff     RQ3: reserved CPU vs frontend SLO violations per method (mean +/- 95% CI, n = 3 runs)
  fig_latency      RQ3: user-facing p95 latency per method (Locust, mean +/- 95% CI, dots = runs)
  fig_f1_steps     RQ2: macro-F1 of the best model per step (3-class -> binary -> +time-series -> +clean
                   -> nested-tuned ensemble), with the chance level and always-vertical for each step
Inputs: results/eval/per_run.csv (analyze_eval.py) and results/selector/rq2_*.csv (train/tune_selector.py)
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
a = ap.parse_args()
P = lambda *x: os.path.join(a.root, "results", *x)
OUT = P("figures"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
ORDER = ["vertical", "hpa", "keda", "rules", "selector"]
NAMES = {"vertical": "Vertical", "hpa": "HPA", "keda": "KEDA", "rules": "Rule-based", "selector": "ML selector"}
COL = {"vertical": "#5B5F97", "hpa": "#028090", "keda": "#F4A259", "rules": "#8A8A8A", "selector": "#C0392B"}

def mci(x):
    x = np.asarray(x, float)
    return x.mean(), (stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0)

def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig); print("wrote", os.path.join(OUT, name + ".png/.pdf"))

# ---------- RQ3 ----------
R = pd.read_csv(P("eval", "per_run.csv"))
fig, ax = plt.subplots(figsize=(6.2, 4.4))
for m in ORDER:
    g = R[R.method == m]
    if g.empty: continue
    (cx, ce), (sy, se) = mci(g.cpu_alloc_m), mci(g["slo_viol_%"])
    ax.errorbar(cx, sy, xerr=ce, yerr=se, fmt="o", ms=9, capsize=4, color=COL[m], label=NAMES[m])
    ax.annotate(NAMES[m], (cx, sy), textcoords="offset points", xytext=(8, 6), fontsize=10, color=COL[m])
ax.set_xlabel("Reserved CPU, all services (millicores)")
ax.set_ylabel("Frontend SLO violations (%)\n(15-s samples with p95 > 500 ms)")
ax.set_title("Cost vs SLO violations on the held-out workload (mean ± 95% CI, 3 runs)", fontsize=11)
ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
save(fig, "fig_tradeoff")

fig, ax = plt.subplots(figsize=(6.2, 4.0))
for i, m in enumerate(ORDER):
    g = R[R.method == m]
    if g.empty: continue
    mu, h = mci(g.locust_p95_ms)
    ax.bar(i, mu, yerr=h, capsize=5, color=COL[m], width=0.6)
    ax.scatter([i] * len(g), g.locust_p95_ms, color="black", s=12, zorder=3)
    ax.text(i, mu + h + 20, f"{mu:.0f}", ha="center", fontsize=10)
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([NAMES[m] for m in ORDER])
ax.set_ylabel("User-facing p95 latency (ms, Locust)")
ax.set_title("p95 latency per method (mean ± 95% CI, dots = runs)", fontsize=11)
save(fig, "fig_latency")

# ---------- RQ2 ----------
steps = [("3 classes", "rq2_leave_one_shape_out_floor50_avg12.csv"),
         ("binary\n(up vs out)", "rq2_leave_one_shape_out_floor50_avg12_binary.csv"),
         ("+ time-series\nfeatures", "rq2_leave_one_shape_out_floor50_avg12_binary_lags.csv"),
         ("+ reliable\nlabels only", "rq2_leave_one_shape_out_floor50_avg12_binary_lags_clean.csv"),
         ("nested-tuned\nensemble", "rq2_tuned_binary_lags.csv")]
rows = []
for label, f in steps:
    path = P("selector", f)
    if not os.path.exists(path): print("missing", f); continue
    T = pd.read_csv(path)
    models = T[~T.model.str.startswith(("always_", "chance_"))]
    if "Ensemble_tuned" in set(T.model):
        best = T[T.model == "Ensemble_tuned"].iloc[0]
        chance = 0.5 if "binary" in f or "tuned" in f else np.nan
    else:
        best = models.sort_values("macro_F1", ascending=False).iloc[0]
        ch = T[T.model.str.startswith("chance_")]
        chance = ch.macro_F1.max() if len(ch) else np.nan
    av = T[T.model == "always_vertical"].macro_F1
    rows.append({"step": label, "best_model": best.model, "macro_F1": best.macro_F1,
                 "chance": chance, "always_vertical": av.iloc[0] if len(av) else np.nan})
S = pd.DataFrame(rows); S.to_csv(os.path.join(OUT, "fig_f1_steps_data.csv"), index=False)
fig, ax = plt.subplots(figsize=(7.2, 4.2))
x = np.arange(len(S))
ax.bar(x, S.macro_F1, color="#028090", width=0.6, label="best model")
for i, r in S.iterrows():
    ax.text(i, r.macro_F1 + 0.015, f"{r.macro_F1:.2f}\n{r.best_model.replace('_tuned', '')}", ha="center", fontsize=9)
ax.plot(x, S.chance, "k--", marker="_", ms=20, label="chance level (best dummy)")
ax.plot(x, S.always_vertical, ":", color="#5B5F97", marker="_", ms=20, label="always vertical")
ax.set_xticks(x); ax.set_xticklabels(S.step, fontsize=9)
ax.set_ylim(0, 1.1); ax.set_ylabel("Macro-F1 (leave-one-shape-out)")
ax.set_title("Selector macro-F1 per improvement step", fontsize=11)
ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
save(fig, "fig_f1_steps")
print(S.round(3).to_string(index=False))
