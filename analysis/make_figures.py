#!/usr/bin/env python3
"""
make_figures.py - report figures, made only from the saved result files (no hand-typed numbers).

    python analysis/make_figures.py --root .
    python analysis/make_figures.py --root . --rq3 _final5:eval_heldout _evalfinal:eval_final _:first      # also round 1

Writes PNG (300 dpi) and PDF (for LaTeX) to results/figures/. Figures have no titles: the caption
goes below the figure in LaTeX (state the workload and number of runs there).
  fig_tradeoff_both   RQ3: reserved CPU vs frontend SLO violations per method, one panel per
                      workload (large marker = mean, small dots = runs, lines = min-max range;
                      95% CIs are reported in the tables, not drawn)
  fig_tradeoff<tag>   the same, one workload per figure
  fig_latency_both    RQ3: user-facing p95 latency per method (Locust), one panel per workload
                      (dots = individual runs, thick line = mean, thin line = min-max range)
  fig_latency<tag>    the same, one workload per figure
  fig_rq1_forecast    RQ1: MAE relative to persistence per forecasting method, for one trace minute
                      and one selector minute (12 trace minutes); dashed line = persistence
  fig_f1_steps        RQ2: macro-F1 of the best model per step (3-class -> binary -> +time-series
                      -> +clean -> nested-tuned ensemble), with chance level and always-vertical
Inputs: results/eval/per_run<tag>.csv (analyze_eval.py), results/rq1/rq1_results.csv
(analysis/forecasting.py) and results/selector/rq2_*.csv (training/train_selector.py, training/tune_selector.py).
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--rq3", nargs="+", default=["_final5:eval_heldout", "_evalfinal:eval_final"],
                help="per_run tag:label pairs; use _ for the untagged first-round file")
a = ap.parse_args()
P = lambda *x: os.path.join(a.root, "results", *x)
OUT = P("figures"); os.makedirs(OUT, exist_ok=True)
# Publication style (as in IEEE papers): serif font like the report text, embedded fonts, light y-grid,
# and a different marker per method so the figures stay readable when printed in black and white.
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 10, "axes.labelsize": 10,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": "0.88", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42})
ORDER = ["vertical", "hpa", "keda", "rules", "selector"]
NAMES = {"vertical": "Vertical", "hpa": "HPA", "keda": "KEDA", "rules": "Rule-based", "selector": "ML selector"}
COL = {"vertical": "#5B5F97", "hpa": "#028090", "keda": "#F4A259", "rules": "#8A8A8A", "selector": "#C0392B"}
MARK = {"vertical": "v", "hpa": "s", "keda": "^", "rules": "D", "selector": "o"}

def mci(x):
    x = np.asarray(x, float)
    return x.mean(), (stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0)

def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig); print("wrote", os.path.join(OUT, name + ".png/.pdf"))

# ---------- RQ3 ----------
WL = []
for spec in a.rq3:
    tag, label = spec.split(":")
    tag = "" if tag == "_" else tag
    f = P("eval", f"per_run{tag}.csv")
    if os.path.exists(f):
        WL.append((tag, label, pd.read_csv(f)))
    else:
        print("missing", f)

def tradeoff(ax, R, label):
    # Small dots = individual runs; large marker = mean; lines = range (min-max) of the runs.
    # The range is shown instead of a t-based CI, which is very wide with 3-5 runs and can go below zero.
    n = R.groupby("method").size()
    for m in ORDER:
        g = R[R.method == m]
        if g.empty: continue
        x, y = g.cpu_alloc_m.values, g["slo_viol_%"].values
        cx, sy = x.mean(), y.mean()
        ax.scatter(x, y, s=18, color=COL[m], marker=MARK[m], alpha=0.45, edgecolor="none", zorder=2)
        ax.plot([x.min(), x.max()], [sy, sy], color=COL[m], lw=1.6, alpha=0.8)
        ax.plot([cx, cx], [y.min(), y.max()], color=COL[m], lw=1.6, alpha=0.8)
        ax.scatter([cx], [sy], s=90, color=COL[m], marker=MARK[m], edgecolor="black", zorder=3, label=NAMES[m])
        ax.annotate(NAMES[m], (cx, sy), textcoords="offset points", xytext=(8, 6), fontsize=9, color="black")
    ax.set_xlabel("Reserved CPU, all services (millicores)")
    ax.set_ylabel("Frontend SLO violations (%)\n(15-s samples with p95 > 500 ms)")
    runs = f"{int(n.min())}{'' if n.min() == n.max() else '-' + str(int(n.max()))} runs"
    ax.set_title(f"{label} ({runs})", fontsize=10, loc="left")   # panel label above the plot, never over data
    ax.set_ylim(bottom=-1)

def latency(ax, R, label):
    # Few runs per method (3-5), so every run is shown as a dot instead of a bar with a
    # t-based CI (which is very wide for n = 3 and can extend below zero).
    # Short horizontal line = mean; vertical line = range (min-max) of the runs.
    for i, m in enumerate(ORDER):
        g = R[R.method == m]
        if g.empty: continue
        y = g.locust_p95_ms.values; mu = y.mean()
        ax.vlines(i, y.min(), y.max(), color=COL[m], lw=2, alpha=0.6)
        ax.hlines(mu, i - 0.25, i + 0.25, color=COL[m], lw=3)
        ax.scatter([i] * len(y), y, color=COL[m], marker=MARK[m], edgecolor="black", s=40, zorder=3)
        ax.text(i + 0.3, mu, f"{mu:.0f}", ha="left", va="center", fontsize=9)
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([NAMES[m] for m in ORDER], fontsize=9)
    ax.set_xlim(-0.6, len(ORDER) - 0.3)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("User-facing p95 latency (ms, Locust)")
    ax.set_title(f"{label} ({len(R[R.method == ORDER[0]])} runs)", fontsize=10, loc="left")   # panel label above the plot

for tag, label, R in WL:
    fig, ax = plt.subplots(figsize=(6.2, 4.4)); tradeoff(ax, R, label); save(fig, f"fig_tradeoff{tag}")
    fig, ax = plt.subplots(figsize=(6.2, 4.0)); latency(ax, R, label); save(fig, f"fig_latency{tag}")
if len(WL) >= 2:
    fig, axs = plt.subplots(1, len(WL), figsize=(5.6 * len(WL), 4.4))
    for k, (ax, (tag, label, R)) in enumerate(zip(axs, WL)): tradeoff(ax, R, f"({chr(97 + k)}) {label}")
    save(fig, "fig_tradeoff_both")
    fig, axs = plt.subplots(1, len(WL), figsize=(5.6 * len(WL), 4.0), sharey=True)
    for k, (ax, (tag, label, R)) in enumerate(zip(axs, WL)): latency(ax, R, f"({chr(97 + k)}) {label}")
    save(fig, "fig_latency_both")

# ---------- RQ1 ----------
rf = P("rq1", "rq1_results.csv")
if os.path.exists(rf):
    Q = pd.read_csv(rf)
    meth = ["naive_trend", "holt", "arima", "gbr"]
    mname = {"naive_trend": "naive trend\n(selector)", "holt": "Holt", "arima": "ARIMA", "gbr": "gradient\nboosting"}
    sets = [("A_trace_minute", "1 trace minute", "#8A8A8A"), ("B_selector_minute", "1 selector minute\n(12 trace minutes)", "#C0392B")]
    fig, ax = plt.subplots(figsize=(6.4, 4.0)); x = np.arange(len(meth)); w = 0.38
    for j, (sname, slab, col) in enumerate(sets):
        q = Q[Q.setting == sname].set_index("method")
        if q.empty: continue
        vals = [q.loc[m, "rel_MAE_vs_persistence"] for m in meth]
        ax.bar(x + (j - 0.5) * w, vals, w, color=col, label=slab)
        for xi, v in zip(x + (j - 0.5) * w, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(1.0, color="black", ls="--", lw=1, label="persistence (= 1)")
    ax.set_xticks(x); ax.set_xticklabels([mname[m] for m in meth], fontsize=9)
    ax.set_ylabel("MAE relative to persistence\n(test week, lower is better)")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    save(fig, "fig_rq1_forecast")
else:
    print("missing", rf)

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
S = pd.DataFrame(rows)
if S.empty:
    raise SystemExit("no RQ2 result files found; RQ2 figure skipped")
S.to_csv(os.path.join(OUT, "fig_f1_steps_data.csv"), index=False)
fig, ax = plt.subplots(figsize=(7.2, 4.2))
x = np.arange(len(S))
ax.bar(x, S.macro_F1, color="#028090", width=0.6, label="best model")
for i, r in S.iterrows():
    ax.text(i, r.macro_F1 + 0.015, f"{r.macro_F1:.2f}\n{r.best_model.replace('_tuned', '')}", ha="center", fontsize=9)
ax.plot(x, S.chance, "k--", marker="_", ms=20, label="chance level (best dummy)")
ax.plot(x, S.always_vertical, ":", color="#5B5F97", marker="_", ms=20, label="always vertical")
ax.set_xticks(x); ax.set_xticklabels(S.step, fontsize=9)
ax.set_ylim(0, 1.1); ax.set_ylabel("Macro-F1 (leave-one-shape-out)")
ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
save(fig, "fig_f1_steps")
print(S.round(3).to_string(index=False))
