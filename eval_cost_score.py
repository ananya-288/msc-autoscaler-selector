#!/usr/bin/env python3
"""
eval_cost_score.py - one combined cost score per method for the live evaluation runs,
using the SAME formula as the training labels (make_labels.py), applied after the fact.

    python eval_cost_score.py --root . --shape eval_heldout --runs 11 12 13 14 15 --tag _final5
    python eval_cost_score.py --root . --shape eval_final   --runs 1 2 3          --tag _evalfinal

Score per service and 1-minute window (lower is better), as in make_labels.py:
    score = 0.6 * SLO-violation share + 0.3 * relative CPU allocation + 0.1 * min(actions, 3) / 3
  - SLO violation (per 15-s sample): p95 > the service's SLO, or success < 99 %.
    SLO: frontend 500 ms; other services 2 x their median p95, at least 50 ms (the floor used
    for the final labels). Here the median is taken over ALL methods and runs of the shape, so
    every method is judged against the same thresholds.
  - relative CPU allocation: (replicas x CPU request) / the largest of the compared methods in the
    same round (run number), service and window. With five methods this is usually HPA.
  - actions: replica changes plus in-place resizes (from <tag>_vertical_actions.csv).
    Resizes done during consolidation (selector log) are not in that file and are not counted.
The first minute and anything after 900 s are skipped; redis-cart is excluded.
Per run: mean score over all service-windows. Per method: mean +/- 95% CI over runs, and the
share of service-windows in which each method had the lowest score.
Differences from the labels: five methods instead of three, and pooled SLO thresholds.
"""
import argparse, os
import numpy as np, pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--shape", required=True)
ap.add_argument("--runs", nargs="+", type=int, required=True)
ap.add_argument("--methods", nargs="+", default=["vertical", "hpa", "keda", "rules", "selector"])
ap.add_argument("--window", type=int, default=60)
ap.add_argument("--w", nargs=3, type=float, default=[0.6, 0.3, 0.1])
ap.add_argument("--slo-floor", type=float, default=50.0)
ap.add_argument("--tag", default="")
a = ap.parse_args()
P = lambda *x: os.path.join(a.root, "results", *x)

frames = []
for m in a.methods:
    for r in a.runs:
        tag = f"{m}_{a.shape}_run{r}"
        f = P("mech", f"{tag}_metrics.csv")
        if not os.path.exists(f):
            print(f"missing: {tag}"); continue
        d = pd.read_csv(f); d["method"] = m; d["run"] = r
        frames.append(d)
d = pd.concat(frames)
d = d[(d.service != "redis-cart") & (d.t_sec >= 60) & (d.t_sec < 900)].copy()
d["success"] = d["success"].fillna(1.0).clip(upper=1.0)
d["alloc"] = d.replicas * d.req_m_pod
d["win"] = (d.t_sec // a.window).astype(int)

slo = (d.groupby("service").p95_ms.median() * 2).clip(lower=a.slo_floor)
slo["frontend"] = 500.0
d["viol"] = ((d.p95_ms > d.service.map(slo)) | (d.success < 0.99)).astype(float)

# actions: replica changes per (method, run, service, window) + resizes from the actuator log
d = d.sort_values(["method", "run", "service", "t_sec"])
d["rep_change"] = d.groupby(["method", "run", "service"]).replicas.diff().fillna(0).ne(0).astype(int)
act = d.groupby(["method", "run", "service", "win"]).rep_change.sum().to_dict()
for m in a.methods:
    for r in a.runs:
        vpath = P("mech", f"{m}_{a.shape}_run{r}_vertical_actions.csv")
        if os.path.exists(vpath):
            va = pd.read_csv(vpath)
            if len(va):
                va = va[(va.t_sec >= 60) & (va.t_sec < 900)]
                for (s, w), n in va.groupby([va.service, (va.t_sec // a.window).astype(int)]).size().items():
                    act[(m, r, s, w)] = act.get((m, r, s, w), 0) + n

g = d.groupby(["method", "run", "service", "win"]).agg(viol=("viol", "mean"), alloc=("alloc", "mean")).reset_index()
g["actions"] = [act.get((x.method, x.run, x.service, x.win), 0) for x in g.itertuples()]
g["alloc_rel"] = g.alloc / g.groupby(["run", "service", "win"]).alloc.transform("max")
w1, w2, w3 = a.w
g["act_term"] = np.minimum(g.actions, 3) / 3
g["score"] = w1 * g.viol + w2 * g.alloc_rel + w3 * g.act_term

# per run
R = g.groupby(["method", "run"])[["score", "viol", "alloc_rel", "act_term"]].mean().reset_index()

# share of service-windows each method wins (only windows where all methods are present)
wide = g.pivot_table(index=["run", "service", "win"], columns="method", values="score").dropna()
wins = (wide.idxmin(axis=1).value_counts(normalize=True) * 100).reindex(a.methods).fillna(0)

def mci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2: return x.mean(), float("nan")
    return x.mean(), stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)

rows = []
for m in a.methods:
    q = R[R.method == m]; row = {"method": m, "n": len(q)}
    for k in ["score", "viol", "alloc_rel", "act_term"]:
        mu, h = mci(q[k]); row[k] = f"{mu:.3f} ± {h:.3f}"
    row["wins_%"] = round(wins[m], 1)
    rows.append(row)
S = pd.DataFrame(rows).set_index("method")
S["rank"] = R.groupby("method").score.mean().reindex(S.index).rank().astype(int)

os.makedirs(P("eval"), exist_ok=True)
R.to_csv(P("eval", f"cost_score_per_run{a.tag}.csv"), index=False)
S.to_csv(P("eval", f"cost_score_summary{a.tag}.csv"))
pd.set_option("display.width", 200)
print("SLO per service (ms):", slo.round(1).to_dict())
print(f"\nCost score on {a.shape}, runs {a.runs} (lower = better; mean ± 95% CI over runs)")
print(f"score = {w1} viol + {w2} alloc_rel + {w3} act_term; wins_% = share of service-windows with the lowest score\n")
print(S.sort_values("rank").to_string())
print(f"\nsaved results/eval/cost_score_per_run{a.tag}.csv, cost_score_summary{a.tag}.csv")
