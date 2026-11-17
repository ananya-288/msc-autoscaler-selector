#!/usr/bin/env python3
"""
eval_warmup_check.py - EXPLORATORY (idea came from the results): were runs on freshly started nodes worse?

    python analysis/eval_warmup_check.py --root . --tags _final5:eval_heldout _evalfinal:eval_final

Run position after each node start comes from results/logs/batch_summary.txt; positions 1-2 = early.
Each run is compared with its own method's median; early vs later with a Mann-Whitney U test.
Output: results/eval/warmup_check.csv
"""

import argparse, os, re
import numpy as np, pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--tags", nargs="+", required=True, help="per_run tag:shape, e.g. _final5:eval_heldout")
ap.add_argument("--early", type=int, default=2)
ap.add_argument("--metrics", nargs="+", default=["slo_viol_%", "locust_p95_ms"])
a = ap.parse_args()
P = lambda *x: os.path.join(a.root, "results", *x)

pos, n = {}, 0
for line in open(P("logs", "batch_summary.txt"), encoding="utf-8", errors="replace"):
    if "node group scaled to 0" in line:
        n = 0
    m = re.search(r"\]\s+start\s+(\S+)", line)
    if m:
        n += 1
        pos[m.group(1)] = n                    # a re-started run keeps its latest position

frames = []
for spec in a.tags:
    tag, shape = spec.split(":")
    R = pd.read_csv(P("eval", f"per_run{tag}.csv")); R["shape"] = shape
    R["run_tag"] = R.method + "_" + shape + "_run" + R.run.astype(int).astype(str)
    R["position"] = R.run_tag.map(pos)
    for k in a.metrics:
        R[f"dev_{k}"] = R[k] - R.groupby("method")[k].transform("median")
    frames.append(R)
D = pd.concat(frames)
missing = D[D.position.isna()]
if len(missing):
    print("no start line found for:", ", ".join(missing.run_tag))
D = D.dropna(subset=["position"]); D["early"] = D.position <= a.early

pd.set_option("display.width", 200)
print(f"EXPLORATORY warm-up check: early = position 1-{a.early} after nodes started\n")
print(f"{D.early.sum()} early runs, {(~D.early).sum()} later runs\n")
print("Early runs:")
cols = ["shape", "method", "run", "position"] + a.metrics + [f"dev_{k}" for k in a.metrics]
print(D[D.early].sort_values(["shape", "position"])[cols].round(1).to_string(index=False))
rows = []
for k in a.metrics:
    e, l = D[D.early][f"dev_{k}"], D[~D.early][f"dev_{k}"]
    p = stats.mannwhitneyu(e, l, alternative="two-sided").pvalue if len(e) and len(l) else np.nan
    rows.append({"metric": k, "early_mean_dev": e.mean(), "early_median_dev": e.median(),
                 "later_mean_dev": l.mean(), "later_median_dev": l.median(), "mannwhitney_p": p})
T = pd.DataFrame(rows)
print("\nDeviation from the method's median (positive = worse than usual for that method):")
print(T.round(3).to_string(index=False))
D.to_csv(P("eval", "warmup_check.csv"), index=False)
print("\nsaved results/eval/warmup_check.csv")
