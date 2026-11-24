#!/usr/bin/env python3
"""
baseline_summary.py - base-paper reproduction (RO1): Kubernetes HPA vs Smart HPA vs ProSmart HPA.

    python analysis/baseline_summary.py --root .

Reads the Locust "Aggregated" row of each valid run (scenario 2 replicas, 20 % CPU target;
runs 2-4, the earlier runs and the test runs are excluded) and reports mean and 95% CI per method:
mean and p95 latency, requests served and failures.
Output: results/eval/baseline_summary.csv and baseline_per_run.csv
"""
import argparse, os
import numpy as np, pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--runs", nargs="+", default=["2", "3", "4"])
ap.add_argument("--methods", nargs="+", default=["hpa", "smarthpa", "prosmart"])
a = ap.parse_args()
P = lambda *x: os.path.join(a.root, "results", *x)

rows = []
for m in a.methods:
    for r in a.runs:
        f = P("locust", f"{m}_2R-20_run{r}_stats.csv")
        if not os.path.exists(f):
            print(f"missing: {f}"); continue
        s = pd.read_csv(f)
        g = s[s["Name"] == "Aggregated"].iloc[0]
        rows.append({"method": m, "run": int(r), "mean_ms": g["Average Response Time"], "p95_ms": g["95%"],
                     "requests": g["Request Count"], "failures": g["Failure Count"]})
D = pd.DataFrame(rows)

def ci(x):
    x = np.asarray(x, float)
    return 0.0 if len(x) < 2 or x.std(ddof=1) == 0 else stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))

cols = ["mean_ms", "p95_ms", "requests", "failures"]
S = D.groupby("method", sort=False)[cols].agg(["mean", ci])
S.columns = [f"{c}_{'ci95' if s == 'ci' else s}" for c, s in S.columns]
S.insert(0, "n", D.groupby("method", sort=False).size())
hpa = S.loc["hpa", "mean_ms_mean"] if "hpa" in S.index else np.nan
S["mean_ms_x_hpa"] = S["mean_ms_mean"] / hpa

print(f"\nBase-paper reproduction (2 replicas, 20 % CPU), runs {a.runs}: mean ± 95% CI\n")
show = pd.DataFrame({m: {"n": int(S.loc[m, "n"]),
                         "mean_ms": f"{S.loc[m,'mean_ms_mean']:.0f} ± {S.loc[m,'mean_ms_ci95']:.0f}",
                         "p95_ms": f"{S.loc[m,'p95_ms_mean']:.0f} ± {S.loc[m,'p95_ms_ci95']:.0f}",
                         "requests": f"{S.loc[m,'requests_mean']:.0f} ± {S.loc[m,'requests_ci95']:.0f}",
                         "failures": f"{S.loc[m,'failures_mean']:.1f}",
                         "x_hpa_latency": f"{S.loc[m,'mean_ms_x_hpa']:.1f}"} for m in S.index}).T
print(show.to_string())
os.makedirs(P("eval"), exist_ok=True)
D.to_csv(P("eval", "baseline_per_run.csv"), index=False)
S.round(3).to_csv(P("eval", "baseline_summary.csv"))
print("\nsaved results/eval/baseline_per_run.csv, baseline_summary.csv")
