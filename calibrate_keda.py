#!/usr/bin/env python3
"""
calibrate_keda.py - derive each service's KEDA target (requests/s per pod) from an HPA run,
so that KEDA aims at the SAME load per pod as HPA (50 % CPU) but reacts to request rate.

    python calibrate_keda.py results/mech/hpa_burst_run1_metrics.csv --target 0.5 --out keda_thresholds.json

For every sample with meaningful load: capacity = (rps / replicas) * (target / cpu_util),
i.e. the request rate one pod can take at 50 % CPU. The median over the run is used.
"""
import argparse, json
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="+")
ap.add_argument("--target", type=float, default=0.5)
ap.add_argument("--out", default="keda_thresholds.json")
a = ap.parse_args()
df = pd.concat([pd.read_csv(c) for c in a.csv])
df = df[(df.service != "redis-cart") & (df.replicas > 0) & (df.rps > 0.5) & (df.cpu_util > 0.10)]
res = {}
for s, g in df.groupby("service"):
    cap = (g.rps / g.replicas) * (a.target / g.cpu_util)
    res[s] = round(max(0.5, float(cap.median())), 2)
    print(f"{s:24s} {res[s]:8.2f} req/s per pod   (from {len(g)} samples)")
json.dump(res, open(a.out, "w"), indent=1)
print(f"saved {a.out}")
