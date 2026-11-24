#!/usr/bin/env python3
"""
eval_cpu_utilisation.py - reserved vs actually used CPU per method (post-hoc, descriptive).

    python analysis/eval_cpu_utilisation.py --root . --shape eval_heldout --runs 11 12 13 14 15 --tag _final5

Per run (60 s <= t < 900 s, redis-cart excluded), summed over all services and averaged over the
15-s samples:
  reserved_m   CPU requested by the ready app containers (replicas x CPU request), as in analyze_eval.py
  used_m       CPU actually used by those containers (metrics-server, collector column cpu_m)
  idle_m       reserved_m - used_m: CPU that was booked but not used
  utilisation_%  used_m / reserved_m
Both exclude the Linkerd proxy. Per method: mean +/- 95% CI over runs. Descriptive only (no tests).
Output: results/eval/cpu_utilisation_per_run<tag>.csv and cpu_utilisation_summary<tag>.csv
"""
import argparse, os
import numpy as np, pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--shape", required=True)
ap.add_argument("--runs", nargs="+", type=int, required=True)
ap.add_argument("--methods", nargs="+", default=["vertical", "hpa", "keda", "rules", "selector"])
ap.add_argument("--tag", default="")
a = ap.parse_args()
P = lambda *x: os.path.join(a.root, "results", *x)

rows = []
for m in a.methods:
    for r in a.runs:
        f = P("mech", f"{m}_{a.shape}_run{r}_metrics.csv")
        if not os.path.exists(f):
            print(f"missing: {m} run {r}"); continue
        d = pd.read_csv(f)
        d = d[(d.service != "redis-cart") & (d.t_sec >= 60) & (d.t_sec < 900)]
        t = d.assign(res=d.replicas * d.req_m_pod).groupby("t_sec")[["res", "cpu_m"]].sum()
        res, used = t.res.mean(), t.cpu_m.mean()
        rows.append({"method": m, "run": r, "reserved_m": res, "used_m": used,
                     "idle_m": res - used, "utilisation_%": 100 * used / res if res else np.nan})
R = pd.DataFrame(rows)

def mci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2: return x.mean(), float("nan")
    return x.mean(), stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)

out = []
for m in a.methods:
    g = R[R.method == m]
    if g.empty: continue
    row = {"method": m, "n": len(g)}
    for k in ["reserved_m", "used_m", "idle_m", "utilisation_%"]:
        mu, h = mci(g[k]); row[k] = f"{mu:.1f} ± {h:.1f}"
    out.append(row)
S = pd.DataFrame(out).set_index("method")
os.makedirs(P("eval"), exist_ok=True)
R.to_csv(P("eval", f"cpu_utilisation_per_run{a.tag}.csv"), index=False)
S.to_csv(P("eval", f"cpu_utilisation_summary{a.tag}.csv"))
pd.set_option("display.width", 200)
print(f"Reserved vs used CPU on {a.shape}, runs {a.runs} (millicores, all services; mean ± 95% CI over runs)\n")
print(S.to_string())
print(f"\nsaved results/eval/cpu_utilisation_per_run{a.tag}.csv, cpu_utilisation_summary{a.tag}.csv")
