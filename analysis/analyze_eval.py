#!/usr/bin/env python3

"""
analyze_eval.py - compare the five methods on one evaluation workload.

    python analysis/analyze_eval.py --root . --shape eval_heldout --runs 11 12 13 14 15 --tag _final5
    python analysis/analyze_eval.py --root . --shape eval_final --runs 1 2 3 --tag _evalfinal

Per run (60 s <= t < 900 s, redis-cart excluded): Locust latency, failures and throughput; frontend
p95 and SLO violations (share of 15-s samples with frontend p95 > 500 ms, the main RQ3 metric);
reserved CPU (replicas x CPU request), replicas, replica changes, resizes and switches.
Locust percentiles cover the whole run, including the first 60 s.
Then mean and 95% CI per method, and Welch t-tests of the selector against each method with Holm
correction. Comparisons where neither group varies are not tested and are listed separately.
Output: results/eval/per_run<tag>.csv, summary<tag>.csv, tests<tag>.csv
"""

import argparse, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3])
ap.add_argument("--methods", nargs="+", default=["vertical", "hpa", "keda", "rules", "selector"])
ap.add_argument("--shape", default="eval_heldout")
ap.add_argument("--slo", type=float, default=500.0)
ap.add_argument("--tag", default="")
a = ap.parse_args()
P = lambda *x: os.path.join(a.root, "results", *x)

rows = []
for m in a.methods:
    for r in a.runs:
        tag = f"{m}_{a.shape}_run{r}"
        mf, lf = P("mech", f"{tag}_metrics.csv"), P("locust", f"{tag}_stats.csv")
        if not (os.path.exists(mf) and os.path.exists(lf)):
            print(f"missing: {tag}"); continue
        d = pd.read_csv(mf)
        d = d[(d.service != "redis-cart") & (d.t_sec >= 60) & (d.t_sec < 900)]
        fe = d[d.service == "frontend"]
        per_t = d.assign(alloc=d.replicas * d.req_m_pod).groupby("t_sec")
        dd = d.sort_values(["service", "t_sec"])
        rep_changes = int(dd.groupby("service").replicas.diff().fillna(0).ne(0).sum())
        loc = pd.read_csv(lf); agg = loc[loc.Name == "Aggregated"].iloc[0]
        va = P("mech", f"{tag}_vertical_actions.csv")
        resizes = max(0, sum(1 for _ in open(va)) - 1) if os.path.exists(va) else 0
        sl = P("logs", f"selector_{tag}.log")                 # + resizes done during consolidation
        if os.path.exists(sl):
            resizes += sum(int(x.split("resized ")[1].split("/")[0])
                           for x in open(sl) if "consolidated (" in x and "resized " in x)
        dec = P("mech", f"{tag}_decisions.csv")
        switches = int(pd.read_csv(dec).action.astype(str).str.startswith("switch").sum()) if os.path.exists(dec) else 0
        rows.append({
            "method": m, "run": r,
            "locust_p50_ms": agg["50%"], "locust_p95_ms": agg["95%"], "locust_p99_ms": agg["99%"],
            "locust_avg_ms": agg["Average Response Time"],
            "failures_%": 100 * agg["Failure Count"] / max(1, agg["Request Count"]),
            "throughput_rps": agg["Requests/s"],
            "fe_p95_ms": fe.p95_ms.mean(), "slo_viol_%": 100 * (fe.p95_ms > a.slo).mean(),
            "cpu_alloc_m": per_t.alloc.sum().mean(), "replicas": per_t.replicas.sum().mean(),
            "replica_changes": rep_changes, "resizes": resizes, "switches": switches,
        })
R = pd.DataFrame(rows)
os.makedirs(P("eval"), exist_ok=True)
R.to_csv(P("eval", f"per_run{a.tag}.csv"), index=False)

METRICS = ["slo_viol_%", "fe_p95_ms", "locust_p50_ms", "locust_p95_ms", "locust_p99_ms", "locust_avg_ms",
           "cpu_alloc_m", "replicas", "replica_changes", "resizes", "switches", "failures_%", "throughput_rps"]

def mci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2: return x.mean(), float("nan")
    return x.mean(), stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)

summary = []
for m in a.methods:
    g = R[R.method == m]
    row = {"method": m, "n": len(g)}
    for k in METRICS:
        mu, h = mci(g[k]); row[k] = f"{mu:.1f} ± {h:.1f}"
    summary.append(row)
S = pd.DataFrame(summary).set_index("method")
S.to_csv(P("eval", f"summary{a.tag}.csv"))
pd.set_option("display.width", 250)
print(f"\nMean ± 95% CI over runs {a.runs} (steady part 60-900 s, redis-cart excluded)\n")
print(S[["n", "slo_viol_%", "fe_p95_ms", "locust_p95_ms", "locust_p99_ms", "cpu_alloc_m", "replicas"]].to_string())
print()
print(S[["replica_changes", "resizes", "switches", "failures_%", "throughput_rps"]].to_string())

if "selector" in a.methods:
    tests = []
    for k in ["slo_viol_%", "fe_p95_ms", "locust_p95_ms", "cpu_alloc_m"]:
        for m in [x for x in a.methods if x != "selector"]:
            s, o = R[R.method == "selector"][k], R[R.method == m][k]
            if len(s) >= 2 and len(o) >= 2:
                if s.std(ddof=1) == 0 and o.std(ddof=1) == 0:
                    # no variation in either group: the t-test is undefined (0/0 or x/0), so it is
                    # not run; the comparison is reported descriptively instead
                    tests.append({"metric": k, "vs": m, "selector_mean": s.mean(), "other_mean": o.mean(),
                                  "p": np.nan, "note": f"not testable: no variation (selector {s.iloc[0]:.1f} "
                                                       f"in all {len(s)} runs, {m} {o.iloc[0]:.1f} in all {len(o)} runs)"})
                else:
                    p = stats.ttest_ind(s, o, equal_var=False).pvalue
                    tests.append({"metric": k, "vs": m, "selector_mean": s.mean(), "other_mean": o.mean(),
                                  "p": p, "note": ""})
    T = pd.DataFrame(tests)
    if len(T):
        T = T.sort_values("p", na_position="last").reset_index(drop=True)
        ok = T.p.notna(); k = int(ok.sum()); adj, run = [], 0.0
        for j, pv in enumerate(T.p[ok]):              # Holm step-down adjustment over testable comparisons only
            run = max(run, min(1.0, (k - j) * pv)); adj.append(run)
        T["p_holm"] = np.nan; T.loc[ok, "p_holm"] = adj
        T.to_csv(P("eval", f"tests{a.tag}.csv"), index=False)
        print("\nSelector vs each method (Welch t-test, Holm-corrected over all", k, "testable comparisons):")
        print(T[ok].drop(columns="note").round({"selector_mean": 1, "other_mean": 1, "p": 4, "p_holm": 4}).to_string(index=False))
        if (~ok).any():
            print("\nNot tested (no variation in either group; report descriptively):")
            for r in T[~ok].itertuples():
                print(f"  {r.metric:14s} vs {r.vs:9s} {r.note}")
print(f"\nsaved results/eval/per_run{a.tag}.csv, summary{a.tag}.csv, tests{a.tag}.csv")
