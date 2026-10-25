#!/usr/bin/env python3
"""
make_labels.py - decide, for every service and every 1-minute window, which mechanism
(HPA, KEDA or vertical) performed best on the SAME traffic.

    python make_labels.py --shape burst --run 1
    python make_labels.py --shape burst --run 1 --window 30      # 30-second windows

Score per mechanism (lower is better), following the weighted-cost idea of Rossi et al. (2019):
    score = 0.6 * SLO-violation share + 0.3 * relative CPU allocation + 0.1 * scaling actions
  - SLO violation (per 15-s sample): p95 latency > the service's SLO, or success < 99 %.
    SLO: frontend 500 ms (user-facing); other services 2 x their median p95 over all three runs.
  - relative CPU allocation: (replicas x CPU request) / the largest of the three mechanisms.
  - scaling actions: replica changes (HPA, KEDA) or in-place resizes (vertical), capped at 3.
The first minute (pod start-up after reset) and anything after 900 s are skipped.
Output: results/labels/<shape>_run<n>_labels.csv and a summary of how often each mechanism wins.
"""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--shape", required=True); ap.add_argument("--run", required=True)
ap.add_argument("--window", type=int, default=60)
ap.add_argument("--w", nargs=3, type=float, default=[0.6, 0.3, 0.1])
ap.add_argument("--dir", default="results/mech")
ap.add_argument("--slo-floor", type=float, default=0.0, help="minimum SLO for non-frontend services (ms)")
a = ap.parse_args()
MECH = ["hpa", "keda", "vertical"]

frames = []
for m in MECH:
    df = pd.read_csv(f"{a.dir}/{m}_{a.shape}_run{a.run}_metrics.csv")
    df["mechanism"] = m
    frames.append(df)
d = pd.concat(frames)
d = d[(d.service != "redis-cart") & (d.t_sec >= 60) & (d.t_sec < 900)].copy()
d["success"] = d["success"].fillna(1.0).clip(upper=1.0)
d["alloc"] = d.replicas * d.req_m_pod
d["win"] = (d.t_sec // a.window).astype(int)

# SLO per service
slo = (d.groupby("service").p95_ms.median() * 2).clip(lower=a.slo_floor)
slo["frontend"] = 500.0
d["viol"] = ((d.p95_ms > d.service.map(slo)) | (d.success < 0.99)).astype(float)

# scaling actions per (mechanism, service, window)
d = d.sort_values(["mechanism", "service", "t_sec"])
d["rep_change"] = d.groupby(["mechanism", "service"]).replicas.diff().fillna(0).ne(0).astype(int)
act = d.groupby(["mechanism", "service", "win"]).rep_change.sum()
vpath = f"{a.dir}/vertical_{a.shape}_run{a.run}_vertical_actions.csv"
if os.path.exists(vpath):
    va = pd.read_csv(vpath)
    if len(va):
        va["win"] = (va.t_sec // a.window).astype(int)
        # one resize per pod; count per service and window
        vc = va.groupby(["service", "win"]).size()
        for (s, w), n in vc.items():
            act.loc[("vertical", s, w)] = act.get(("vertical", s, w), 0) + n

g = d.groupby(["mechanism", "service", "win"]).agg(viol=("viol", "mean"), alloc=("alloc", "mean"),
        p95=("p95_ms", "median"), rps=("rps", "mean")).reset_index()
g["actions"] = [act.get((r.mechanism, r.service, r.win), 0) for r in g.itertuples()]
g["alloc_rel"] = g.alloc / g.groupby(["service", "win"]).alloc.transform("max")
w1, w2, w3 = a.w
g["score"] = w1 * g.viol + w2 * g.alloc_rel + w3 * np.minimum(g.actions, 3) / 3

wide = g.pivot_table(index=["service", "win"], columns="mechanism", values="score").dropna()
wide["label"] = wide[MECH].idxmin(axis=1)
wide["margin"] = wide[MECH].apply(lambda r: np.sort(r.values)[1] - np.sort(r.values)[0], axis=1)
os.makedirs("results/labels", exist_ok=True)
tag = f"_floor{int(a.slo_floor)}" if a.slo_floor else ""
out = f"results/labels/{a.shape}_run{a.run}_w{a.window}{tag}_labels.csv"
wide.reset_index().to_csv(out, index=False)
g.to_csv(out.replace("_labels.csv", "_window_details.csv"), index=False)

print("SLO per service (ms):", slo.round(1).to_dict())
print(f"{len(wide)} labelled service-windows -> {out}\n")
print("Overall winner share:")
print((wide.label.value_counts(normalize=True) * 100).round(1).to_string(), "\n")
print("Winner share per service (%):")
print((pd.crosstab(wide.index.get_level_values(0), wide.label, normalize="index") * 100).round(0).to_string(), "\n")
print("Mean score per mechanism (lower = better):")
print(g.groupby("mechanism")[["score", "viol", "alloc_rel", "actions"]].mean().round(3).to_string())
print(f"\nClose calls (margin < 0.05): {(wide.margin < 0.05).mean() * 100:.0f} % of windows")
