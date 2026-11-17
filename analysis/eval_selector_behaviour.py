#!/usr/bin/env python3
"""
eval_selector_behaviour.py - how the selector and the rule-based switcher behaved live.

    python analysis/eval_selector_behaviour.py --root . --shape eval_heldout --runs 11 12 13 14 15 --tag _final5

Reports: time each service spent on KEDA; switches per service and direction; traffic trend at each
switch; selector confidence; per-service SLO violations and reserved CPU for all methods; and a
timeline figure for one run. Decision times use the selector's own clock.
Output: results/eval/behaviour_*<tag>.csv, per_service<tag>.csv, results/figures/fig_selector_timeline*
"""

import argparse, os, json
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--shape", required=True)
ap.add_argument("--runs", nargs="+", type=int, required=True)
ap.add_argument("--switchers", nargs="+", default=["selector", "rules"])
ap.add_argument("--methods", nargs="+", default=["vertical", "hpa", "keda", "rules", "selector"])
ap.add_argument("--slo-floor", type=float, default=50.0)
ap.add_argument("--end", type=float, default=900.0)
ap.add_argument("--figure-run", type=int, default=None, help="run for the timeline figure (default: first)")
ap.add_argument("--tag", default="")
a = ap.parse_args()
P = lambda *x: os.path.join(a.root, "results", *x)
os.makedirs(P("eval"), exist_ok=True)
pd.set_option("display.width", 220)

thr = None
fs = P("selector", "final_selector.json")
if os.path.exists(fs):
    thr = json.load(open(fs)).get("threshold")

# ---------------- decisions ----------------
dec = []
for m in a.switchers:
    for r in a.runs:
        f = P("mech", f"{m}_{a.shape}_run{r}_decisions.csv")
        if not os.path.exists(f):
            print(f"missing decisions: {m} run {r}"); continue
        d = pd.read_csv(f); d["method"] = m; d["run"] = r
        d["after"] = np.where(d.action.astype(str).str.startswith("switch"), d.proposed, d.current_before)
        dec.append(d)
D = pd.concat(dec, ignore_index=True)

# 1. time-weighted share on KEDA per service
rows = []
for (m, r, s), g in D.sort_values("t_sec").groupby(["method", "run", "service"]):
    t = g.t_sec.to_numpy(); nxt = np.append(t[1:], max(a.end, t[-1]))
    dur = np.clip(nxt - t, 0, None)
    rows.append({"method": m, "run": r, "service": s,
                 "horizontal_%": 100 * dur[(g.after == "keda").to_numpy()].sum() / dur.sum() if dur.sum() else np.nan})
H = pd.DataFrame(rows)
Hs = H.pivot_table(index="service", columns="method", values="horizontal_%", aggfunc="mean").round(1)
Hs.loc["ALL (mean)"] = H.groupby("method")["horizontal_%"].mean().round(1)

# 2. switches per service and direction
D["kind"] = np.select([D.action.str.startswith("switch vertical->"), D.action.str.startswith("switch keda->"),
                       D.action.str.startswith("hold")], ["scale_out", "scale_back", "held"], "keep")
SW = (D[D.kind != "keep"].groupby(["method", "service", "kind"]).size().unstack(fill_value=0)
      .div(len(a.runs)).round(1))

# 3. timing of switches vs traffic trend
S = D[D.kind.isin(["scale_out", "scale_back"])].copy()
S["trend"] = np.where(S.slope > 0, "rising", np.where(S.slope < 0, "falling", "flat"))
TM = S.groupby(["method", "kind", "trend"]).size().unstack(fill_value=0)
TMp = (TM.div(TM.sum(axis=1), axis=0) * 100).round(0)

# 4. confidence of selector decisions
conf = None
Ds = D[(D.method == "selector") & D.prob.notna()]
if len(Ds):
    conf = {"decisions": len(Ds), "mean_P(horizontal)": round(Ds.prob.mean(), 3),
            "proposed_horizontal_%": round(100 * (Ds.proposed == "keda").mean(), 1)}
    if thr is not None:
        conf["threshold"] = thr
        conf["within_0.1_of_threshold_%"] = round(100 * ((Ds.prob - thr).abs() < 0.1).mean(), 1)

# decision spacing (kubectl actions delay the loop)
gap = D.groupby(["method", "run"]).t_sec.apply(lambda x: np.diff(np.sort(x.unique())).mean()).groupby("method").mean()

# ---------------- per-service results (all methods) ----------------
fr = []
for m in a.methods:
    for r in a.runs:
        f = P("mech", f"{m}_{a.shape}_run{r}_metrics.csv")
        if os.path.exists(f):
            x = pd.read_csv(f); x["method"] = m; x["run"] = r; fr.append(x)
PS = None
if fr:
    M = pd.concat(fr)
    M = M[(M.service != "redis-cart") & (M.t_sec >= 60) & (M.t_sec < 900)].copy()
    M["success"] = M["success"].fillna(1.0).clip(upper=1.0)
    slo = (M.groupby("service").p95_ms.median() * 2).clip(lower=a.slo_floor); slo["frontend"] = 500.0
    M["viol"] = ((M.p95_ms > M.service.map(slo)) | (M.success < 0.99)).astype(float)
    M["alloc"] = M.replicas * M.req_m_pod
    per_run = M.groupby(["method", "run", "service"]).agg(viol=("viol", "mean"), alloc=("alloc", "mean"),
                                                         p95=("p95_ms", "median")).reset_index()
    PS = per_run.groupby(["service", "method"])[["viol", "alloc", "p95"]].mean()
    PSv = (PS.viol.unstack() * 100).round(1)[a.methods]
    PSc = PS.alloc.unstack().round(0)[a.methods]
    per_run.to_csv(P("eval", f"per_service{a.tag}.csv"), index=False)

# ---------------- output ----------------
H.to_csv(P("eval", f"behaviour_horizontal_share{a.tag}.csv"), index=False)
D.to_csv(P("eval", f"behaviour_decisions{a.tag}.csv"), index=False)
print(f"Selector behaviour on {a.shape}, runs {a.runs}\n")
print("1. Time on KEDA (horizontal) per service, % of decision period (mean over runs):")
print(Hs.to_string())
print("\n2. Switches per run, by service and direction (held = wanted switch blocked by 120-s hold):")
print(SW.to_string())
print("\n3. Traffic trend at the moment of each switch (% of that switch type):")
print(TMp.to_string())
print("   counts:"); print(TM.to_string())
if conf:
    print("\n4. Selector confidence:", conf)
print("\nMean time between decisions (s):", gap.round(1).to_dict())
if PS is not None:
    print("\n5a. Per-service SLO violations (% of 15-s samples; frontend 500 ms, others 2 x pooled median p95, >= 50 ms):")
    print(PSv.to_string())
    print("\n5b. Per-service mean reserved CPU (millicores):")
    print(PSc.to_string())

# ---------------- figure ----------------
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fr_run = a.figure_run if a.figure_run is not None else a.runs[0]
    g = D[(D.method == "selector") & (D.run == fr_run)].sort_values("t_sec")
    if len(g):
        svcs = sorted(g.service.unique())
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 2.2]})
        fe = g[g.service == "frontend"]
        ax1.plot(fe.t_sec, fe.fe_rps, color="black", marker="o", ms=3)
        ax1.set_ylabel("frontend req/s\n(last minute)")
        for i, s in enumerate(svcs):
            q = g[g.service == s]; t = q.t_sec.to_numpy(); nxt = np.append(t[1:], max(a.end, t[-1]))
            for t0, t1, st in zip(t, nxt, q.after):
                ax2.barh(i, t1 - t0, left=t0, height=0.7, color="#028090" if st == "keda" else "#C9CBE0")
            for t0, k in zip(q.t_sec, q.action.astype(str)):
                if k.startswith("switch"):
                    ax2.plot(t0, i, marker="|", color="black", ms=12)
        ax2.set_yticks(range(len(svcs))); ax2.set_yticklabels(svcs, fontsize=8)
        ax2.set_xlabel("selector time (s)")
        from matplotlib.patches import Patch
        ax2.legend(handles=[Patch(color="#028090", label="scale out (KEDA)"), Patch(color="#C9CBE0", label="scale up (vertical)")],
                   fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
        fig.tight_layout()
        os.makedirs(P("figures"), exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(P("figures", f"fig_selector_timeline{a.tag}_run{fr_run}.{ext}"), dpi=300, metadata={"CreationDate": None} if ext == "pdf" else None)
        print(f"\nwrote results/figures/fig_selector_timeline{a.tag}_run{fr_run}.png/.pdf")
except ImportError:
    pass
print(f"saved results/eval/behaviour_*{a.tag}.csv and per_service{a.tag}.csv")
