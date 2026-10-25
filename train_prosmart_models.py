#!/usr/bin/env python3
"""
train_prosmart_models.py - retrain ProSmart HPA's 22 PROPHET models on YOUR Smart HPA runs,
following the paper (JSS 2025, Sec. 5.1): one demand model and one capacity model per
microservice, trained on time series recorded while the service ran under Smart HPA,
with ds = elapsed test time and y = replicas.

    cd ~/shpa
    python train_prosmart_models.py --runs 2 3 4 --maxr 2 --target 20 --out models_2R20

Inputs (written by run_experiment.sh):
  results/smarthpa/Test/<MAXR> Replicas + <T>%/Run <n>/<svc>.xlsx   (Smart HPA Knowledge Base)
  results/cluster/smarthpa_<MAXR>R-<T>_run<n>_replicas_timeline.txt
  results/logs/info_smarthpa_<MAXR>R-<T>_run<n>.txt                  (start time)

Demand target   y = "Desired Replicas" from the Knowledge Base (Smart HPA's demand, before capping).
Capacity target y = replicas the service actually ran with while the resource manager was active
                    (any service above MAXR at that sample), otherwise MAXR.
                    The paper does not publish its capacity labels; this is our approximation
                    and must be stated in the report.

Outputs: <out>/<svc>_Demand_Model.pkl, <out>/<svc>_capacity_model.pkl,
         <out>/validation_demand.csv  (leave-one-run-out MAE: PROPHET vs persistence baseline)
"""
import argparse, logging, pickle, sys, warnings
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
from prophet import Prophet

SERVICES = ["frontend", "adservice", "cartservice", "checkoutservice", "currencyservice",
            "emailservice", "paymentservice", "productcatalogservice",
            "recommendationservice", "redis-cart", "shippingservice"]

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--runs", nargs="+", type=int, required=True)
ap.add_argument("--maxr", type=int, default=2)
ap.add_argument("--target", type=int, default=20)
ap.add_argument("--window", type=int, default=25, help="horizon for the persistence baseline (s)")
ap.add_argument("--out", default="models")
a = ap.parse_args()
R = Path(a.results); tag = f"{a.maxr}R-{a.target}"
out = Path(a.out); out.mkdir(exist_ok=True)

def new_model():
    return Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)

def ds(sec):
    return pd.to_datetime(np.asarray(sec, float), unit="s")

def load_demand(run, svc):
    p = R / "smarthpa" / "Test" / f"{a.maxr} Replicas + {a.target}%" / f"Run {run}" / f"{svc}.xlsx"
    df = pd.read_excel(p, header=0)
    df = df.iloc[:, [0, 3]].dropna()
    df.columns = ["t", "y"]
    return df.astype(float).sort_values("t").reset_index(drop=True)

def load_capacity(run):
    info = R / "logs" / f"info_smarthpa_{tag}_run{run}.txt"
    start = dict(l.strip().split("=", 1) for l in open(info) if "=" in l)["start_utc"]
    t0 = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    rows, t = [], None
    for line in open(R / "cluster" / f"smarthpa_{tag}_run{run}_replicas_timeline.txt"):
        line = line.strip()
        if line.startswith("#"):
            hh = datetime.strptime(line[1:].strip(), "%H:%M:%S")
            t = (hh - t0.replace(year=1900, month=1, day=1)).total_seconds() % 86400
        elif line and t is not None:
            name, reps = line.split()[:2]
            rows.append((t, name, float(reps) if reps not in ("<none>", "") else 0.0))
    df = pd.DataFrame(rows, columns=["t", "svc", "r"]).pivot_table(index="t", columns="svc", values="r")
    active = (df > a.maxr).any(axis=1)
    cap = df.where(active, other=float(a.maxr), axis=0)
    return cap

# ---------- load everything ----------
D = {}
for run in a.runs:
    for s in SERVICES:
        try:
            D[(run, s)] = load_demand(run, s)
        except Exception as e:
            sys.exit(f"Cannot read demand data for run {run}, {s}: {e}")
C = {run: load_capacity(run) for run in a.runs}
print(f"Loaded {len(a.runs)} runs: " + ", ".join(
    f"run {r} ({len(D[(r,'frontend')])} KB rows, {len(C[r])} timeline samples)" for r in a.runs))

# ---------- leave-one-run-out validation (demand) ----------
val = []
if len(a.runs) >= 2:
    for held in a.runs:
        train = [r for r in a.runs if r != held]
        for s in SERVICES:
            tr = pd.concat([D[(r, s)] for r in train])
            m = new_model().fit(pd.DataFrame({"ds": ds(tr.t), "y": tr.y}))
            te = D[(held, s)]
            yhat = np.ceil(m.predict(pd.DataFrame({"ds": ds(te.t)}))["yhat"].values)
            # persistence: demand known `window` seconds earlier
            idx = np.searchsorted(te.t.values, te.t.values - a.window, side="right") - 1
            ok = idx >= 0
            pers = te.y.values[idx[ok]]
            val.append({"held_out_run": held, "service": s,
                        "MAE_prophet": np.mean(np.abs(yhat[ok] - te.y.values[ok])),
                        "MAE_persistence": np.mean(np.abs(pers - te.y.values[ok])),
                        "n": int(ok.sum())})
    v = pd.DataFrame(val)
    v.to_csv(out / "validation_demand.csv", index=False)
    print("\nLeave-one-run-out demand MAE (replicas), mean over held-out runs:")
    print(v.groupby("service")[["MAE_prophet", "MAE_persistence"]].mean().round(2).to_string())
    print(f"ALL services: PROPHET {v.MAE_prophet.mean():.2f}  vs  persistence({a.window}s) "
          f"{v.MAE_persistence.mean():.2f}")

# ---------- final models on all runs ----------
for s in SERVICES:
    tr = pd.concat([D[(r, s)] for r in a.runs])
    m = new_model().fit(pd.DataFrame({"ds": ds(tr.t), "y": tr.y}))
    pickle.dump(m, open(out / f"{s}_Demand_Model.pkl", "wb"))
    cap = pd.concat([C[r][s].rename("y").reset_index() for r in a.runs if s in C[r]])
    if cap.empty:
        sys.exit(f"No timeline data for {s}")
    mc = new_model().fit(pd.DataFrame({"ds": ds(cap.t), "y": cap.y}))
    pickle.dump(mc, open(out / f"{s}_capacity_model.pkl", "wb"))
print(f"\nSaved 22 models to {out.resolve()}")
