#!/usr/bin/env python3
"""
live_selector.py - choose HPA, KEDA or vertical scaling for each microservice, every minute, live.

    python live_selector.py --mode ensemble --minutes 16 --log results/mech/<tag>_decisions.csv   (FINAL)
    python live_selector.py --mode knn   --minutes 16 --log results/mech/<tag>_decisions.csv
    python live_selector.py --mode rules --minutes 16 --log results/mech/<tag>_decisions.csv
    python live_selector.py --mode knn --dry-run          # offline test with synthetic traffic

MAPE-K loop:
  Monitor : every 15 s, read each service's request rate from Linkerd's Prometheus
            (same query as collector.py) and keep the last two minutes.
  Analyse : once a minute, build the SAME features used in training: rps, slope, burstiness,
            peak/mean, frontend rps + slope, forecast (rps + slope), service (one-hot).
  Plan    : ensemble (FINAL) - binary decision, scale up (vertical) or scale out (KEDA):
                    soft vote of MLP + ExtraTrees + kNN with the settings and threshold in
                    results/selector/final_selector.json (chosen by build_final_model.py),
                    trained at start-up on the binary + time-series training table; uses the
                    extra time-series features (previous minutes, 3-minute mean, acceleration).
            knn   - k-nearest-neighbours (k = 7) trained on training_dataset_floor50_avg12.csv;
                    switch away from vertical only if P(best) >= gate (0.8, chosen offline).
            rules - simple baseline: KEDA when the service's traffic is rising fast
                    (slope > 15 % of rps per minute), otherwise vertical.
            Both: vertical is the default; a service keeps its mechanism for >= 120 s after a switch.
  Execute : only ONE mechanism per service: delete the old object, create the new one
            (HPA: CPU 50 %, 1-5 replicas; KEDA: request-rate target from keda_thresholds.json;
             vertical: VPA in Off mode, applied by vertical_actuator.py through in-place resize).
            Switching to vertical CONSOLIDATES the service (fix after the first evaluation round):
            every running pod is resized in place to cover the service's current total CPU use
            (x1.15 margin, 50-1000m, limit = 2 x request) and the deployment is scaled to 1 replica,
            so 'vertical' always means one resized pod. Switching to HPA/KEDA leaves the current
            pod size as it is; new replicas start from the deployment's default request.
            After consolidation the vertical actuator keeps applying the native VPA target, which
            may shrink the pod again if VPA history was built while load was split over replicas
            (reported as a switching cost, not corrected).
"""
import argparse, csv, json, subprocess, sys, time, urllib.parse, urllib.request
from collections import deque
import numpy as np, pandas as pd

SERVICES = ["adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice", "frontend",
            "paymentservice", "productcatalogservice", "recommendationservice", "shippingservice"]
FEATS = ["rps", "slope", "burst", "peak_ratio", "fe_rps", "fe_slope", "forecast"]
LOGF = FEATS + ["rps_lag1", "rps_lag2", "slope_lag1", "rps_roll3", "accel"]
NS = "default"
PROM_IN_CLUSTER = "http://prometheus.linkerd-viz.svc.cluster.local:9090"
Q_RPS = 'sum by (deployment) (rate(request_total{namespace="default",direction="inbound"}[1m]))'

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["ensemble", "knn", "rules"], required=True)
ap.add_argument("--final", default="results/selector/final_selector.json")
ap.add_argument("--minutes", type=float, default=16)
ap.add_argument("--log", default="decisions.csv")
ap.add_argument("--train", default="results/selector/training_dataset_floor50_avg12.csv")
ap.add_argument("--gate", type=float, default=0.8)
ap.add_argument("--hold", type=float, default=120)
ap.add_argument("--rise", type=float, default=0.15, help="rules mode: slope/rps threshold for KEDA")
ap.add_argument("--port", type=int, default=9092)
ap.add_argument("--dry-run", action="store_true", help="no cluster: synthetic traffic, print decisions")
a = ap.parse_args()

# ---------------- model ----------------
model = None
ENS = None
if a.mode == "ensemble":
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    cfg = json.load(open(a.final))
    FEATS = cfg["features"]; COLS = cfg["columns"]
    tr = pd.read_csv(cfg["training_data"])
    Xtr = pd.concat([tr[FEATS].astype(float), pd.get_dummies(tr.service, prefix="svc").astype(float)], axis=1)[COLS]
    ytr = (tr.label == "horizontal").astype(int)
    g = cfg["models"]
    ENS = [make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=tuple(g["MLP"]["h"]), alpha=g["MLP"]["alpha"],
                                                         max_iter=2000, random_state=0)),
           ExtraTreesClassifier(n_estimators=300, max_depth=g["ExtraTrees"]["depth"], min_samples_leaf=g["ExtraTrees"]["leaf"],
                                max_features=g["ExtraTrees"]["feat"], class_weight="balanced", random_state=0, n_jobs=-1),
           make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=g["kNN"]["k"], weights=g["kNN"]["w"]))]
    ENS = [m.fit(Xtr, ytr) for m in ENS]
    THRESH = cfg["threshold"]
    print(f"ensemble trained on {len(tr)} rows from {cfg['training_data']}; threshold {THRESH}; features {FEATS}")
elif a.mode == "knn":
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    tr = pd.read_csv(a.train)
    Xtr = pd.concat([tr[FEATS].astype(float),
                     pd.get_dummies(tr.service, prefix="svc").astype(float)], axis=1)
    COLS = list(Xtr.columns)
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)).fit(Xtr, tr.label)
    print(f"kNN trained on {len(tr)} rows from {a.train}; classes {list(model.classes_)}; gate {a.gate}")

def plan(svc, f):
    """return (mechanism, probability-or-None)"""
    if a.mode == "ensemble":
        row = pd.DataFrame([{**{k: f[k] for k in FEATS}, **{c: 0.0 for c in COLS if c.startswith("svc_")}}])
        if f"svc_{svc}" in row:
            row[f"svc_{svc}"] = 1.0
        ph = float(np.mean([m.predict_proba(row[COLS])[0][1] for m in ENS]))
        return ("keda" if ph >= THRESH else "vertical"), ph
    if a.mode == "rules":
        rising = f["rps"] > 0 and f["slope"] / f["rps"] > a.rise
        return ("keda" if rising else "vertical"), None
    row = pd.DataFrame([{**{k: f[k] for k in FEATS}, **{c: 0.0 for c in COLS if c.startswith("svc_")}}])
    if f"svc_{svc}" in row:
        row[f"svc_{svc}"] = 1.0
    pr = model.predict_proba(row[COLS])[0]
    best = int(np.argmax(pr)); mech = model.classes_[best]
    if mech != "vertical" and pr[best] < a.gate:
        return "vertical", float(pr[best])
    return mech, float(pr[best])

# ---------------- execute ----------------
THR = json.load(open("keda_thresholds.json")) if not a.dry_run else {}

def kubectl(args, stdin=None):
    if a.dry_run:
        return True
    r = subprocess.run(["kubectl"] + args, input=stdin, text=True, capture_output=True)
    if r.returncode != 0:
        print(f"kubectl {' '.join(args[:3])} FAILED: {r.stderr.strip()}", file=sys.stderr, flush=True)
    return r.returncode == 0

def consolidate(svc):
    """switch to vertical = one pod: resize all pods to cover current total CPU use, then scale to 1"""
    if a.dry_run:
        return "dry-run"
    from kubernetes import client, config
    config.load_kube_config()
    core, cust = client.CoreV1Api(), client.CustomObjectsApi()
    pods = [p for p in core.list_namespaced_pod(NS, label_selector=f"app={svc}").items
            if p.status.phase == "Running" and not p.metadata.deletion_timestamp]
    if len(pods) <= 1:
        return "already 1 replica"
    usage = 0.0
    for m in cust.list_namespaced_custom_object("metrics.k8s.io", "v1beta1", NS, "pods")["items"]:
        if m["metadata"]["name"] in {p.metadata.name for p in pods}:
            for c in m["containers"]:
                if c["name"] == "server":
                    v = str(c["usage"]["cpu"])
                    usage += float(v[:-1]) / 1e6 if v.endswith("n") else float(v[:-1]) / 1e3 if v.endswith("u") \
                        else float(v[:-1]) if v.endswith("m") else float(v) * 1000
    new = int(min(1000, max(50, np.ceil(usage * 1.15))))
    body = json.dumps({"spec": {"containers": [{"name": "server", "resources": {
        "requests": {"cpu": f"{new}m"}, "limits": {"cpu": f"{2 * new}m"}}}]}})
    ok = sum(kubectl(["patch", "pod", p.metadata.name, "-n", NS, "--subresource", "resize", "--patch", body])
             for p in pods)
    time.sleep(5)
    scaled = kubectl(["scale", "deploy", svc, "--replicas=1"])
    return (f"{len(pods)} pods -> 1 pod at {new}m (total use {usage:.0f}m); "
            f"resized {ok}/{len(pods)}, scaled={'ok' if scaled else 'FAILED'}")

def remove_all(svc):
    kubectl(["delete", "scaledobject", f"{svc}-rps", "--ignore-not-found"])
    kubectl(["delete", "hpa", svc, "--ignore-not-found"])
    kubectl(["delete", "vpa", f"{svc}-vpa", "--ignore-not-found"])

def apply(svc, mech):
    remove_all(svc)
    if mech == "hpa":
        kubectl(["autoscale", "deploy", svc, "--cpu=50%", "--min=1", "--max=5"])
    elif mech == "keda":
        kubectl(["apply", "-f", "-"], stdin=f"""apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata: {{name: {svc}-rps}}
spec:
  scaleTargetRef: {{name: {svc}}}
  minReplicaCount: 1
  maxReplicaCount: 5
  pollingInterval: 15
  triggers:
  - type: prometheus
    metadata:
      serverAddress: {PROM_IN_CLUSTER}
      query: sum(rate(request_total{{namespace="default",deployment="{svc}",direction="inbound"}}[1m]))
      threshold: "{THR.get(svc, 10)}"
""")
    else:
        kubectl(["apply", "-f", "-"], stdin=f"""apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata: {{name: {svc}-vpa}}
spec:
  targetRef: {{apiVersion: apps/v1, kind: Deployment, name: {svc}}}
  updatePolicy: {{updateMode: "Off"}}
  resourcePolicy:
    containerPolicies:
    - {{containerName: server, controlledResources: [cpu], minAllowed: {{cpu: 50m}}, maxAllowed: {{cpu: 1000m}}}}
    - {{containerName: linkerd-proxy, mode: "Off"}}
""")

# ---------------- monitor ----------------
pf = None
def start_pf():
    global pf
    if a.dry_run:
        return
    if pf:
        pf.kill()
    pf = subprocess.Popen(["kubectl", "-n", "linkerd-viz", "port-forward", "svc/prometheus", f"{a.port}:9090"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)

def read_rps(t):
    if a.dry_run:                                   # synthetic: frontend ramps up, others follow
        base = 5 + 2.0 * (t / 60)
        return {s: base * (3 if s == "currencyservice" else 1) * (1 + 0.05 * np.random.randn()) for s in SERVICES}
    url = f"http://localhost:{a.port}/api/v1/query?" + urllib.parse.urlencode({"query": Q_RPS})
    with urllib.request.urlopen(url, timeout=5) as r:
        res = json.load(r)["data"]["result"]
    return {x["metric"].get("deployment"): float(x["value"][1]) for x in res}

# ---------------- loop ----------------
start_pf()
hist = {s: deque(maxlen=16) for s in SERVICES}         # 16 x 15 s = last four minutes
current = {s: "vertical" for s in SERVICES}
last_switch = {s: -1e9 for s in SERVICES}
for s in SERVICES:                                     # start state: vertical everywhere
    apply(s, "vertical")
f_log = open(a.log, "w", newline=""); w = csv.writer(f_log)
w.writerow(["t_sec", "service", *LOGF, "proposed", "prob", "current_before", "action"])
t0 = time.time(); tick = 0; switches = 0
interval = 0.01 if a.dry_run else 15
while time.time() - t0 < a.minutes * 60 if not a.dry_run else tick < a.minutes * 4:
    t = tick * 15.0 if a.dry_run else time.time() - t0
    try:
        rps = read_rps(t)
    except Exception as e:
        print("prometheus error:", e, file=sys.stderr); start_pf(); time.sleep(interval); tick += 1; continue
    for s in SERVICES:
        hist[s].append(rps.get(s, 0.0))
    tick += 1
    need = 12 if a.mode == "ensemble" else 8          # ensemble needs 3 minutes (lag features)
    if tick % 4 or tick < need:                        # decide once a minute
        time.sleep(interval); continue
    def stats(s):
        h = list(hist[s]); now, prev = h[-4:], h[-8:-4]
        prev2 = h[-12:-8] if len(h) >= 12 else prev
        m = float(np.mean(now)); m1 = float(np.mean(prev)); m2 = float(np.mean(prev2))
        return m, float(np.std(now, ddof=1)), float(np.max(now)), m - m1, m1, m2
    fe_m, _, _, fe_sl, _, _ = stats("frontend")
    for s in SERVICES:
        before = current[s]
        m, sd, mx, sl, m1, m2 = stats(s)
        f = {"rps": m, "slope": sl, "burst": sd / m if m else 0.0, "peak_ratio": mx / m if m else 1.0,
             "fe_rps": fe_m, "fe_slope": fe_sl, "forecast": max(0.0, m + sl),
             # time-series features, defined exactly as in train_selector.py --lags
             "rps_lag1": m1, "rps_lag2": m2, "slope_lag1": m1 - m2,
             "rps_roll3": (m + m1 + m2) / 3, "accel": sl - (m1 - m2)}
        mech, p = plan(s, f)
        action = "keep"
        if mech != current[s]:
            if t - last_switch[s] >= a.hold:
                apply(s, mech); current[s] = mech; last_switch[s] = t
                action = f"switch {before}->{mech}"; switches += 1
                if mech == "vertical":
                    info = consolidate(s)
                    print(f"t={t:.0f}s {s}: consolidated ({info})", flush=True)
            else:
                action = f"hold (wanted {mech})"
        w.writerow([round(t, 1), s, *[round(f[k], 4) for k in LOGF], mech,
                    "" if p is None else round(p, 3), before, action])
        if a.dry_run and action != "keep":
            print(f"t={t:5.0f}s {s:22s} {action}  p={p}")
    f_log.flush()
    print(f"[t={t:.0f}s] switches so far: {switches}", flush=True)
    time.sleep(interval)
f_log.close()
if pf:
    pf.kill()
print(f"live_selector ({a.mode}): {switches} switches; decisions in {a.log}")
