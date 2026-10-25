#!/usr/bin/env python3
"""
collector.py - record per-service metrics every 15 s during a run (features and labels).

    python collector.py --out results/mech/<tag>_metrics.csv --minutes 16 --mechanism hpa

Columns (one row per service per sample):
  t_sec, service, mechanism, rps, p95_ms, success, cpu_m (server containers, sum),
  req_m_pod (mean CPU request of server container per ready pod), replicas (ready pods),
  cpu_util (cpu_m / total server CPU request)
Sources: Linkerd viz Prometheus (rps, latency, success), metrics-server (CPU), pod specs
(requests, which reflect in-place resizes).
"""
import argparse, csv, subprocess, time, urllib.parse, urllib.request, json, sys
from kubernetes import client, config

SERVICES = ["frontend", "adservice", "cartservice", "checkoutservice", "currencyservice",
            "emailservice", "paymentservice", "productcatalogservice",
            "recommendationservice", "shippingservice", "redis-cart"]
NS = "default"
Q = {
 "rps": 'sum by (deployment) (rate(request_total{namespace="default",direction="inbound"}[1m]))',
 "ok":  'sum by (deployment) (rate(response_total{namespace="default",direction="inbound",classification="success"}[1m]))',
 "p95": 'histogram_quantile(0.95, sum by (le, deployment) '
        '(rate(response_latency_ms_bucket{namespace="default",direction="inbound"}[1m])))',
}

def cpu_to_m(v):
    v = str(v)
    if v.endswith("n"): return float(v[:-1]) / 1e6
    if v.endswith("u"): return float(v[:-1]) / 1e3
    if v.endswith("m"): return float(v[:-1])
    return float(v) * 1000

def prom(q, port):
    url = f"http://localhost:{port}/api/v1/query?" + urllib.parse.urlencode({"query": q})
    with urllib.request.urlopen(url, timeout=5) as r:
        res = json.load(r)["data"]["result"]
    out = {}
    for x in res:
        try:
            val = float(x["value"][1])
            if val == val:                      # drop NaN
                out[x["metric"].get("deployment", "?")] = val
        except Exception:
            pass
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=16)
    ap.add_argument("--mechanism", default="?")
    ap.add_argument("--interval", type=float, default=15)
    ap.add_argument("--port", type=int, default=9091)
    a = ap.parse_args()

    config.load_kube_config()
    core, cust = client.CoreV1Api(), client.CustomObjectsApi()
    pf = subprocess.Popen(["kubectl", "-n", "linkerd-viz", "port-forward", "svc/prometheus", f"{a.port}:9090"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    f = open(a.out, "w", newline=""); w = csv.writer(f)
    w.writerow(["t_sec", "service", "mechanism", "rps", "p95_ms", "success", "cpu_m",
                "req_m_pod", "replicas", "cpu_util"])
    t0 = time.time(); n = 0
    try:
        while time.time() - t0 < a.minutes * 60:
            tick = time.time()
            try:
                rps, ok, p95 = prom(Q["rps"], a.port), prom(Q["ok"], a.port), prom(Q["p95"], a.port)
            except Exception as e:                       # restart port-forward if it dropped
                print("prometheus error:", e, file=sys.stderr)
                pf.kill(); pf = subprocess.Popen(["kubectl", "-n", "linkerd-viz", "port-forward",
                          "svc/prometheus", f"{a.port}:9090"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(3); continue
            usage = {}
            try:
                for p in cust.list_namespaced_custom_object("metrics.k8s.io", "v1beta1", NS, "pods")["items"]:
                    for c in p["containers"]:
                        if c["name"] in ("server", "redis"):
                            usage[p["metadata"]["name"]] = cpu_to_m(c["usage"]["cpu"])
            except Exception as e:
                print("metrics-server error:", e, file=sys.stderr)
            agg = {s: {"cpu": 0.0, "req": [], "n": 0} for s in SERVICES}
            for p in core.list_namespaced_pod(NS).items:
                app = (p.metadata.labels or {}).get("app")
                if app not in agg or p.status.phase != "Running" or p.metadata.deletion_timestamp:
                    continue
                if not all(cs.ready for cs in (p.status.container_statuses or [])):
                    continue
                cont = next((c for c in p.spec.containers if c.name in ("server", "redis")), None)
                req = cpu_to_m(cont.resources.requests.get("cpu", "0")) if cont and cont.resources.requests else 0
                agg[app]["n"] += 1; agg[app]["req"].append(req); agg[app]["cpu"] += usage.get(p.metadata.name, 0.0)
            t = round(tick - t0, 1)
            for s in SERVICES:
                g = agg[s]; reqsum = sum(g["req"])
                r = rps.get(s, 0.0)
                w.writerow([t, s, a.mechanism, round(r, 3), round(p95.get(s, float("nan")), 1),
                            round(ok.get(s, 0.0) / r, 4) if r > 0 else "",
                            round(g["cpu"], 1), round(reqsum / g["n"], 1) if g["n"] else 0, g["n"],
                            round(g["cpu"] / reqsum, 4) if reqsum else ""])
            f.flush(); n += 1
            time.sleep(max(0, a.interval - (time.time() - tick)))
    finally:
        pf.kill(); f.close()
        print(f"collector: {n} samples written to {a.out}")

if __name__ == "__main__":
    main()
