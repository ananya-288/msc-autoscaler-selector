#!/usr/bin/env python3
"""
vertical_actuator.py - vertical-only mechanism: native VPA recommender (updateMode Off)
+ in-place pod resize (no restart), following Bouasker et al. (NetSoft 2026).

Every 15 s, for each service with a VPA object, read the recommended CPU target of the
'server' container and resize every running pod in place if it differs by > 10 %.
Limit = 2 x request (same ratio as Online Boutique's defaults).

    python vertical_actuator.py --minutes 16 --log results/mech/<tag>_vertical_actions.csv
"""
import argparse, csv, time, sys, json, subprocess
from kubernetes import client, config

NS = "default"

def cpu_to_m(v):
    v = str(v)
    if v.endswith("n"): return float(v[:-1]) / 1e6
    if v.endswith("u"): return float(v[:-1]) / 1e3
    if v.endswith("m"): return float(v[:-1])
    return float(v) * 1000

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=16)
    ap.add_argument("--interval", type=float, default=15)
    ap.add_argument("--log", required=True)
    ap.add_argument("--min-m", type=int, default=50)
    ap.add_argument("--max-m", type=int, default=1000)
    a = ap.parse_args()
    config.load_kube_config()
    core, cust = client.CoreV1Api(), client.CustomObjectsApi()
    f = open(a.log, "w", newline=""); w = csv.writer(f)
    w.writerow(["t_sec", "service", "pod", "from_m", "to_m"])
    t0 = time.time(); actions = 0
    while time.time() - t0 < a.minutes * 60:
        tick = time.time()
        try:
            vpas = cust.list_namespaced_custom_object("autoscaling.k8s.io", "v1", NS, "verticalpodautoscalers")["items"]
        except Exception as e:
            print("vpa list error:", e, file=sys.stderr); time.sleep(a.interval); continue
        for v in vpas:
            svc = v["spec"]["targetRef"]["name"]
            recs = (v.get("status", {}).get("recommendation", {}) or {}).get("containerRecommendations", [])
            rec = next((r for r in recs if r["containerName"] == "server"), None)
            if not rec:
                continue
            target = int(min(a.max_m, max(a.min_m, cpu_to_m(rec["target"]["cpu"]))))
            for p in core.list_namespaced_pod(NS, label_selector=f"app={svc}").items:
                if p.status.phase != "Running" or p.metadata.deletion_timestamp:
                    continue
                cont = next((c for c in p.spec.containers if c.name == "server"), None)
                if not cont:
                    continue
                cur = cpu_to_m((cont.resources.requests or {}).get("cpu", "0"))
                if cur > 0 and abs(target - cur) / cur <= 0.10:
                    continue
                body = {"spec": {"containers": [{"name": "server", "resources": {
                        "requests": {"cpu": f"{target}m"}, "limits": {"cpu": f"{2 * target}m"}}}]}}
                try:
                    if hasattr(core, "patch_namespaced_pod_resize"):
                        core.patch_namespaced_pod_resize(p.metadata.name, NS, body)
                    else:
                        subprocess.run(["kubectl", "patch", "pod", p.metadata.name, "-n", NS, "--subresource",
                                        "resize", "--patch", json.dumps(body)], check=True, capture_output=True)
                    w.writerow([round(time.time() - t0, 1), svc, p.metadata.name, int(cur), target])
                    f.flush(); actions += 1
                except Exception as e:
                    print(f"resize failed {p.metadata.name}: {e}", file=sys.stderr)
        time.sleep(max(0, a.interval - (time.time() - tick)))
    f.close()
    print(f"vertical_actuator: {actions} resize actions")

if __name__ == "__main__":
    main()
