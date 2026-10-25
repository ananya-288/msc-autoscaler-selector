"""
hpa_logger.py - record the default Kubernetes HPA in the SAME format as
Smart HPA's Knowledge Base, so both autoscalers can be analysed with the same
metrics (Supply CPU, overutilization, under/overprovisioning - paper Table I).

It does NOT scale anything. It only watches. Run it while the HPAs are active.

For each service it writes  <out>/<service>.xlsx  with the same columns Smart HPA uses:
  A Test Time (sec) | B CPU Usage Percentage | C Current Replicas |
  D Desired Replicas | E Max. Replicas | F Scaling Action | G HPA Desired (capped)

Measured exactly like Smart HPA's Microservice Managers (same formula, same source):
  CPU %   = mean(kubectl top pods CPU of the service) / CPU request * 100
  Desired = ceil(current replicas * CPU % / target)     (Algorithm 1, line 1)
Column D is the UNCAPPED demand (can exceed max), as in Smart HPA. HPA's own
desiredReplicas is capped at max, so it is stored separately in column G.

Usage (PowerShell, from C:\\shpa):
  python hpa_logger.py --minutes 15 --out "results\\hpa\\Test\\5 Replicas + 50%\\Run 1"
"""
import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

from openpyxl import Workbook

SERVICES = ["frontend", "adservice", "cartservice", "currencyservice", "checkoutservice",
            "emailservice", "paymentservice", "shippingservice", "productcatalogservice",
            "recommendationservice", "redis-cart"]
HEADERS = ["Test Time (sec)", "CPU Usage Percentage", "Current Replicas",
           "Desired Replicas", "Max. Replicas", "Scaling Action", "HPA Desired (capped)"]


def kubectl(args):
    return subprocess.check_output(["kubectl"] + args, text=True, stderr=subprocess.STDOUT)


def cpu_to_milli(value):
    """'100m' -> 100, '1' -> 1000, '0.5' -> 500"""
    value = str(value).strip()
    return int(value[:-1]) if value.endswith("m") else int(float(value) * 1000)


def service_of(pod_name):
    """frontend-58b4dc4d4d-n46l6 -> frontend ; redis-cart-788fbddc7-52ffb -> redis-cart"""
    return pod_name.rsplit("-", 2)[0]


def read_deployments():
    data = json.loads(kubectl(["get", "deploy", "-o", "json"]))
    out = {}
    for d in data["items"]:
        name = d["metadata"]["name"]
        req = d["spec"]["template"]["spec"]["containers"][0].get("resources", {}) \
               .get("requests", {}).get("cpu")
        out[name] = {"available": d.get("status", {}).get("availableReplicas", 0) or 0,
                     "request": cpu_to_milli(req) if req else None}
    return out


def read_pod_cpu():
    """{service: [cpu_milli_of_each_pod]} from kubectl top pods"""
    out = {}
    for line in kubectl(["top", "pods", "--no-headers"]).splitlines():
        cols = line.split()
        if len(cols) >= 2:
            out.setdefault(service_of(cols[0]), []).append(cpu_to_milli(cols[1]))
    return out


def read_hpas():
    data = json.loads(kubectl(["get", "hpa", "-o", "json"]))
    return {h["spec"]["scaleTargetRef"]["name"]: {
                "max": h["spec"]["maxReplicas"],
                "desired": h.get("status", {}).get("desiredReplicas")}
            for h in data["items"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--interval", type=float, default=10, help="seconds between samples")
    ap.add_argument("--target", type=float, default=50, help="CPU target %% (same as HPA)")
    ap.add_argument("--out", required=True, help="folder for the .xlsx files")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    books = {}
    for s in SERVICES:
        wb = Workbook()
        wb.active.append(HEADERS)
        books[s] = wb

    hpas = read_hpas()
    missing = [s for s in SERVICES if s not in hpas]
    if missing:
        print("WARNING: no HPA found for:", ", ".join(missing))
        print("Did you run the kubectl autoscale command first?")

    start = time.time()
    end = start + args.minutes * 60
    n = 0
    while time.time() < end:
        tick = time.time()
        try:
            deps, cpu, hpas = read_deployments(), read_pod_cpu(), read_hpas()
        except subprocess.CalledProcessError as e:
            print("kubectl failed, skipping this sample:", e.output.strip()[:200])
            time.sleep(args.interval)
            continue

        t = round(tick - start, 1)
        for s in SERVICES:
            d, h = deps.get(s), hpas.get(s)
            if not d or not d["request"] or not cpu.get(s):
                continue  # pods starting / no metrics yet
            current = d["available"] or len(cpu[s])
            pct = statistics.mean(cpu[s]) / d["request"] * 100
            desired = math.ceil(current * pct / args.target)
            max_r = h["max"] if h else None
            action = ("scale up" if desired > current else
                      "scale down" if desired < current and desired >= 1 else "no scale")
            books[s].active.append([t, pct, current, desired, max_r, action,
                                    h["desired"] if h else None])

        for s, wb in books.items():  # save every sample, so a crash loses nothing
            try:
                wb.save(out / f"{s}.xlsx")
            except PermissionError:
                # Usually the file is open in Excel. Data stays in memory and is
                # written at the next sample once the file is closed.
                print(f"  WARNING: cannot write {s}.xlsx (is it open in Excel? close it)")
        n += 1
        fe = books["frontend"].active
        last = [c.value for c in fe[fe.max_row]] if fe.max_row > 1 else None
        print(f"[{t:6.0f}s] sample {n} saved"
              + (f" | frontend cpu {last[1]:.0f}% replicas {last[2]} demand {last[3]}" if last else ""))
        time.sleep(max(0, args.interval - (time.time() - tick)))

    print(f"Done. {n} samples written to {out.resolve()}")


if __name__ == "__main__":
    main()
