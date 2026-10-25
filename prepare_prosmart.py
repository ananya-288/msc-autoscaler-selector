#!/usr/bin/env python3
"""
prepare_prosmart.py - build a runnable copy of the authors' ProSmart HPA code.

    python prepare_prosmart.py ~/ProSmart_HPA --models models_2R20 --minutes 15 \
        --maxr 2 --target 20 --window 25 --force

The authors' algorithm is NOT changed. The released code does not run as published,
so the following repairs are applied and must be declared in the report:

  P1  Each Microservice Manager calls Analyse() with 6 arguments, but Analyse() is
      defined with 7 (Predicted_Demand missing) -> TypeError on the first loop.
      Fix: pass current_replicas as the missing argument (Analyse overwrites it anyway).
  P2  Regressors are built with pd.concat() on plain numbers -> TypeError.
      Fix: build the same three regressors as a one-row DataFrame. (The published
      .pkl models were trained without regressors, so Prophet ignores them.)
  P3  math.ceil() on a pandas Series -> made explicit (.iloc[0]); predictions below
      min_replica are clamped to min_replica (otherwise kubectl scales a service to 0).
  P4  Values read back from the ARM_*.txt files are written as floats ("7.0") but read
      with int() -> ValueError. Fix: int(float(...)).
  P5  Application_Resource_Manager imports all 11 managers while they import it
      (circular import), so 10 of 11 managers never see Application_Resource_Manager
      -> NameError when the ARM is needed. Fix: remove the unused imports.
  P6  ARM spells "productcatalogueservice" but the manager writes
      "productcatalogservice.txt" -> FileNotFoundError. Fix: use the real name.
  P7  ARM stores strings ("scale up") in numeric numpy arrays -> ValueError.
      Fix: dtype=object for the two scaling-action arrays.
  P8  ARM balancing loops can run past the 11 services / never terminate when the
      budget cannot be balanced. Fix: index and iteration guards.
  P9  Main.py iterates over the Pool object ("for _ in pool") -> TypeError.
      Fix: remove; Pool.map() already waits for all managers.
  P11 Managers and the ARM run as parallel processes and exchange data through small
      .txt files. A file being rewritten can be read while still empty -> IndexError,
      which stops the whole controller. Fix: write these files atomically
      (write a temporary file, then os.replace), so readers always see a complete file.
  P12 ARM Plan sorts services with np.argsort (ascending), i.e. LEAST over-provisioned first,
      although the paper (Sec. 4.2) and the code comments say "most to least". Under-provisioned
      services are then treated as donors with negative residual resource, which increases
      capacity instead of removing it, so the application-wide capacity is not conserved
      (run 2 used ~37 pods instead of the 22-pod budget). Fix, following the paper's text:
      sort descending, take only donors that really have residual resource, and if the
      budget still cannot be balanced, trim the gains of the least under-provisioned
      services so total capacity never exceeds the original budget.
  P10 Status files (ARM_*.txt, <service>.txt) and fresh Knowledge Base workbooks are
      created before the run, because the code reads them before writing them.
Scenario settings (target_cpu, max_replica), test length and prediction window are
the values the authors tell users to edit by hand.
"""
import argparse, re, shutil, sys
from pathlib import Path

SERVICES = ["frontend", "adservice", "cartservice", "checkoutservice", "currencyservice",
            "emailservice", "paymentservice", "productcatalogservice",
            "recommendationservice", "redis-cart", "shippingservice"]
MANAGER_FILE = {s: ("rediscart.py" if s == "redis-cart" else f"{s}.py") for s in SERVICES}
CPU_REQUEST = {s: 100 for s in SERVICES} | {"cartservice": 200, "adservice": 200, "redis-cart": 70}

def sub(text, pattern, repl, expect, label, fname, flags=0):
    new, n = re.subn(pattern, repl, text, flags=flags)
    if n != expect:
        sys.exit(f"PATCH {label} failed in {fname}: expected {expect} match(es), found {n}")
    return new

HELPER = (
    "import os as _os, contextlib as _ctx\n"
    "@_ctx.contextmanager\n"
    "def _atomic_open(path):\n"
    "    tmp = f'{path}.{_os.getpid()}.tmp'\n"
    "    with open(tmp, 'w') as f:\n"
    "        yield f\n"
    "    _os.replace(tmp, path)\n\n")

def atomic(t, fname):
    new, n = re.subn(r"with open\((f?'[^']*\.txt'), 'w'\) as file:", r"with _atomic_open(\1) as file:", t)
    if n < 1:
        sys.exit(f"PATCH P11 failed in {fname}: no status-file writes found")
    return HELPER + new

ap = argparse.ArgumentParser()
ap.add_argument("repo", help="path to cloned ProSmart_HPA repository")
ap.add_argument("--out", default="prosmart_run")
ap.add_argument("--models", default=None,
                help="folder with <svc>_Demand_Model.pkl and <svc>_capacity_model.pkl "
                     "(default: the authors' published models)")
ap.add_argument("--minutes", type=float, default=15.0)
ap.add_argument("--maxr", type=int, default=2)
ap.add_argument("--target", type=int, default=20)
ap.add_argument("--window", type=int, default=25, help="prediction window in seconds")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

code = Path(a.repo).expanduser() / "ProSmart HPA Codebase"
mm = code / "Microservice Managers"
if not mm.is_dir():
    sys.exit(f"Cannot find {mm}")
out = Path(a.out)
if out.exists():
    if not a.force:
        sys.exit(f"{out} exists; use --force")
    shutil.rmtree(out)
out.mkdir()
secs = int(round(a.minutes * 60))

# ---------- Microservice Managers ----------
for s in SERVICES:
    f = MANAGER_FILE[s]
    t = (mm / f).read_text(encoding="utf-8").replace("\r\n", "\n")
    t = sub(t, r"Analyse \(Test_Time, current_replicas, current_cpu, target_cpu, cpu_request, min_replica\)",
            "Analyse (Test_Time, current_replicas, current_cpu, target_cpu, cpu_request, min_replica, current_replicas)",
            1, "P1", f)
    t = sub(t, r"regressors = pd\.concat\(\[current_replicas, cpu_percentage, target_cpu\], axis=1\)",
            "regressors = pd.DataFrame({'current_replicas': [current_replicas], 'cpu_percentage': [cpu_percentage], "
            "'target_cpu': [target_cpu]}, index=futures.index)", 1, "P2", f)
    t = sub(t, r"Predicted_Demand = math\.ceil\(forecast \['yhat'\]\)",
            "Predicted_Demand = max(min_replica, math.ceil(float(forecast['yhat'].iloc[0])))", 1, "P3", f)
    t = sub(t, r"int\(lines\[2\]\.strip\(\)\)", "int(float(lines[2].strip()))", 1, "P4a", f)
    t = sub(t, r"int\(lines\[3\]\.strip\(\)\)", "int(float(lines[3].strip()))", 1, "P4b", f)
    t = sub(t, r"\(Predicted_Time \* 5\) \+ 25", f"(Predicted_Time * 5) + {a.window}", 1, "window", f)
    t = sub(t, r"if Predicted_Time > 900:", f"if Predicted_Time > {secs}:", 1, "horizon-if", f)
    t = sub(t, r"Predicted_Time = 900\b", f"Predicted_Time = {secs}", 1, "horizon-set", f)
    t = sub(t, r"target_cpu = \d+", f"target_cpu = {a.target}", 1, "target", f)
    t = sub(t, r"max_replica = \d+\n", f"max_replica = {a.maxr}\n", 1, "maxr", f)
    t = atomic(t, f)
    (out / f).write_text(t, encoding="utf-8")

# ---------- Application Resource Manager ----------
f = "Application_Resource_Manager.py"
t = (code / f).read_text(encoding="utf-8").replace("\r\n", "\n")
t = sub(t, r"^from (frontend|adservice|cartservice|checkoutservice|currencyservice|emailservice|"
           r"paymentservice|productcatalogservice|recommendationservice|shippingservice|rediscart) import \*\n",
        "", 11, "P5", f, flags=re.M)
t = sub(t, r"productcatalogueservice", "productcatalogservice", 2, "P6", f)
t = sub(t, r"regressors = pd\.concat\(\[max_replica, current_replicas, Predicted_Demand\], axis=1\)",
        "regressors = pd.DataFrame({'max_replica': [max_replica], 'current_replicas': [current_replicas], "
        "'Predicted_Demand': [Predicted_Demand]}, index=futures.index)", 1, "P2-ARM", f)
t = sub(t, r"Predicted_Capacity  = math\.ceil\(forecast \['yhat'\]\)",
        "Predicted_Capacity  = max(1, math.ceil(float(forecast['yhat'].iloc[0])))", 1, "P3-ARM", f)
t = sub(t, r"scaling_action_array = np\.zeros\(11\)", "scaling_action_array = np.zeros(11, dtype=object)", 1, "P7a", f)
t = sub(t, r"resource_wise_SD_array = np\.zeros\(11\)", "resource_wise_SD_array = np.zeros(11, dtype=object)", 1, "P7b", f)
t = sub(t, r"while \(added_resource != removed_resource\):",
        "_guard = 0\n    while (added_resource != removed_resource) and (_guard < 50):\n        _guard += 1",
        1, "P8a", f)
t = sub(t, r"while \(resource_needs_to_be_removed != 0\):",
        "while (resource_needs_to_be_removed > 0) and (j < 11):", 1, "P8b", f)
t = sub(t, r"while \(resource_needs_to_be_added != 0\):",
        "while (resource_needs_to_be_added > 0) and (j < 11):", 1, "P8c", f)
t = sub(t, r"(over = np\.argsort\(\(Predicted_Capacity_array - Predicted_Demand_array\) \* cpu_request_array\))",
        r"\1[::-1]", 1, "P12a", f)
t = sub(t, r"(under = np\.argsort\(\(Predicted_Demand_array - Predicted_Capacity_array\) \* cpu_request_array\))",
        r"\1[::-1]", 1, "P12b", f)
t = sub(t, r"while \(resource_needs_to_be_removed > 0\) and \(j < 11\):",
        "while (resource_needs_to_be_removed > 0) and (j < 11) and "
        "(Predicted_Capacity_array[over[j]] - Predicted_Demand_array[over[j]] > 0):", 1, "P12c", f)
t = sub(t, r"while \(resource_needs_to_be_added > 0\) and \(j < 11\):",
        "while (resource_needs_to_be_added > 0) and (j < 11) and "
        "(Predicted_Demand_array[under[j]] - Predicted_Capacity_array[under[j]] > 0):", 1, "P12d", f)
t = sub(t, r"\n    return Predicted_Capacity_array\n",
        "\n    while added_resource > removed_resource:                      # P12: never exceed budget\n"
        "        gainers = [i for i in range(11) if Predicted_Capacity_array[i] > max_replica_array[i]]\n"
        "        if not gainers: break\n"
        "        k = min(gainers, key=lambda i: Predicted_Demand_array[i] - Predicted_Capacity_array[i])\n"
        "        Predicted_Capacity_array[k] -= 1\n"
        "        added_resource -= cpu_request_array[k]\n"
        "    return Predicted_Capacity_array\n", 1, "P12e", f)
t = atomic(t, f)
(out / f).write_text(t, encoding="utf-8")

# ---------- Main ----------
t = (code / "Main.py").read_text(encoding="utf-8").replace("\r\n", "\n")
t = sub(t, r"desired_time = 900", f"desired_time = {secs}", 1, "timer", "Main.py")
t = sub(t, r"\n[ \t]*for _ in pool:[^\n]*\n[ \t]*_\.join\(\)", "\n", 1, "P9", "Main.py")
(out / "run_prosmart_hpa.py").write_text(t, encoding="utf-8")

# ---------- Models ----------
src_d = Path(a.models) if a.models else mm / "Trained PROPHET models - Resource Demand Prediction"
src_c = Path(a.models) if a.models else code / "Trained PROPHET models - Resource Capacity Prediction"
for s in SERVICES:
    for src, name in [(src_d, f"{s}_Demand_Model.pkl"), (src_c, f"{s}_capacity_model.pkl")]:
        p = src / name
        if not p.exists():
            sys.exit(f"Missing model {p}")
        shutil.copy(p, out / name)

# ---------- P10: status files + fresh Knowledge Base ----------
from openpyxl import Workbook
kb = out / "Knowledge Base"; kb.mkdir()
for s in SERVICES:
    (out / f"ARM_{s}.txt").write_text("0\n0\n0\n0\n")
    (out / f"{s}.txt").write_text(f"1\nno scale\n1\n{a.maxr}\n{CPU_REQUEST[s]}\n")
    Workbook().save(kb / f"{s}.xlsx")

# ---------- compile check ----------
import py_compile
for p in out.glob("*.py"):
    py_compile.compile(str(p), doraise=True)
print(f"Ready: {out.resolve()}")
print(f"Scenario {a.maxr}R-{a.target}%, {secs} s, prediction window {a.window} s, "
      f"models from {'authors (published)' if not a.models else a.models}")
print("Repairs P1-P12 applied (see header of this script).")
