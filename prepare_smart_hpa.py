"""
Build a ready-to-run copy of Smart HPA from the cloned repo.

Why this is needed
------------------
The repo as shipped does not run straight from its folders:
  * the analyzer does `from Adaptive_Resource_Manager import *`, but the file is
    named "Adaptive Resource Manager.py" (spaces) -> ImportError
  * the analyzer does `from frontend import *` etc., but the manager scripts sit
    in a sub-folder -> ImportError
  * it reads/writes "./Knowledge Base/<service>.xlsx", so it must be run from a
    folder that contains "Knowledge Base"
This script fixes all three by building one flat folder, and optionally shortens
the test time (the paper's load test is 900 s = 15 min).

Robustness patches (do NOT change the scaling algorithm; report them in the thesis):
  R1 kubectl reads in the Microservice Managers are retried up to 5 times
     (a single network timeout used to crash the whole controller)
  R2 a deployment with 0 available replicas (empty availableReplicas) is read as 0
     instead of crashing with ValueError
  R3 if no pod has CPU metrics yet, CPU is taken as 0 instead of crashing
  R4 failed `kubectl scale` commands are retried up to 3 times instead of being
     silently dropped
Use --no-patches to build the authors' code exactly as released.

Usage
-----
    git clone https://github.com/HussainAhmad05/Smart_HPA.git
    python prepare_smart_hpa.py ./Smart_HPA --minutes 5
    cd smart_hpa_run
    python run_smart_hpa.py        # needs kubectl pointing at your cluster
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("repo", help="path to the cloned Smart_HPA repository")
ap.add_argument("--out", default="smart_hpa_run", help="output folder")
ap.add_argument("--minutes", type=float, default=15.0,
                help="controller run time in minutes (paper: 15)")
ap.add_argument("--force", action="store_true", help="overwrite output folder")
ap.add_argument("--no-patches", action="store_true",
                help="do not apply robustness patches R1-R4 (authors' code as released)")
args = ap.parse_args()

code = Path(args.repo) / "Smart HPA Codebase"
managers = code / "Microservice Managers"
arm = code / "Adaptive Resource Manager.py"
analyzer = code / "Microservice Capacity Analyzer.py"
kb = code / "Knowledge Base"
for p in (managers, arm, analyzer, kb):
    if not p.exists():
        sys.exit(f"Missing {p}. Pass the path to the cloned Smart_HPA repo.")

out = Path(args.out)
if out.exists():
    if not args.force:
        sys.exit(f"{out} already exists. Use --force to overwrite.")
    shutil.rmtree(out)
out.mkdir(parents=True)

RETRY_HELPER = (
    "import subprocess as _sp, time as _t\n"
    "def _retry_check_output(*a, **k):  # R1: retry kubectl reads on network errors\n"
    "    for _i in range(5):\n"
    "        try:\n"
    "            return _sp.check_output(*a, **k)\n"
    "        except _sp.CalledProcessError:\n"
    "            if _i == 4:\n"
    "                raise\n"
    "            _t.sleep(2)\n\n"
)


def patch(text, old, new, expected, label, fname):
    n = text.count(old)
    if n != expected:
        sys.exit(f"Patch {label} failed in {fname}: found {n} matches, expected {expected}")
    return text.replace(old, new)


# 1) managers -> same folder as the analyzer
for f in managers.glob("*.py"):
    text = f.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    if not args.no_patches:
        n_calls = text.count("subprocess.check_output(")
        text = patch(text, "subprocess.check_output(", "_retry_check_output(", n_calls, "R1", f.name)
        text = RETRY_HELPER + text
        text = patch(text, "int(Available_Replicas.strip(\"'\"))",
                     "int(Available_Replicas.strip(\"'\") or 0)", 1, "R2", f.name)
        text = patch(text, "current_cpu = math.ceil(statistics.mean(cpu_add))",
                     "current_cpu = math.ceil(statistics.mean(cpu_add)) if cpu_add else 0",
                     1, "R3", f.name)
    (out / f.name).write_text(text, encoding="utf-8")

# 2) resource manager: file name must be importable
shutil.copy(arm, out / "Adaptive_Resource_Manager.py")

# 3) knowledge base (Excel logs the controller reads and writes)
shutil.copytree(kb, out / "Knowledge Base")

# 4) analyzer, with the run time patched
text = analyzer.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
if not args.no_patches:
    text = patch(text, "    os.system(execute_command)\n",
                 "    for _i in range(3):  # R4: retry failed kubectl scale\n"
                 "        if os.system(execute_command) == 0:\n"
                 "            break\n"
                 "        time.sleep(2)\n", 1, "R4", analyzer.name)
seconds = int(round(args.minutes * 60))
new_text, n = re.subn(r"desired_time\s*=\s*900", f"desired_time = {seconds}", text)
if n != 1:
    sys.exit("Could not find the 'desired_time = 900' line to patch.")
(out / "run_smart_hpa.py").write_text(new_text, encoding="utf-8")

# 5) sanity check: do all modules import?
check = (
    "import importlib\n"
    "mods=['frontend','adservice','cartservice','checkoutservice','currencyservice',"
    "'emailservice','paymentservice','productcatalogservice','recommendationservice',"
    "'shippingservice','rediscart','Adaptive_Resource_Manager']\n"
    "[importlib.import_module(m) for m in mods]\n"
    "print('imports OK:', len(mods), 'modules')\n"
)
r = subprocess.run([sys.executable, "-c", check], cwd=out, capture_output=True, text=True)
print(r.stdout.strip() or r.stderr.strip())
if r.returncode != 0:
    print("\nInstall the dependencies first:\n"
          "  pip install numpy openpyxl psutil matplotlib")
    sys.exit(1)

print(f"\nReady: {out.resolve()}")
print(f"Controller will run for {seconds} s ({args.minutes} min).")
print("Robustness patches R1-R4:", "NOT applied (--no-patches)" if args.no_patches else "applied")
print("Next: point kubectl at your cluster, deploy Online Boutique, then")
print(f"  cd {out} && python run_smart_hpa.py")
