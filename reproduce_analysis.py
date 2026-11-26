#!/usr/bin/env python3
"""
reproduce_analysis.py - rerun the whole evaluation analysis from the raw run files, in order.

    python -m pip install -r requirements.txt
    python reproduce_analysis.py          # run from the repository root

Needs the raw files in results/ (metrics, decisions, vertical actions, Locust stats, batch and
selector logs) and workloads/wc98_per_minute.csv. Writes results/eval/ (incl. the base-paper summary), results/rq1/ and
results/figures/. Stops at the first step that fails.
It does NOT retrain or change the frozen selector (results/selector/final_selector.json).
"""
import os, subprocess, sys, time

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # always run from the repository root

PY = sys.executable
STEPS = [
    ("Base-paper reproduction (RO1): HPA vs Smart HPA vs ProSmart HPA",
     ["analysis/baseline_summary.py", "--root", "."]),
    ("RQ3 main: eval_heldout runs 11-15",
     ["analysis/analyze_eval.py", "--root", ".", "--shape", "eval_heldout", "--runs", "11", "12", "13", "14", "15", "--tag", "_final5"]),
    ("RQ3 generalisation: eval_final runs 1-3",
     ["analysis/analyze_eval.py", "--root", ".", "--shape", "eval_final", "--runs", "1", "2", "3", "--tag", "_evalfinal"]),
    ("RQ3 preliminary round: eval_heldout runs 1-3",
     ["analysis/analyze_eval.py", "--root", ".", "--shape", "eval_heldout", "--runs", "1", "2", "3"]),
    ("Consistency check: extra vertical-only run (run 4) in the final session",
     ["analysis/analyze_eval.py", "--root", ".", "--shape", "eval_heldout", "--runs", "4",
      "--methods", "vertical", "--tag", "_vertical4"]),
]
for shape, runs, tag in [("eval_heldout", ["11", "12", "13", "14", "15"], "_final5"),
                         ("eval_final", ["1", "2", "3"], "_evalfinal")]:
    for w, wtag in [(["0.6", "0.3", "0.1"], ""), (["0.8", "0.2", "0.0"], "_w820"), (["0.4", "0.5", "0.1"], "_w451")]:
        STEPS.append((f"Cost score {shape} weights {'/'.join(w)}",
                      ["analysis/eval_cost_score.py", "--root", ".", "--shape", shape, "--runs", *runs, "--w", *w, "--tag", tag + wtag]))
    STEPS.append((f"Effect sizes {shape}", ["analysis/eval_effect_sizes.py", "--root", ".", "--tag", tag]))
    STEPS.append((f"Reserved vs used CPU {shape}",
                  ["analysis/eval_cpu_utilisation.py", "--root", ".", "--shape", shape, "--runs", *runs, "--tag", tag]))
    STEPS.append((f"Selector behaviour {shape}",
                  ["analysis/eval_selector_behaviour.py", "--root", ".", "--shape", shape, "--runs", *runs, "--tag", tag]))
STEPS += [
    ("Warm-up check (exploratory)",
     ["analysis/eval_warmup_check.py", "--root", ".", "--tags", "_final5:eval_heldout", "_evalfinal:eval_final"]),
    ("Feature skew: training vs live selector features", ["training/check_feature_skew.py", "--root", "."]),
    ("RQ1 forecasting comparison", ["analysis/forecasting.py", "--root", "."]),
    ("Figures", ["analysis/make_figures.py", "--root", "."]),
]

t0 = time.time()
for i, (name, cmd) in enumerate(STEPS, 1):
    print(f"\n[{i}/{len(STEPS)}] {name}\n    python {' '.join(cmd)}", flush=True)
    r = subprocess.run([PY, *cmd])
    if r.returncode != 0:
        sys.exit(f"\nSTOPPED: step {i} ({name}) failed with exit code {r.returncode}")
print(f"\nAll {len(STEPS)} steps finished in {time.time() - t0:.0f} s. Outputs: results/eval, results/rq1, results/figures")
