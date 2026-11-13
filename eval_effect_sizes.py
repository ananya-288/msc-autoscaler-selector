#!/usr/bin/env python3
"""
eval_effect_sizes.py - effect sizes and minimum detectable differences for the selector
comparisons, computed from the per-run results of analyze_eval.py.

    python eval_effect_sizes.py --root . --tag _final5        # eval_heldout, runs 11-15
    python eval_effect_sizes.py --root . --tag _evalfinal     # eval_final, runs 1-3

For each metric and each other method (selector vs other, same four metrics as analyze_eval.py):
  diff         selector mean - other mean (negative = selector lower)
  hedges_g     standardised difference: diff / pooled SD, with the small-sample correction
               J = 1 - 3 / (4 (n1 + n2) - 9). Rough guide: 0.2 small, 0.5 medium, 0.8 large.
               Not defined when both groups have no variation (pooled SD = 0).
  mdd          minimum detectable difference: the smallest true difference that a two-sided
               two-sample t-test with these run counts would detect with 80 % power at
               alpha = 0.05, given the observed pooled SD (noncentral t; equal-variance
               approximation of the Welch test used in analyze_eval.py; no Holm adjustment,
               so the real threshold after correction is somewhat larger).
  mdd_%_other  mdd as a percentage of the other method's mean.
Reading: if |diff| < mdd, the experiment could not have reliably detected a difference of
the observed size; "not significant" then means "too small to detect with n runs",
not "no difference".
"""
import argparse, os
import numpy as np, pandas as pd
from scipy import stats, optimize

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--tag", required=True)
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--power", type=float, default=0.80)
a = ap.parse_args()
P = lambda *x: os.path.join(a.root, "results", *x)
R = pd.read_csv(P("eval", f"per_run{a.tag}.csv"))
METRICS = ["slo_viol_%", "fe_p95_ms", "locust_p95_ms", "cpu_alloc_m"]
others = [m for m in R.method.unique() if m != "selector"]

def power_t(d, n1, n2, alpha):
    """power of a two-sided two-sample t-test for standardised difference d"""
    df = n1 + n2 - 2
    nc = d / np.sqrt(1 / n1 + 1 / n2)
    tc = stats.t.ppf(1 - alpha / 2, df)
    return 1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)

def min_d(n1, n2, alpha, power):
    return optimize.brentq(lambda d: power_t(d, n1, n2, alpha) - power, 1e-6, 50)

rows = []
for k in METRICS:
    for m in others:
        s, o = R[R.method == "selector"][k].astype(float), R[R.method == m][k].astype(float)
        n1, n2 = len(s), len(o)
        if n1 < 2 or n2 < 2:
            continue
        sp = np.sqrt(((n1 - 1) * s.var(ddof=1) + (n2 - 1) * o.var(ddof=1)) / (n1 + n2 - 2))
        diff = s.mean() - o.mean()
        J = 1 - 3 / (4 * (n1 + n2) - 9)
        g = diff / sp * J if sp > 0 else np.nan
        mdd = min_d(n1, n2, a.alpha, a.power) * sp if sp > 0 else np.nan
        rows.append({"metric": k, "vs": m, "n_sel": n1, "n_other": n2,
                     "selector_mean": s.mean(), "other_mean": o.mean(), "diff": diff,
                     "pooled_sd": sp, "hedges_g": g, "mdd": mdd,
                     "mdd_%_other": 100 * mdd / o.mean() if o.mean() else np.nan,
                     "detectable": (abs(diff) >= mdd) if sp > 0 else np.nan})
T = pd.DataFrame(rows)
T.to_csv(P("eval", f"effect_sizes{a.tag}.csv"), index=False)
pd.set_option("display.width", 220)
d0 = min_d(len(R[R.method == "selector"]), len(R[R.method == others[0]]), a.alpha, a.power)
print(f"Effect sizes, selector vs each method ({a.tag}); MDD at {int(a.power*100)} % power, alpha {a.alpha}")
print(f"(with these run counts, a difference must be about {d0:.2f} pooled SDs to be reliably detected)\n")
print(T.drop(columns=["n_sel", "n_other"]).round({"selector_mean": 1, "other_mean": 1, "diff": 1, "pooled_sd": 1,
      "hedges_g": 2, "mdd": 1, "mdd_%_other": 0}).to_string(index=False))
print(f"\nsaved results/eval/effect_sizes{a.tag}.csv")
