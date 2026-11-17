#!/usr/bin/env python3
"""
forecasting.py - RQ1: how well can simple and learned methods forecast the next interval's
request rate, and is the selector's naive trend forecast good enough?

    python analysis/forecasting.py --root .            # needs: pip install statsmodels

Data: workloads/wc98_per_minute.csv (WorldCup98 requests per minute), tournament period
10 June - 12 July 1998 (the period the workload shapes were taken from).

Two settings, both forecasting ONE step ahead:
  A  "trace minute":    the raw per-minute series.
  B  "selector minute": 12-minute block means. The workloads compress the trace 12x, so one minute
                        of the live selector covers 12 trace minutes; this is the step the selector
                        actually forecasts.
Methods:
  persistence   next = current
  naive_trend   next = current + (current - previous)      <- the selector's 'forecast' feature
  holt          Holt's linear exponential smoothing; parameters fitted on the training part
  arima         ARIMA(p,1,q), order chosen by AIC on the training part from a small grid
  gbr           histogram gradient boosting on lagged values, rolling means/std and time of day;
                predicts the change to the next step; 4 settings compared on the validation part
Splits, in time order (no shuffling): train 60 %, validation 20 %, test 20 %. Holt and ARIMA are
fitted on train + validation; their parameters are then kept fixed on the test part, where every
forecast uses only data up to that step. Scores are on the test part only.
Metrics: MAE, RMSE, MAPE (steps with an actual value of 0 are left out of MAPE), and MAE relative
to persistence (< 1 = better than persistence). A bootstrap over days gives a 95% CI for the MAE
difference between each method and naive_trend.
"""
import argparse, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--start", default="1998-06-10")
ap.add_argument("--end", default="1998-07-13")            # exclusive
ap.add_argument("--block", type=int, default=12)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
try:
    from statsmodels.tsa.holtwinters import Holt
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    raise SystemExit("statsmodels is missing: pip install statsmodels")
from sklearn.ensemble import HistGradientBoostingRegressor

s = pd.read_csv(os.path.join(a.root, "workloads", "wc98_per_minute.csv"), index_col=0, parse_dates=True)["requests"]
s = s[(s.index >= a.start) & (s.index < a.end)].astype(float)
OUT = os.path.join(a.root, "results", "rq1"); os.makedirs(OUT, exist_ok=True)
print(f"series: {len(s)} minutes, {s.index.min()} to {s.index.max()}; zero minutes: {(s == 0).sum()}")

def features(y, idx, minutes_per_step):
    """lag features known at time t (to predict t+1)"""
    f = pd.DataFrame(index=y.index)
    for k in range(12):
        f[f"lag{k}"] = y.shift(k)
    f["d1"] = y - y.shift(1); f["d2"] = y.shift(1) - y.shift(2)
    for w in (3, 12):
        f[f"mean{w}"] = y.rolling(w).mean(); f[f"std{w}"] = y.rolling(w).std()
    tod = (idx.hour * 60 + idx.minute) / 1440.0
    f["tod_sin"] = np.sin(2 * np.pi * tod); f["tod_cos"] = np.cos(2 * np.pi * tod)
    return f

def run(y, name):
    y = y.reset_index(drop=False); idx = pd.DatetimeIndex(y.iloc[:, 0]); y = pd.Series(y.iloc[:, 1].to_numpy(), index=idx)
    n = len(y); n_tr, n_va = int(0.6 * n), int(0.2 * n)
    te0 = n_tr + n_va                                         # first forecast target index in test
    tgt = np.arange(te0, n)                                   # we forecast y[t] using data up to t-1
    actual = y.to_numpy()[tgt]
    preds, info = {}, {}
    v = y.to_numpy()
    preds["persistence"] = v[tgt - 1]
    preds["naive_trend"] = np.maximum(0, v[tgt - 1] + (v[tgt - 1] - v[tgt - 2]))
    # Holt: fit on train+val, then run the recursion with fixed parameters over the whole series
    t0 = time.time()
    hf = Holt(v[:te0], initialization_method="estimated").fit(optimized=True)
    al, be = hf.params["smoothing_level"], hf.params["smoothing_trend"]
    lvl, trd = hf.params["initial_level"], hf.params["initial_trend"]
    hp = np.empty(n)
    for t in range(n):
        hp[t] = lvl + trd                                     # forecast for step t
        nl = al * v[t] + (1 - al) * (lvl + trd); trd = be * (nl - lvl) + (1 - be) * trd; lvl = nl
    preds["holt"] = np.maximum(0, hp[tgt]); info["holt"] = f"alpha={al:.3f}, beta={be:.3f}"
    # ARIMA: small grid, AIC on train+val; then apply the fixed parameters to the whole series
    t0 = time.time(); best = None
    for order in [(0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1), (2, 1, 2)]:
        try:
            r = ARIMA(v[:te0], order=order, trend="n").fit()
            if best is None or r.aic < best[1].aic:
                best = (order, r)
        except Exception as e:
            print(f"  ARIMA{order} failed: {e}")
    full = best[1].apply(v)
    ap_ = full.get_prediction(start=te0, end=n - 1, dynamic=False).predicted_mean
    preds["arima"] = np.maximum(0, np.asarray(ap_)); info["arima"] = f"order={best[0]}"
    # gradient boosting on lag features, predicting the change y[t+1] - y[t]
    t0 = time.time()
    F = features(y, idx, 1)
    X = F.shift(1)                                            # row t uses features known at t-1
    d = pd.Series(v, index=y.index) - pd.Series(v, index=y.index).shift(1)
    ok = X.notna().all(axis=1).to_numpy()
    rows = np.arange(n)
    tr = rows[ok & (rows < n_tr)]; va = rows[ok & (rows >= n_tr) & (rows < te0)]
    best_g = None
    for lr in (0.05, 0.1):
        for leaves in (15, 31):
            m = HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaves, max_iter=300,
                                              random_state=a.seed).fit(X.iloc[tr], d.iloc[tr])
            mae = np.mean(np.abs(v[va - 1] + m.predict(X.iloc[va]) - v[va]))
            if best_g is None or mae < best_g[0]:
                best_g = (mae, lr, leaves)
    _, lr, leaves = best_g
    trva = rows[ok & (rows < te0)]
    m = HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaves, max_iter=300,
                                      random_state=a.seed).fit(X.iloc[trva], d.iloc[trva])
    preds["gbr"] = np.maximum(0, v[tgt - 1] + m.predict(X.iloc[tgt]))
    info["gbr"] = f"learning_rate={lr}, max_leaf_nodes={leaves}"

    # scores
    rows_out = []
    pmae = np.mean(np.abs(preds["persistence"] - actual))
    nz = actual > 0
    day = idx[tgt].normalize()
    days = day.unique(); rng = np.random.default_rng(a.seed)
    err = {k: np.abs(p - actual) for k, p in preds.items()}
    bidx = [np.where(day == d_)[0] for d_ in days]
    for k, p in preds.items():
        e = p - actual
        diffs = []
        for _ in range(a.boot):                                # bootstrap over days
            pick = np.concatenate([bidx[j] for j in rng.integers(0, len(days), len(days))])
            diffs.append(err[k][pick].mean() - err["naive_trend"][pick].mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows_out.append({"setting": name, "method": k, "MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2)),
                         "MAPE_%": 100 * np.mean(np.abs(e[nz]) / actual[nz]), "rel_MAE_vs_persistence": np.mean(np.abs(e)) / pmae,
                         "MAE_minus_naive": np.mean(np.abs(e)) - np.mean(np.abs(preds["naive_trend"] - actual)),
                         "CI95_low": lo, "CI95_high": hi, "settings": info.get(k, "")})
    pd.DataFrame({"time": idx[tgt], "actual": actual, **preds}).to_csv(os.path.join(OUT, f"rq1_predictions_{name}.csv"), index=False)
    return pd.DataFrame(rows_out), idx[te0], len(tgt), len(days)

A = s
B = s.resample(f"{a.block}min").mean().dropna()
res = []
for y, name, unit in [(A, "A_trace_minute", "1 trace minute"), (B, "B_selector_minute", f"{a.block}-minute block")]:
    print(f"\nsetting {name}: one step = {unit}, {len(y)} steps")
    r, start, nt, nd = run(y, name)
    print(f"  test part: {nt} steps from {start} ({nd} days)")
    res.append(r)
R = pd.concat(res)
R.to_csv(os.path.join(OUT, "rq1_results.csv"), index=False)
pd.set_option("display.width", 220)
show = R.drop(columns="settings").copy()
for c in ["MAE", "RMSE", "MAE_minus_naive", "CI95_low", "CI95_high"]:
    show[c] = show[c].round(0)
show["MAPE_%"] = show["MAPE_%"].round(2); show["rel_MAE_vs_persistence"] = show["rel_MAE_vs_persistence"].round(3)
print("\nTest-set results (requests per minute; lower is better)")
print("MAE_minus_naive < 0 = better than the selector's naive trend; CI95 = bootstrap over test days\n")
print(show.to_string(index=False))
print("\nfitted settings:")
for r in R.itertuples():
    if r.settings:
        print(f"  {r.setting:18s} {r.method:12s} {r.settings}")
print(f"\nsaved {OUT}/rq1_results.csv and rq1_predictions_*.csv")
