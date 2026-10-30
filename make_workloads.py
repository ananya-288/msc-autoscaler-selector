#!/usr/bin/env python3
"""
make_workloads.py - build the five Locust traffic shapes from the WorldCup98 trace.

    python make_workloads.py                 # downloads the data, writes workloads/*.csv + figure
    python make_workloads.py --no-download   # reuse workloads/wc98_per_minute.csv

Data source
  Original trace: Internet Traffic Archive, WorldCup98 access logs (Arlitt and Jin, 2000).
  Per-minute request counts: github.com/AndrewLawrence80/WorldCup98, folder GroupedLog
  (files wc_day<D>_<k>.csv with columns: minute timestamp, requests, bytes).

Steps
  1. Download every GroupedLog file, sum requests per minute -> workloads/wc98_per_minute.csv
  2. For each shape take 180 minutes (3 hours) starting at a fixed timestamp (below),
     smooth with a centred 3-minute rolling mean.
  3. Scale so the segment's own peak = 300 users (floor 20 users):
         users = max(20, round(300 * smoothed / max(smoothed)))
  4. Compress 12x: each trace minute becomes 5 seconds -> 180 rows, 15 minutes (t_sec = 0, 5, ..., 895).

Segments (start time, GMT as in the trace)
  gentle        1998-06-12 06:30   steady, low variation
  burst         1998-06-25 15:00   sharp peak in the middle
  rise          1998-07-07 17:30   strong steady increase
  decline       1998-06-15 20:00   high, then strong decrease
  eval_heldout  1998-07-03 18:45   mixed (rise, peak, fall); from days not used for training
Training shapes were picked from automatically ranked 3-hour windows (burst ratio, slope,
monotonicity, variation) during the tournament (10 June - 12 July 1998). The held-out shape
was chosen from 30 June - 3 July as the window whose smoothed peak is in the middle
(40-70 % of the window) with the largest peak / first-30-minutes ratio.
"""
import argparse, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

BASE = "https://raw.githubusercontent.com/AndrewLawrence80/WorldCup98/HEAD/GroupedLog/"
SEGMENTS = {"gentle": "1998-06-12 06:30", "burst": "1998-06-25 15:00", "rise": "1998-07-07 17:30",
            "decline": "1998-06-15 20:00", "eval_heldout": "1998-07-03 18:45"}
PEAK, FLOOR, SPEED, LENGTH = 300, 20, 12, 180

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="workloads")
ap.add_argument("--no-download", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
per_min = os.path.join(a.out, "wc98_per_minute.csv")

if not a.no_download:
    def fetch(name):
        try:
            with urllib.request.urlopen(BASE + name, timeout=30) as r:
                return pd.read_csv(r, names=["ts", "n", "bytes"], parse_dates=["ts"])
        except Exception:
            return None                                   # that day has fewer parts
    names = [f"wc_day{d}_{k}.csv" for d in range(5, 93) for k in range(1, 12)]
    with ThreadPoolExecutor(24) as ex:
        parts = [p for p in ex.map(fetch, names) if p is not None]
    print(f"downloaded {len(parts)} files")
    s = pd.concat(parts).groupby("ts").n.sum().sort_index().asfreq("min", fill_value=0)
    s.to_csv(per_min, header=["requests"])
    print(f"wrote {per_min}: {len(s)} minutes, {s.index.min()} to {s.index.max()}")

s = pd.read_csv(per_min, index_col=0, parse_dates=True)["requests"]
for name, start in SEGMENTS.items():
    w = s[pd.Timestamp(start):][:LENGTH].astype(float)
    sm = w.rolling(3, center=True, min_periods=1).mean().values
    users = np.maximum(FLOOR, np.round(PEAK * sm / sm.max())).astype(int)
    t = (np.arange(LENGTH) * 60 / SPEED).astype(int)
    pd.DataFrame({"t_sec": t, "users": users}).to_csv(os.path.join(a.out, f"{name}.csv"), index=False)
    print(f"{name:13s} from {start}: users {users.min()}-{users.max()}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 5, figsize=(18, 3))
    for k, name in enumerate(SEGMENTS):
        d = pd.read_csv(os.path.join(a.out, f"{name}.csv"))
        ax[k].plot(d.t_sec / 60, d.users); ax[k].set_ylim(0, 320); ax[k].set_xlabel("test minute")
        ax[k].set_title(f"{name}\nWorldCup98 {SEGMENTS[name]} (3 h -> 15 min)", fontsize=9)
    ax[0].set_ylabel("Locust users"); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "workload_shapes.png"), dpi=130)
    print("wrote workload_shapes.png")
except ImportError:
    pass
