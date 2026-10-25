"""
trace_shape.py - make Locust follow a WorldCup98-derived user profile.

Use together with the normal user script (same shop behaviour, real traffic shape):
    WORKLOAD=workloads/burst.csv locust -f locust_authors.py,trace_shape.py \
        --host $FE --headless --csv results/locust/x

The CSV has columns t_sec,users (one row every 5 s). Locust changes the number of
active users to follow it, then stops when the profile ends (15 min).
How the CSVs were made: 3 hours of WorldCup98 requests-per-minute (smoothed over 3 min),
compressed 12x, scaled so the segment's own peak = 300 users (floor 20).
"""
import csv, os
from locust import LoadTestShape


class TraceShape(LoadTestShape):
    path = os.environ.get("WORKLOAD", "workloads/burst.csv")
    with open(path) as f:
        rows = [(int(r["t_sec"]), int(r["users"])) for r in csv.DictReader(f)]
    step = rows[1][0] - rows[0][0]
    end = rows[-1][0] + step
    spawn_rate = float(os.environ.get("SPAWN_RATE", "20"))   # users per second while adjusting

    def tick(self):
        t = self.get_run_time()
        if t >= self.end:
            return None                       # profile finished -> stop the test
        i = min(int(t // self.step), len(self.rows) - 1)
        return self.rows[i][1], self.spawn_rate
