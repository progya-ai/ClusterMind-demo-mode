"""E4a: trace-driven guard evaluation (no LLM). For random (machine, t) pairs
we compute the typed state at proposal time t and at confirmation time t+D
directly from the instance table, then ask whether each guard would have
executed a shutdown that lands on a busy node at execution time."""
import json
import numpy as np
from cmbench.snapshots import load_tables, machine_state

rng = np.random.default_rng(7)
I = load_tables()
machines = I["machine"].unique()
DELAYS = [0, 60, 300, 900, 1800]
N_T = 150
times = rng.uniform(2.5e6, 6.4e6, size=N_T).round()
rows = []
for t in times:
    seen = I.loc[I["start_time"] <= t, "machine"].unique()
    ms = rng.choice(seen, size=min(20, len(seen)), replace=False)
    s0 = machine_state(I, t, ms)
    for d in DELAYS:
        s1 = s0 if d == 0 else machine_state(I, t + d, ms)
        for m in ms:
            a, b = s0.loc[m], s1.loc[m]
            rows.append({"t": int(t), "delay": d, "machine": m,
                         "prop": {k: float(a[k]) for k in ["running_instances", "gpu_util_pct", "gpu_mem_gb"]},
                         "conf": {k: float(b[k]) for k in ["running_instances", "gpu_util_pct", "gpu_mem_gb"]}})
json.dump(rows, open("data/action_trace_scenarios.json", "w"))
print(len(rows))
