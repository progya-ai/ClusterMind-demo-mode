"""E4a-idle: shutdown candidates = machines idle at proposal time t.
How often does such a node become busy before confirmation at t+D?"""
import json
import numpy as np
from cmbench.snapshots import load_tables, machine_state
rng = np.random.default_rng(11)
I = load_tables()
DELAYS = [60, 300, 900, 1800, 3600]
rows = []
for t in rng.uniform(2.5e6, 6.4e6, size=300).round():
    seen = I.loc[I["start_time"] <= t, "machine"].unique()
    s0 = machine_state(I, t, seen)
    idle = s0[s0.running_instances == 0].index.values
    if len(idle) == 0: continue
    ms = rng.choice(idle, size=min(10, len(idle)), replace=False)
    for d in DELAYS:
        s1 = machine_state(I, t + d, ms)
        for m in ms:
            rows.append({"t": int(t), "delay": d, "machine": m, "busy_at_confirm": bool(s1.loc[m].running_instances > 0),
                         "util_at_confirm": float(s1.loc[m].gpu_util_pct)})
json.dump(rows, open("data/action_idle_candidates.json", "w"))
print(len(rows))
