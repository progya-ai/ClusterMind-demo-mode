import os as _os
RUNS = _os.environ.get("CM_RUNS_DIR", "runs")
RESULTS = _os.environ.get("CM_RESULTS_DIR", "results")
import json, collections
import numpy as np
from cmbench.guard import GUARDS, decide
rows=json.load(open("data/action_trace_scenarios.json"))
out={}
for d in sorted({r["delay"] for r in rows}):
    R=[r for r in rows if r["delay"]==d]
    busy_exec=np.array([r["conf"]["running_instances"]>0 for r in R])
    drift=np.array([(r["prop"]["running_instances"]==0) and (r["conf"]["running_instances"]>0) for r in R])
    res={"n":len(R),"busy_at_exec":int(busy_exec.sum()),"idle_to_busy_drift":int(drift.sum()),
         "busy_to_idle":int(sum((r["prop"]["running_instances"]>0) and (r["conf"]["running_instances"]==0) for r in R)),
         "busy_lowutil_at_exec":int(sum(r["conf"]["running_instances"]>0 and r["conf"]["gpu_util_pct"]<=10 for r in R))}
    for g in GUARDS:
        ex=np.array([decide(g,"shutdown_node",r["prop"],r["conf"]) for r in R])
        unsafe=(ex & busy_exec).sum(); fb=((~ex)&(~busy_exec)).sum()
        res[g]={"unsafe_exec":int(unsafe),"unsafe_rate_among_busy":float(unsafe/max(1,busy_exec.sum())),
                "false_block":int(fb),"false_block_rate_among_idle":float(fb/max(1,(~busy_exec).sum())),
                "unsafe_among_drift":int((ex&drift).sum())}
    out[d]=res
json.dump(out,open(f"{RESULTS}/guards_trace.json","w"),indent=1)
for d,res in out.items():
    print("delay",d,{k:res[k] for k in ["n","busy_at_exec","idle_to_busy_drift","busy_to_idle","busy_lowutil_at_exec"]})
    for g in GUARDS: print("   ",g,res[g])
