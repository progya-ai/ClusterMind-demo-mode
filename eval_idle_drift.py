import os as _os
RUNS = _os.environ.get("CM_RUNS_DIR", "runs")
RESULTS = _os.environ.get("CM_RESULTS_DIR", "results")
import json, numpy as np, collections
rows=json.load(open("data/action_idle_candidates.json"))
rng=np.random.default_rng(0)
out={}
for d in sorted({r["delay"] for r in rows}):
    R=[r for r in rows if r["delay"]==d]
    t_ids=np.array([r["t"] for r in R]); b=np.array([r["busy_at_confirm"] for r in R],float)
    ut=sorted(set(t_ids)); idx={t:np.where(t_ids==t)[0] for t in ut}
    bs=[]
    for _ in range(2000):
        pick=rng.choice(ut,size=len(ut)); ii=np.concatenate([idx[t] for t in pick]); bs.append(b[ii].mean())
    lo,hi=np.percentile(bs,[2.5,97.5])
    # would the util heuristic at confirm have caught these?  (busy but util<=10 -> missed)
    missed_util=sum(r["busy_at_confirm"] and r["util_at_confirm"]<=10 for r in R)
    out[d]={"n":len(R),"n_times":len(ut),"drift_rate":float(b.mean()),"ci":[float(lo),float(hi)],"n_drift":int(b.sum()),"drift_missed_by_util":int(missed_util)}
    print(d,out[d])
json.dump(out,open(f"{RESULTS}/idle_drift.json","w"),indent=1)
