import json, time
from cmbench.snapshots import load_tables, build_snapshots
from cmbench.questions import build_questions
t=time.time(); I=load_tables(); print("loaded",time.time()-t)
snaps=build_snapshots(I,n_snapshots=40,n_nodes=8,seed=0)
qs=build_questions(snaps,seed=0)
json.dump(snaps,open("data/snapshots.json","w"),indent=1); json.dump(qs,open("data/questions.json","w"),indent=1)
print(len(snaps),len(qs),time.time()-t)
import collections
c=collections.Counter()
for s in snaps:
    for r in s["nodes"]:
        c["idle" if r["running_instances"]==0 else ("busy_hi" if r["gpu_util_pct"]>=50 else ("busy_lo" if r["gpu_util_pct"]<=10 else "busy_mid"))]+=1
        c["fail>0"]+= r["failed_last_hour"]>0
print(c)
print(json.dumps(snaps[3],indent=0)[:1500])
print([ (q["question"],q["gold"]) for q in qs[:8]])
