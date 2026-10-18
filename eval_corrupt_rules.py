import json, collections, sys
from cmbench.corrupt import build
from cmbench.evidence import retrieve
from cmbench.checkers import check_v0, check_v1, check_v1s
seed=int(sys.argv[1]) if len(sys.argv)>1 else 1
snaps=json.load(open("data/snapshots.json")); S={s["snapshot_id"]:s for s in snaps}
items=build(snaps,seed=seed)
res=collections.defaultdict(lambda: collections.Counter())
for it in items:
    ev=retrieve(S[it["snapshot_id"]],"", "full")
    for name,f in [("v0",check_v0),("v1",check_v1),("v1s",check_v1s)]:
        vd=f(it["answer"],ev)["verdict"]; res[(it["category"],name)][vd in ("supported","abstain")]+=1
for cat in sorted({i["category"] for i in items}):
    print(f"{cat:18s}", "  ".join(f"{n}:pass={res[(cat,n)][True]}/{sum(res[(cat,n)].values())}" for n in ["v0","v1","v1s"]))
