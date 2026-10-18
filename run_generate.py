"""Generate answers for the test questions. Resumable JSONL output.
usage: python run_generate.py MODEL RETRIEVAL STYLE SEEDS [LIMIT]
"""
import os as _os
RUNS = _os.environ.get("CM_RUNS_DIR", "runs")
RESULTS = _os.environ.get("CM_RESULTS_DIR", "results")
import json, os, sys, time
from cmbench.evidence import retrieve
from cmbench.prompts import diagnosis_system, diagnosis_user
from cmbench import llm

model, mode, style, seeds = sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4].split(",")]
limit = int(sys.argv[5]) if len(sys.argv) > 5 else None
snaps = {s["snapshot_id"]: s for s in json.load(open("data/snapshots.json"))}
qs = json.load(open("data/questions.json"))
if limit:
    qs = qs[:limit]
os.makedirs(RUNS, exist_ok=True)
out = f"{RUNS}/gen_{model}_{mode}_{style}.jsonl"
done = set()
if os.path.exists(out):
    for l in open(out):
        r = json.loads(l); done.add((r["qid"], r["seed"]))
f = open(out, "a")
t0 = time.time(); n = 0
for q in qs:
    ev = retrieve(snaps[q["snapshot_id"]], q["question"], mode=mode, k=3)
    for seed in seeds:
        if (q["qid"], seed) in done:
            continue
        r = llm.chat(model, diagnosis_system(ev.text), diagnosis_user(q["question"], style), seed=seed)
        rec = {"qid": q["qid"], "seed": seed, "model": model, "retrieval": mode, "style": style,
               "evidence_nodes": ev.nodes(), "evidence_digest": ev.digest, "answer": r["text"],
               "latency_s": r["latency_s"], "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"]}
        f.write(json.dumps(rec) + "\n"); f.flush(); n += 1
print(f"done {n} in {time.time()-t0:.0f}s -> {out}")
