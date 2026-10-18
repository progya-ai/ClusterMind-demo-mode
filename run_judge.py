"""Model-based verifiers (baselines): LLM-as-judge (Qwen2.5-7B) and
self-check (the answering model itself). Score = P(UNSUPPORTED) from the
next-token distribution restricted to the two label tokens; decision =
score > 0.5. Deterministic (no sampling).
usage: python run_judge.py JUDGE_MODEL ITEMSET
  ITEMSET: corrupt  | gen:<model>_<retrieval>_<style>:<seed>
"""
import os as _os
RUNS = _os.environ.get("CM_RUNS_DIR", "runs")
RESULTS = _os.environ.get("CM_RESULTS_DIR", "results")
import json, os, sys, time
import numpy as np
from cmbench.evidence import retrieve
from cmbench.prompts import judge_system, judge_user
from cmbench import llm

judge, itemset = sys.argv[1], sys.argv[2]
snaps = {s["snapshot_id"]: s for s in json.load(open("data/snapshots.json"))}
Q = {q["qid"]: q for q in json.load(open("data/questions.json"))}

items = []
if itemset == "corrupt":
    for it in json.load(open("data/corruptions_test.json")):
        items.append({"id": it["iid"], "snapshot_id": it["snapshot_id"],
                      "question": "What is the current state of the cluster?", "answer": it["answer"], "evidence_mode": "full"})
else:
    tag, seed = itemset.split(":")[1], int(itemset.split(":")[2])
    for l in open(f"{RUNS}/gen_{tag}.jsonl"):
        r = json.loads(l)
        if r["seed"] != seed:
            continue
        q = Q[r["qid"]]
        items.append({"id": f"{r['qid']}|{seed}", "snapshot_id": q["snapshot_id"], "question": q["question"],
                      "answer": r["answer"], "evidence_mode": r["retrieval"]})


def fmt(model, system, user):
    if model.startswith("qwen"):
        return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n")
    return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n")


m = llm.get(judge)
tid_sup = m.tokenize(b"SUPPORTED", add_bos=False, special=False)[0]
tid_uns = m.tokenize(b"UNSUPPORTED", add_bos=False, special=False)[0]
assert tid_sup != tid_uns
os.makedirs(RUNS, exist_ok=True)
out = f"{RUNS}/judge_{judge}_{itemset.replace(':', '_')}.jsonl"
done = set()
if os.path.exists(out):
    done = {json.loads(l)["id"] for l in open(out)}
f = open(out, "a")
items.sort(key=lambda x: (x["snapshot_id"], x["id"]))
t0 = time.time()
for it in items:
    if it["id"] in done:
        continue
    ev = retrieve(snaps[it["snapshot_id"]], it["question"], mode=it["evidence_mode"], k=3)
    toks = m.tokenize(fmt(judge, judge_system(ev.text), judge_user(it["question"], it["answer"])).encode(),
                      add_bos=False, special=True)
    t = time.time()
    # reuse the KV cache for the longest common prefix (evidence block is shared per snapshot)
    prev = list(m.input_ids[:m.n_tokens])
    k = 0
    while k < min(len(prev), len(toks) - 1) and prev[k] == toks[k]:
        k += 1
    m.n_tokens = k
    m.eval(toks[k:])
    logits = np.ctypeslib.as_array(m._ctx.get_logits(), shape=(m.n_vocab(),)).astype(np.float64)
    a, b = logits[tid_sup], logits[tid_uns]
    p_uns = float(1.0 / (1.0 + np.exp(a - b)))
    f.write(json.dumps({"id": it["id"], "judge": judge, "p_unsupported": p_uns,
                        "latency_s": time.time() - t, "n_prompt": len(toks)}) + "\n"); f.flush()
print("done", out, time.time() - t0)
