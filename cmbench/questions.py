"""
Question templates with gold answers computed directly from snapshot records
(the SQL-equivalent of a GROUP BY / ARGMAX over the typed state).
Each template has three author-written paraphrases.
"""
import numpy as np

TEMPLATES = {
    "argmax_util": [
        "Which node has the highest GPU utilization right now?",
        "What node is using the most GPU at the moment?",
        "Tell me which node is the busiest in terms of GPU usage.",
    ],
    "value_util": [
        "What is the GPU utilization of {node}?",
        "How busy is the GPU on {node}? Give me the utilization.",
        "Report the current GPU usage for {node}.",
    ],
    "value_mem": [
        "How much GPU memory is in use on {node}?",
        "What is the GPU memory consumption of {node}?",
        "Give me the GPU memory usage on {node} in GB.",
    ],
    "idle_set": [
        "Which nodes are idle?",
        "Are there any nodes with nothing running on them? List them.",
        "List every node that is not running any instances.",
    ],
    "value_failed": [
        "How many instances failed on {node} in the last hour?",
        "Did anything fail on {node} recently? How many instances?",
        "Give me the number of failed instances on {node} over the past hour.",
    ],
    "busy_yesno": [
        "Is {node} running any jobs right now?",
        "Is anything currently running on {node}?",
        "Is {node} busy or idle?",
    ],
    "trap_missing": [
        "What is the GPU utilization of {node}?",
        "How much GPU memory is in use on {node}?",
        "Is {node} running any jobs right now?",
    ],
    "argmax_failed": [
        "Which node had the most failed instances in the last hour?",
        "Where are failures concentrated? Which node had the most in the past hour?",
        "Name the node with the highest number of recent failures.",
    ],
}


def gold_for(tpl, recs, node=None):
    by = {r["node"]: r for r in recs}
    if tpl == "argmax_util":
        m = max(r["gpu_util_pct"] for r in recs)
        return {"nodes": [r["node"] for r in recs if r["gpu_util_pct"] == m], "value": m, "metric": "gpu_util_pct"}
    if tpl == "argmax_failed":
        m = max(r["failed_last_hour"] for r in recs)
        if m == 0:
            return {"nodes": [], "value": 0, "metric": "failed_last_hour"}
        return {"nodes": [r["node"] for r in recs if r["failed_last_hour"] == m], "value": m, "metric": "failed_last_hour"}
    if tpl == "idle_set":
        return {"nodes": sorted(r["node"] for r in recs if r["running_instances"] == 0)}
    if tpl == "value_util":
        return {"nodes": [node], "value": by[node]["gpu_util_pct"], "metric": "gpu_util_pct"}
    if tpl == "value_mem":
        return {"nodes": [node], "value": by[node]["gpu_mem_gb"], "metric": "gpu_mem_gb"}
    if tpl == "value_failed":
        return {"nodes": [node], "value": by[node]["failed_last_hour"], "metric": "failed_last_hour"}
    if tpl == "busy_yesno":
        return {"nodes": [node], "busy": by[node]["running_instances"] > 0,
                "value": by[node]["running_instances"], "metric": "running_instances"}
    if tpl == "trap_missing":
        return {"nodes": [], "abstain": True, "missing_node": node}
    raise KeyError(tpl)


def build_questions(snapshots, seed=0):
    rng = np.random.default_rng(seed)
    qs = []
    for s in snapshots:
        recs = s["nodes"]
        names = [r["node"] for r in recs]
        n = len(names)
        for tpl, paras in TEMPLATES.items():
            p = int(rng.integers(len(paras)))
            node = None
            if "{node}" in paras[p]:
                if tpl == "trap_missing":
                    node = f"Node-{int(rng.integers(n + 2, n + 12))}"  # e.g. Node-10..Node-19
                elif tpl == "busy_yesno":
                    node = str(rng.choice(names))
                else:
                    node = str(rng.choice(names))
            q = paras[p].format(node=node) if node else paras[p]
            qs.append({
                "qid": f"{s['snapshot_id']}_{tpl}", "snapshot_id": s["snapshot_id"],
                "template": tpl, "paraphrase": p, "question": q, "target_node": node,
                "gold": gold_for(tpl, recs, node),
            })
    return qs
