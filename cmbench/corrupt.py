"""Controlled answer set for checker evaluation (E1a). Every item has a label
known by construction: 'supported' (all claims true w.r.t. the snapshot) or
'unsupported' (contains at least one false claim), plus a category.
Category 'no_claim' is reported separately (no ground-truth support label)."""
import numpy as np

UTIL_PH = ["{n} is at {v}% GPU utilization.", "GPU utilization on {n} is {v}%.", "{n} shows a GPU utilization of {v}%."]
MEM_PH = ["{n} is using {v} GB of GPU memory.", "GPU memory in use on {n}: {v} GB.", "{n} has {v} GB of GPU memory allocated."]
RUN_PH = ["{n} has {v} running instances.", "There are {v} instances running on {n}.", "{n} is running {v} jobs."]
FAIL_PH = ["{n} had {v} failed instances in the last hour.", "In the past hour, {v} instances failed on {n}."]
PH = {"gpu_util_pct": UTIL_PH, "gpu_mem_gb": MEM_PH, "running_instances": RUN_PH, "failed_last_hour": FAIL_PH}


def fmt(metric, x, rng, exact=False):
    if metric in ("running_instances", "failed_last_hour"):
        return str(int(x))
    if exact or rng.random() < 0.5:
        return f"{x:.1f}"
    return str(int(round(x)))  # the rounded form an LM typically writes


def sentence(metric, node, x, rng, exact=False):
    return str(rng.choice(PH[metric])).format(n=node, v=fmt(metric, x, rng, exact))


def perturb(metric, x, rng, large=False):
    if metric in ("running_instances", "failed_last_hour"):
        d = int(rng.integers(3, 10)) if large else int(rng.integers(1, 3))
        return max(0, x + d) if (rng.random() < 0.5 or x - d < 0) else x - d
    if large:
        return x * 2 + 20
    d = max(3.0, abs(x) * float(rng.uniform(0.08, 0.2)))
    return x + d if (rng.random() < 0.5 or x - d < 0) else x - d


def build(snapshots, seed=0):
    rng = np.random.default_rng(seed)
    items = []
    for s in snapshots:
        recs = s["nodes"]
        n = len(recs)
        def pick():
            return recs[int(rng.integers(n))]
        def add(cat, label, text):
            items.append({"iid": f"{s['snapshot_id']}_{cat}_{len([i for i in items if i['snapshot_id']==s['snapshot_id']])}",
                          "snapshot_id": s["snapshot_id"], "category": cat, "label": label, "answer": text})
        metrics = ["gpu_util_pct", "gpu_mem_gb", "running_instances", "failed_last_hour"]
        # correct, single and multi-claim
        for _ in range(2):
            r = pick(); m = str(rng.choice(metrics))
            add("correct_single", "supported", sentence(m, r["node"], r[m], rng))
        r = pick(); m1, m2 = rng.choice(metrics, size=2, replace=False)
        add("correct_multi", "supported", sentence(m1, r["node"], r[m1], rng) + " " + sentence(m2, r["node"], r[m2], rng))
        # numeric corruptions
        r = pick(); m = str(rng.choice(metrics))
        add("value_small", "unsupported", sentence(m, r["node"], perturb(m, r[m], rng), rng, exact=True))
        r = pick(); m = str(rng.choice(metrics))
        add("value_large", "unsupported", sentence(m, r["node"], perturb(m, r[m], rng, large=True), rng, exact=True))
        # metric swap: a true number of another metric of the same node
        for _ in range(10):
            r = pick(); m1, m2 = rng.choice(["gpu_util_pct", "gpu_mem_gb", "running_instances"], size=2, replace=False)
            if abs(r[m1] - r[m2]) > 2:
                add("metric_swap", "unsupported", sentence(str(m1), r["node"], r[m2], rng, exact=True)); break
        # node swap: another node's true value attributed to this node
        for _ in range(10):
            a, b = rng.choice(n, size=2, replace=False); m = str(rng.choice(metrics))
            if abs(recs[a][m] - recs[b][m]) > 2:
                add("node_swap", "unsupported", sentence(m, recs[a]["node"], recs[b][m], rng, exact=True)); break
        # fabricated node whose name has an evidence node as prefix (Node-1x)
        fake = f"Node-1{int(rng.integers(0, 10))}" if n < 10 else f"Node-{n + 5}"
        m = str(rng.choice(metrics)); r = pick()
        add("fabricated_node", "unsupported", sentence(m, fake, r[m], rng))
        # true claim followed by a fabricated one
        for _ in range(10):
            r = pick(); m1, m2 = rng.choice(metrics, size=2, replace=False)
            fakev = perturb(str(m2), r[m2], rng, large=True)
            add("extra_fabricated", "unsupported", sentence(str(m1), r["node"], r[m1], rng) + " " + sentence(str(m2), r["node"], fakev, rng, exact=True)); break
        # categorical state claims
        idle = [x for x in recs if x["running_instances"] == 0]; busy = [x for x in recs if x["running_instances"] > 0]
        if idle and busy:
            add("state_correct", "supported", f"{rng.choice(idle)['node']} is idle.")
            add("state_false", "unsupported", f"{rng.choice(busy)['node']} is idle and can be shut down.")
        # presence claims
        add("absence_correct", "supported", f"{fake} is not present in the telemetry.")
        add("absence_false", "unsupported", f"{pick()['node']} is not present in the telemetry.")
        # qualitative falsehood around a true number (outside v1's claim language)
        if idle:
            r = idle[int(rng.integers(len(idle)))]
            add("qualitative_false", "unsupported", f"{r['node']} is overloaded and close to failing, with {r['gpu_util_pct']:.1f}% GPU utilization.")
        # no verifiable claim at all
        add("no_claim", None, f"{pick()['node']} looks healthy and needs no action.")
    return items
