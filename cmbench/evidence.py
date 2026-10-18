"""
Evidence snapshot E: typed records rendered ONCE; the same rendering goes to
the model prompt, the operator's evidence panel, and the checker.

Two retrieval configurations are supported:
  - 'tfidf' : the original ClusterMind retriever (TF-IDF over templated
              per-node summaries, cosine similarity, top-k), re-implemented
              over the typed cluster records.
  - 'full'  : the entire snapshot (all nodes) is the evidence.
"""
import hashlib
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

METRICS = ["running_instances", "gpu_util_pct", "gpu_mem_gb", "failed_last_hour"]


def flags(r):
    f = []
    if r["running_instances"] == 0:
        f.append("idle")
    else:
        f.append("busy")
    if r["gpu_util_pct"] >= 400:
        f.append("heavy GPU load")
    if r["failed_last_hour"] > 0:
        f.append("recent failures")
    return f


def render_record(r, t):
    return (f"{r['node']} at t={t}s: running instances {r['running_instances']}, "
            f"GPU utilization {r['gpu_util_pct']:.1f}% (100% = one GPU), "
            f"GPU memory {r['gpu_mem_gb']:.1f} GB, "
            f"failed instances in last hour {r['failed_last_hour']} — "
            f"{', '.join(flags(r))}.")


class Evidence:
    """Frozen, hashed evidence set E (a list of typed records)."""

    def __init__(self, records, t):
        self.records = [dict(r) for r in records]
        self.t = t
        self.text = "\n".join(render_record(r, t) for r in self.records)
        self.digest = hashlib.sha256(
            json.dumps(self.records, sort_keys=True).encode()).hexdigest()[:16]
        self.by_node = {r["node"].lower(): r for r in self.records}

    def nodes(self):
        return [r["node"] for r in self.records]


def retrieve(snapshot, question, mode="full", k=3):
    recs, t = snapshot["nodes"], snapshot["t"]
    if mode == "full":
        return Evidence(recs, t)
    texts = [render_record(r, t) for r in recs]
    vec = TfidfVectorizer()
    M = vec.fit_transform(texts).toarray()
    q = vec.transform([question]).toarray()[0]
    def cos(a, b):
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / d) if d else 0.0
    scores = [cos(q, M[i]) for i in range(len(recs))]
    order = np.argsort(-np.array(scores), kind="stable")[:k]
    return Evidence([recs[i] for i in order], t)
