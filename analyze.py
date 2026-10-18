"""All quantitative results. Writes results/*.json and prints tables."""
import os as _os
RUNS = _os.environ.get("CM_RUNS_DIR", "runs")
RESULTS = _os.environ.get("CM_RESULTS_DIR", "results")
import json, os, glob, collections, time, re
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
from cmbench.evidence import retrieve
from cmbench.checkers import check_v0, check_v1, check_v1s, check_v2
from cmbench.score import score

os.makedirs(RESULTS, exist_ok=True)
SN = {s["snapshot_id"]: s for s in json.load(open("data/snapshots.json"))}
Q = {q["qid"]: q for q in json.load(open("data/questions.json"))}
RNG = np.random.default_rng(0)
B = 2000
CHECKERS = {"v0": check_v0, "v1": check_v1, "v1s": check_v1s, "v2": lambda a, e: check_v2(a, e, "")}
DELIVER = {"supported", "abstain"}


def boot(values, groups, stat=np.mean, b=B):
    values, groups = np.asarray(values, float), np.asarray(groups)
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}
    out = []
    for _ in range(b):
        ii = np.concatenate([idx[g] for g in RNG.choice(ug, size=len(ug))])
        v = values[ii]
        out.append(stat(v) if len(v) else np.nan)
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


def boot_ratio(num, den, groups, b=B):
    num, den, groups = np.asarray(num, float), np.asarray(den, float), np.asarray(groups)
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}
    out = []
    for _ in range(b):
        ii = np.concatenate([idx[g] for g in RNG.choice(ug, size=len(ug))])
        d = den[ii].sum()
        out.append(num[ii].sum() / d if d else np.nan)
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_only": n10, "b_only": n01, "p": float(p)}


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def load_gen(tag):
    p = f"{RUNS}/gen_{tag}.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def annotate(rows):
    for r in rows:
        q = Q[r["qid"]]
        ev = retrieve(SN[q["snapshot_id"]], q["question"], mode=r["retrieval"], k=3)
        r["_snap"] = q["snapshot_id"]; r["_tpl"] = q["template"]
        r["correct"] = bool(score(q, r["answer"]))
        for k, f in CHECKERS.items():
            r[k] = f(r["answer"], ev)["verdict"]
        r["v2"] = check_v2(r["answer"], ev, q["question"])["verdict"]
        g = set(q["gold"]["nodes"]) | ({q["target_node"]} if q["target_node"] and q["template"] != "trap_missing" else set())
        r["gold_in_evidence"] = g <= set(ev.nodes())
    return rows


def gate_stats(rows, gate):
    corr = np.array([r["correct"] for r in rows]); grp = [r["_snap"] for r in rows]
    if gate == "none":
        dl = np.ones(len(rows), bool)
    elif gate.startswith("judge") or gate.startswith("self"):
        dl = np.array([r.get(gate, 1.0) < 0.5 for r in rows])
    else:
        dl = np.array([r[gate] in DELIVER if gate != "v0" else r[gate] == "supported" for r in rows])
    cov = dl.mean(); acc_d = corr[dl].mean() if dl.any() else float("nan")
    return {"coverage": float(cov), "coverage_ci": boot(dl, grp),
            "delivered_acc": float(acc_d), "delivered_acc_ci": boot_ratio(corr & dl, dl, grp),
            "delivered_err": float(1 - acc_d) if dl.any() else float("nan"),
            "errors_delivered_per_100": float(((~corr) & dl).mean() * 100),
            "correct_withheld_per_100": float((corr & ~dl).mean() * 100),
            "n": len(rows), "_dl": dl}


def e2_endtoend():
    global RNG
    RNG = np.random.default_rng(0)
    out = {}
    for tag in ["qwen1.5b_full_contract", "llama1b_full_contract", "qwen1.5b_tfidf_contract", "llama1b_tfidf_contract",
                "qwen1.5b_tfidf_original", "qwen1.5b_full_original", "qwen7b_full_contract"]:
        rows = annotate(load_gen(tag))
        if not rows:
            continue
        res = {"n": len(rows)}
        by_seed = collections.defaultdict(list)
        for r in rows:
            by_seed[r["seed"]].append(r)
        res["acc_by_seed"] = {s: float(np.mean([r["correct"] for r in v])) for s, v in by_seed.items()}
        rows0 = by_seed[min(by_seed)]
        res["acc"] = float(np.mean([r["correct"] for r in rows]))
        res["acc_ci"] = boot([r["correct"] for r in rows], [r["_snap"] for r in rows])
        res["per_template_acc"] = {t: float(np.mean([r["correct"] for r in rows if r["_tpl"] == t])) for t in sorted({r["_tpl"] for r in rows})}
        res["verdicts"] = {k: dict(collections.Counter(r[k] for r in rows)) for k in CHECKERS}
        res["gold_in_evidence"] = float(np.mean([r["gold_in_evidence"] for r in rows if r["_tpl"] != "trap_missing"]))
        gates = {}
        for g in ["none", "v0", "v1", "v1s", "v2"]:
            st = [gate_stats(v, g) for v in by_seed.values()]
            pooled = gate_stats(rows, g)  # CIs: bootstrap over snapshots on all seeds pooled
            gates[g] = {k: (float(np.mean([s[k] for s in st])) if not k.endswith("_ci") else pooled[k])
                        for k in st[0] if k != "_dl"}
            gates[g]["delivered_err_by_seed"] = [s["delivered_err"] for s in st]
            gates[g]["coverage_by_seed"] = [s["coverage"] for s in st]
        res["gates"] = gates
        # paired tests on seed-0: are delivered answers more often correct under v1 than v0?
        s0 = {g: gate_stats(rows0, g)["_dl"] for g in ["v0", "v1", "v1s", "v2"]}
        corr0 = np.array([r["correct"] for r in rows0])
        res["mcnemar_error_delivered_v0_vs_v1"] = mcnemar((~corr0) & s0["v0"], (~corr0) & s0["v1"])
        res["mcnemar_error_delivered_v0_vs_v1s"] = mcnemar((~corr0) & s0["v0"], (~corr0) & s0["v1s"])
        res["mcnemar_error_delivered_v1s_vs_v2"] = mcnemar((~corr0) & s0["v1s"], (~corr0) & s0["v2"])
        # per-template gating
        res["per_template_gate_v1s"] = {}
        for t in sorted({r["_tpl"] for r in rows0}):
            rr = [r for r in rows if r["_tpl"] == t]
            st = gate_stats(rr, "v1s"); st2 = gate_stats(rr, "v2")
            res["per_template_gate_v1s"][t] = {"coverage": st["coverage"], "delivered_acc": st["delivered_acc"],
                                               "coverage_v2": st2["coverage"], "delivered_acc_v2": st2["delivered_acc"],
                                               "acc": float(np.mean([r["correct"] for r in rr]))}
        if "tfidf" in tag:
            miss = [r for r in rows if not r["gold_in_evidence"] and r["_tpl"] != "trap_missing"]
            res["tfidf_miss_rate"] = len(miss) / max(1, sum(r["_tpl"] != "trap_missing" for r in rows))
            res["tfidf_correct_but_flagged_due_to_miss"] = float(np.mean([r["correct"] and r["v1s"] not in DELIVER for r in miss])) if miss else None
        res["latency_mean_s"] = float(np.mean([r["latency_s"] for r in rows]))
        res["latency_p50_s"] = float(np.median([r["latency_s"] for r in rows]))
        res["latency_p95_s"] = float(np.percentile([r["latency_s"] for r in rows], 95))
        res["completion_tokens_mean"] = float(np.mean([r["completion_tokens"] for r in rows]))
        out[tag] = res
    json.dump(out, open(f"{RESULTS}/e2_endtoend.json", "w"), indent=1)
    return out


def load_judge(judge, itemset):
    p = f"{RUNS}/judge_{judge}_{itemset}.jsonl"
    return {json.loads(l)["id"]: json.loads(l)["p_unsupported"] for l in open(p)} if os.path.exists(p) else {}


def detect(y, flag):
    y, flag = np.asarray(y, bool), np.asarray(flag, bool)
    tp, fp, fn, tn = (y & flag).sum(), (~y & flag).sum(), (y & ~flag).sum(), (~y & ~flag).sum()
    P = tp / (tp + fp) if tp + fp else float("nan"); R = tp / (tp + fn) if tp + fn else float("nan")
    return {"precision": float(P), "recall": float(R), "f1": float(2 * P * R / (P + R)) if P + R else 0.0,
            "fpr": float(fp / (fp + tn)) if fp + tn else float("nan"), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def e1_corrupt():
    global RNG
    RNG = np.random.default_rng(0)
    items = json.load(open("data/corruptions_test.json"))
    lab = [i for i in items if i["label"] is not None]
    ev = {sid: retrieve(SN[sid], "", "full") for sid in SN}
    y = [i["label"] == "unsupported" for i in lab]; grp = [i["snapshot_id"] for i in lab]
    res = {"n_labelled": len(lab), "n_pos": int(sum(y)), "rules": {}, "models": {}, "per_category": {}}
    flags = {}
    t0 = time.time()
    for k, f in CHECKERS.items():
        fl = [f(i["answer"], ev[i["snapshot_id"]])["verdict"] not in (DELIVER if k != "v0" else {"supported"}) for i in lab]
        flags[k] = np.array(fl); res["rules"][k] = detect(y, fl)
        res["rules"][k]["f1_ci"] = boot_f1(y, fl, grp)
    res["checker_latency_us_per_answer"] = (time.time() - t0) / (3 * len(lab)) * 1e6
    for judge in ["qwen1.5b", "llama1b", "qwen7b"]:
        J = load_judge(judge, "corrupt")
        if len(J) < len(items):
            continue
        s = np.array([J[i["iid"]] for i in lab])
        d = detect(y, s >= 0.5); d["auroc"] = float(roc_auc_score(y, s)); d["f1_ci"] = boot_f1(y, s >= 0.5, grp)
        # best-threshold F1 (oracle, optimistic for the judge)
        ths = np.unique(s); best = max(detect(y, s >= t)["f1"] for t in ths)
        d["f1_oracle_threshold"] = float(best)
        res["models"][judge] = d; flags[judge] = s >= 0.5
    cats = sorted({i["category"] for i in items})
    for c in cats:
        its = [i for i in items if i["category"] == c]
        row = {}
        for k, f in CHECKERS.items():
            row[k] = float(np.mean([f(i["answer"], ev[i["snapshot_id"]])["verdict"] not in (DELIVER if k != "v0" else {"supported"}) for i in its]))
        for judge in res["models"]:
            J = load_judge(judge, "corrupt")
            row[judge] = float(np.mean([J[i["iid"]] >= 0.5 for i in its]))
        res["per_category"][c] = {"label": its[0]["label"], "flag_rate": row, "n": len(its)}
    tests = {}
    for other in [k for k in flags if k not in ("v1s", "v2")]:
        corr_a = flags["v1s"] == np.array(y); corr_b = flags[other] == np.array(y)
        tests[f"v1s_vs_{other}"] = mcnemar(corr_a, corr_b)
    adj = holm([t["p"] for t in tests.values()])
    for (k, t), a in zip(tests.items(), adj):
        t["p_holm"] = a
    res["mcnemar_correctness_v1s_vs"] = tests
    json.dump(res, open(f"{RESULTS}/e1_corrupt.json", "w"), indent=1)
    return res


def boot_f1(y, fl, grp):
    y, fl, grp = np.asarray(y, bool), np.asarray(fl, bool), np.asarray(grp)
    ug = np.unique(grp); idx = {g: np.where(grp == g)[0] for g in ug}; out = []
    for _ in range(B):
        ii = np.concatenate([idx[g] for g in RNG.choice(ug, size=len(ug))])
        out.append(detect(y[ii], fl[ii])["f1"])
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["e1", "e2"]
    if "e1" in what:
        r = e1_corrupt(); print(json.dumps({k: v for k, v in r.items() if k != "per_category"}, indent=1)[:4000])
        for c, v in r["per_category"].items():
            print(f"{c:18s} {v['label']!s:12s}", {k: round(x, 2) for k, x in v["flag_rate"].items()})
    if "e2" in what:
        r = e2_endtoend()
        for tag, v in r.items():
            print(tag, "acc", round(v["acc"], 3), v["acc_by_seed"], "gold_in_E", round(v["gold_in_evidence"], 2))
            for g, s in v["gates"].items():
                print("   ", g, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in s.items() if k in ("coverage", "delivered_acc", "delivered_err", "errors_delivered_per_100", "correct_withheld_per_100")})
