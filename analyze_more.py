"""Ablations, robustness, judge-gating comparison, action results."""
import os as _os
RUNS = _os.environ.get("CM_RUNS_DIR", "runs")
RESULTS = _os.environ.get("CM_RESULTS_DIR", "results")
import json, os, collections, re, time
import numpy as np
from sklearn.metrics import roc_auc_score
import cmbench.checkers as C
import analyze as _A
from cmbench.evidence import retrieve, Evidence
from cmbench.score import score
from cmbench.guard import parse_action, resolve, decide, GUARDS
from analyze import SN, Q, boot, boot_ratio, mcnemar, holm, load_gen, annotate, load_judge, detect, DELIVER

OUT = {}


def ev_full(sid):
    return retrieve(SN[sid], "", "full")


# ---------------------------------------------------------------- tolerance / binding ablations
def ablate_checker():
    _A.RNG = np.random.default_rng(0)  # deterministic bootstrap regardless of which analyses run
    items = json.load(open("data/corruptions_test.json"))
    lab = [i for i in items if i["label"] is not None]
    y = np.array([i["label"] == "unsupported" for i in lab])
    real = annotate(load_gen("qwen1.5b_full_contract") + load_gen("llama1b_full_contract"))
    res = {}
    base = dict(C.DEFAULT_TOL)
    for mult in [0.0, 0.5, 1.0, 2.0, 5.0]:
        tol = {k: v * mult for k, v in base.items()}
        if mult == 0:
            tol = {k: 0.0 for k in base}
        fl = np.array([C.check_v1(i["answer"], ev_full(i["snapshot_id"]), tol=tol, state_claims=True)["verdict"] not in DELIVER for i in lab])
        vs = [C.check_v1(r["answer"], ev_full(Q[r["qid"]]["snapshot_id"]), tol=tol, state_claims=True)["verdict"] for r in real]
        dl = np.array([v in DELIVER for v in vs]); corr = np.array([r["correct"] for r in real])
        res[f"tol_x{mult}"] = {"corrupt": detect(y, fl), "real_coverage": float(dl.mean()), "real_delivered_acc": float(corr[dl].mean())}
    # strict binding of unbound numbers
    fl = np.array([C.check_v1(i["answer"], ev_full(i["snapshot_id"]), state_claims=True, strict_unbound=True)["verdict"] not in DELIVER for i in lab])
    vs = [C.check_v1(r["answer"], ev_full(Q[r["qid"]]["snapshot_id"]), state_claims=True, strict_unbound=True)["verdict"] for r in real]
    dl = np.array([v in DELIVER for v in vs]); corr = np.array([r["correct"] for r in real])
    res["strict_unbound"] = {"corrupt": detect(y, fl), "real_coverage": float(dl.mean()), "real_delivered_acc": float(corr[dl].mean())}
    # no masking of node ids / timestamps (monkeypatch)
    orig_mask = C._masked
    C._masked = lambda text, spans: text
    fl = np.array([C.check_v1s(i["answer"], ev_full(i["snapshot_id"]))["verdict"] not in DELIVER for i in lab])
    vs = [C.check_v1s(r["answer"], ev_full(Q[r["qid"]]["snapshot_id"]))["verdict"] for r in real]
    C._masked = orig_mask
    dl = np.array([v in DELIVER for v in vs])
    res["no_masking"] = {"corrupt": detect(y, fl), "real_coverage": float(dl.mean()), "real_delivered_acc": float(corr[dl].mean())}
    # unverifiable treated as deliverable (i.e. drop the 'no claim => not grounded' rule)
    vs = [r["v1s"] for r in real]
    dl = np.array([v in DELIVER | {"unverifiable"} for v in vs])
    res["deliver_unverifiable"] = {"real_coverage": float(dl.mean()), "real_delivered_acc": float(corr[dl].mean())}
    OUT["checker_ablations"] = res


# ---------------------------------------------------------------- H2: frozen snapshot vs re-reading live state
def ablate_shared_evidence():
    _A.RNG = np.random.default_rng(0)  # deterministic bootstrap regardless of which analyses run
    from cmbench.snapshots import load_tables, machine_state
    I = load_tables()
    real = annotate(load_gen("qwen1.5b_full_contract"))
    real = [r for r in real if r["seed"] == 0] + [r for r in annotate(load_gen("llama1b_full_contract")) if r["seed"] == 0]
    res = {}
    for d in [60, 300, 900, 1800]:
        live = {}
        for sid, s in SN.items():
            ms = [n["machine"] for n in s["nodes"]]
            st = machine_state(I, s["t"] + d, ms)
            recs = []
            for n in s["nodes"]:
                r = st.loc[n["machine"]]
                recs.append({"node": n["node"], "machine": n["machine"], "running_instances": int(r.running_instances),
                             "gpu_util_pct": float(r.gpu_util_pct), "gpu_mem_gb": float(r.gpu_mem_gb),
                             "failed_last_hour": int(r.failed_last_hour)})
            live[sid] = Evidence(recs, s["t"] + d)
        changed_nodes = np.mean([any(a[k] != b[k] for k in ["running_instances", "gpu_util_pct", "gpu_mem_gb", "failed_last_hour"])
                                 for sid in SN for a, b in zip(SN[sid]["nodes"], live[sid].records)])
        flips = []; sup_to_uns = []; uns_to_sup = []
        for r in real:
            v_frozen = r["v1s"]
            v_live = C.check_v1s(r["answer"], live[Q[r["qid"]]["snapshot_id"]])["verdict"]
            flips.append(v_frozen != v_live)
            sup_to_uns.append(v_frozen in DELIVER and v_live not in DELIVER)
            uns_to_sup.append(v_frozen not in DELIVER and v_live in DELIVER)
        res[d] = {"frac_node_records_changed": float(changed_nodes), "verdict_flip_rate": float(np.mean(flips)),
                  "grounded_answer_flagged_by_live": float(np.mean(sup_to_uns)),
                  "flagged_answer_passed_by_live": float(np.mean(uns_to_sup)),
                  "flip_ci": boot(flips, [r["_snap"] for r in real])}
    OUT["shared_evidence"] = res


# ---------------------------------------------------------------- judge-gated vs rule-gated on real outputs
def judge_gating():
    _A.RNG = np.random.default_rng(0)  # deterministic bootstrap regardless of which analyses run
    res = {}
    for tag, judges in [("qwen1.5b_full_contract", ["qwen1.5b", "qwen7b"]), ("llama1b_full_contract", ["llama1b", "qwen7b"])]:
        rows = [r for r in annotate(load_gen(tag)) if r["seed"] == 0]
        corr = np.array([r["correct"] for r in rows]); grp = [r["_snap"] for r in rows]
        dl_rule = np.array([r["v2"] in DELIVER for r in rows])
        cov_rule = dl_rule.mean()
        entry = {"n": len(rows), "rule_v2": {"coverage": float(cov_rule), "delivered_acc": float(corr[dl_rule].mean())}}
        for j in judges:
            J = load_judge(j, f"gen_{tag}_0")
            if len(J) < len(rows):
                continue
            s = np.array([J[f"{r['qid']}|0"] for r in rows])
            dl = s < 0.5
            # threshold matched to the rule's coverage
            th = np.quantile(s, cov_rule)
            dlm = s <= th
            entry[j] = {"auroc_vs_incorrect": float(roc_auc_score(~corr, s)) if 0 < corr.mean() < 1 else None,
                        "coverage@0.5": float(dl.mean()), "delivered_acc@0.5": float(corr[dl].mean()) if dl.any() else None,
                        "coverage@matched": float(dlm.mean()), "delivered_acc@matched": float(corr[dlm].mean()),
                        "delivered_acc@matched_ci": boot_ratio(corr & dlm, dlm, grp),
                        "mcnemar_errors_delivered_rule_vs_judge@matched": mcnemar((~corr) & dl_rule, (~corr) & dlm),
                        "latency_mean_s": float(np.mean([json.loads(l)["latency_s"] for l in open(f"{RUNS}/judge_{j}_gen_{tag}_0.jsonl")]))}
        entry["rule_v2"]["delivered_acc_ci"] = boot_ratio(corr & dl_rule, dl_rule, grp)
        res[tag] = entry
    OUT["judge_gating"] = res


# ---------------------------------------------------------------- actions (LLM proposals)
def actions_llm():
    _A.RNG = np.random.default_rng(0)  # deterministic bootstrap regardless of which analyses run
    res = {}
    for model in ["qwen1.5b", "llama1b"]:
        p = f"{RUNS}/actions_{model}.jsonl"
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p)]
        by = collections.defaultdict(list)
        for r in rows:
            s = SN[r["snapshot_id"]]; inv = {n["node"]: n for n in s["nodes"]}
            cmd = parse_action(r["raw"])
            res_cmd, why = resolve(cmd, r["instruction"], inv)
            prop = res_cmd is not None and res_cmd["action"] in ("shutdown_node", "migrate_job")
            target = res_cmd["node"] if res_cmd else None
            busy = inv[target]["running_instances"] > 0 if target else None
            outcome = {"parsed": cmd is not None, "proposed_disruptive": prop, "resolve": why, "target": target, "target_busy": busy,
                       "model_node": (cmd or {}).get("node"), "action": (cmd or {}).get("action")}
            for g in GUARDS:
                ex = prop and decide(g, "shutdown_node", inv[target], inv[target])
                outcome[f"exec_{g}"] = bool(ex)
                outcome[f"unsafe_{g}"] = bool(ex and busy)
            by[r["kind"]].append(outcome)
        summ = {}
        for k, v in by.items():
            summ[k] = {"n": len(v), "parse_rate": float(np.mean([o["parsed"] for o in v])),
                       "proposed_disruptive": float(np.mean([o["proposed_disruptive"] for o in v])),
                       "target_busy_given_proposed": float(np.mean([o["target_busy"] for o in v if o["proposed_disruptive"]])) if any(o["proposed_disruptive"] for o in v) else None,
                       "resolve_reasons": dict(collections.Counter(o["resolve"] for o in v)),
                       **{f"unsafe_{g}": float(np.mean([o[f"unsafe_{g}"] for o in v])) for g in GUARDS}}
        res[model] = summ
    OUT["actions_llm"] = res


if __name__ == "__main__" and "annot" not in __import__("sys").argv:
    import sys
    what = sys.argv[1:] or ["ablate", "shared", "judge", "actions"]
    prev = json.load(open(f"{RESULTS}/more.json")) if os.path.exists(f"{RESULTS}/more.json") else {}
    if "ablate" in what: ablate_checker()
    if "shared" in what: ablate_shared_evidence()
    if "judge" in what: judge_gating()
    if "actions" in what: actions_llm()
    prev.update(OUT)
    json.dump(prev, open(f"{RESULTS}/more.json", "w"), indent=1, default=str)
    print(json.dumps(OUT, indent=1, default=str)[:6000])


def annot_eval():
    _A.RNG = np.random.default_rng(0)  # deterministic bootstrap regardless of which analyses run
    A = json.load(open("data/annotation_llama1b_seed0.json"))["items"]
    lab = {a["qid"]: a["label"] for a in A}
    rows = [r for r in annotate(load_gen("llama1b_full_contract")) if r["seed"] == 0 and r["qid"] in lab]
    y = np.array([lab[r["qid"]] == "unsupported" for r in rows])
    res = {"n": len(rows), "n_unsupported": int(y.sum())}
    for g in ["v0", "v1", "v1s", "v2"]:
        fl = np.array([(r[g] != "supported") if g == "v0" else (r[g] not in DELIVER) for r in rows])
        res[g] = detect(y, fl)
    for j in ["llama1b", "qwen7b"]:
        J = load_judge(j, "gen_llama1b_full_contract_0")
        if len(J) < 320:
            continue
        s = np.array([J[f"{r['qid']}|0"] for r in rows])
        d = detect(y, s >= 0.5); d["auroc"] = float(roc_auc_score(y, s))
        d["f1_oracle_threshold"] = float(max(detect(y, s >= t)["f1"] for t in np.unique(s)))
        res[j] = d
    corr = np.array([r["correct"] for r in rows])
    res["p_correct_given_supported_label"] = float(corr[~y].mean())
    res["p_correct_given_unsupported_label"] = float(corr[y].mean())
    OUT["annotation"] = res


def checker_latency():
    import timeit
    items = json.load(open("data/corruptions_test.json"))
    evs = {sid: ev_full(sid) for sid in SN}
    out = {}
    for name, f in [("v0", C.check_v0), ("v1s", C.check_v1s), ("v2", lambda a, e: C.check_v2(a, e, "Which nodes are idle?"))]:
        t = timeit.timeit(lambda: [f(i["answer"], evs[i["snapshot_id"]]) for i in items], number=5)
        out[name] = t / (5 * len(items)) * 1e6
    OUT["checker_latency_us"] = out


if __name__ == "__main__" and "annot" in __import__("sys").argv:
    annot_eval(); checker_latency()
    prev = json.load(open(f"{RESULTS}/more.json")) if os.path.exists(f"{RESULTS}/more.json") else {}
    prev.update(OUT); json.dump(prev, open(f"{RESULTS}/more.json", "w"), indent=1, default=str)
    print(json.dumps(OUT, indent=1, default=str))
