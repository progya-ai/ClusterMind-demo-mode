"""Generate LaTeX tables and number macros from results/*.json (no hand-typed numbers)."""
import os as _os
RUNS = _os.environ.get("CM_RUNS_DIR", "runs")
RESULTS = _os.environ.get("CM_RESULTS_DIR", "results")
TABLES = _os.environ.get("CM_TABLES_DIR", "results/tables")
MACROS = _os.path.join(_os.path.dirname(TABLES), "macros.tex")
import json, os
os.makedirs(TABLES, exist_ok=True)
E2 = json.load(open(f"{RESULTS}/e2_endtoend.json"))
E1 = json.load(open(f"{RESULTS}/e1_corrupt.json"))
M = json.load(open(f"{RESULTS}/more.json"))
G = json.load(open(f"{RESULTS}/guards_trace.json"))
D = json.load(open(f"{RESULTS}/idle_drift.json"))
mac = {}


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def ci(c, d=1):
    return f"[{100 * c[0]:.{d}f}, {100 * c[1]:.{d}f}]"


# ------------------------------------------------------------------ Table: end-to-end gating
rows = [("qwen1.5b_full_contract", "Qwen2.5-1.5B$^\\dagger$", "3"), ("llama1b_full_contract", "Llama-3.2-1B", "3"),
        ("qwen7b_full_contract", "Qwen2.5-7B", "1")]
gname = {"none": "none", "v0": "v0 (prototype)", "v1": "v1", "v1s": "v1s", "v2": "v2 (question-aware)"}
L = [r"\begin{tabular}{llrrrr}", r"\toprule",
     r"Model (seeds) & Gate & Coverage & Delivered acc.\ [95\% CI] & Err.\ delivered & Correct withheld \\",
     r"\midrule"]
for tag, name, ns in rows:
    v = E2[tag]
    for k, g in enumerate(["none", "v0", "v1s", "v2"]):
        s = v["gates"][g]
        first = f"{name} ({ns})" if k == 0 else ""
        L.append(f"{first} & {gname[g]} & {pct(s['coverage'])} & {pct(s['delivered_acc'])} {ci(s['delivered_acc_ci'])} & "
                 f"{s['errors_delivered_per_100']:.1f} & {s['correct_withheld_per_100']:.1f} \\\\")
    L.append(r"\midrule" if tag != rows[-1][0] else r"\bottomrule")
L.append(r"\end{tabular}")
open(f"{TABLES}/endtoend.tex", "w").write("\n".join(L))

q, l, b = E2["qwen1.5b_full_contract"], E2["llama1b_full_contract"], E2["qwen7b_full_contract"]
mac.update({
    "RqwenAcc": pct(q["acc"]) + r"\%", "RllamaAcc": pct(l["acc"]) + r"\%", "RbigAcc": pct(b["acc"]) + r"\%",
    "RllamaVtwoAcc": pct(l["gates"]["v2"]["delivered_acc"]) + r"\%", "RllamaVtwoCov": pct(l["gates"]["v2"]["coverage"]) + r"\%",
    "RqwenVtwoAcc": pct(q["gates"]["v2"]["delivered_acc"]) + r"\%", "RqwenVtwoCov": pct(q["gates"]["v2"]["coverage"]) + r"\%",
    "RbigVtwoAcc": pct(b["gates"]["v2"]["delivered_acc"]) + r"\%", "RbigVtwoCov": pct(b["gates"]["v2"]["coverage"]) + r"\%",
    "RllamaVoneSAcc": pct(l["gates"]["v1s"]["delivered_acc"]) + r"\%", "RllamaVoneSCov": pct(l["gates"]["v1s"]["coverage"]) + r"\%",
    "RllamaVzeroAcc": pct(l["gates"]["v0"]["delivered_acc"]) + r"\%", "RllamaVzeroCov": pct(l["gates"]["v0"]["coverage"]) + r"\%",
    "RllamaErrNone": f"{l['gates']['none']['errors_delivered_per_100']:.1f}", "RllamaErrVzero": f"{l['gates']['v0']['errors_delivered_per_100']:.1f}",
    "RllamaErrVoneS": f"{l['gates']['v1s']['errors_delivered_per_100']:.1f}", "RllamaErrVtwo": f"{l['gates']['v2']['errors_delivered_per_100']:.1f}",
    "RllamaWithheldVtwo": f"{l['gates']['v2']['correct_withheld_per_100']:.1f}",
    "RllamaSeedAccs": ", ".join(pct(l["acc_by_seed"][s]) for s in sorted(l["acc_by_seed"])) ,
    "RllamaVtwoCovSeeds": ", ".join(pct(x) for x in l["gates"]["v2"]["coverage_by_seed"]),
    "RqwenLat": f"{q['latency_mean_s']:.1f}", "RqwenLatP": f"{q['latency_p95_s']:.1f}",
    "RllamaLat": f"{l['latency_mean_s']:.1f}", "RllamaLatP": f"{l['latency_p95_s']:.1f}",
    "RbigLat": f"{b['latency_mean_s']:.1f}", "RbigLatP": f"{b['latency_p95_s']:.1f}",
    "RmcnLlamaVzeroA": str(l["mcnemar_error_delivered_v0_vs_v1"]["a_only"]), "RmcnLlamaVzeroB": str(l["mcnemar_error_delivered_v0_vs_v1"]["b_only"]),
    "RmcnLlamaVzeroP": f"{l['mcnemar_error_delivered_v0_vs_v1']['p']:.1e}".replace("e-", r"\times10^{-") + "}",
    "RmcnLlamaVtwoA": str(l["mcnemar_error_delivered_v1s_vs_v2"]["a_only"]),
    "RmcnLlamaVtwoP": f"{l['mcnemar_error_delivered_v1s_vs_v2']['p']:.1e}".replace("e-", r"\times10^{-") + "}",
})

# ------------------------------------------------------------------ Table: per-template (Llama, v2)
tpl_names = {"argmax_util": "highest GPU util.", "value_util": "GPU util.\\ of node", "value_mem": "GPU memory of node",
             "idle_set": "idle nodes (set)", "value_failed": "failures on node", "busy_yesno": "is node busy?",
             "trap_missing": "missing node", "argmax_failed": "most failures"}
L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
     r" & \multicolumn{3}{c}{Llama-3.2-1B (test)} & \multicolumn{3}{c}{Qwen2.5-1.5B (dev)} \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
     r"Question type & Acc. & v2 cov. & v2 del.\ acc. & Acc. & v2 cov. & v2 del.\ acc. \\", r"\midrule"]
for t, nm in tpl_names.items():
    a, c = l["per_template_gate_v1s"][t], q["per_template_gate_v1s"][t]
    f = lambda x: "--" if x != x else pct(x, 0)
    L.append(f"{nm} & {f(a['acc'])} & {f(a['coverage_v2'])} & {f(a['delivered_acc_v2'])} & {f(c['acc'])} & {f(c['coverage_v2'])} & {f(c['delivered_acc_v2'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{TABLES}/pertemplate.tex", "w").write("\n".join(L))

# ------------------------------------------------------------------ Table: detection
A = M.get("annotation", {})
det_rows = [("v0 (prototype)", "v0", "v0"), ("v1", "v1", "v1"), ("v1s", "v1s", "v1s"), ("v2", "v2", "v2")]
L = [r"\begin{tabular}{lcccccccc}", r"\toprule",
     r" & \multicolumn{5}{c}{Controlled set ($n{=}560$, labels by construction)} & \multicolumn{3}{c}{Llama outputs ($n{=}100$, manual)} \\",
     r"\cmidrule(lr){2-6}\cmidrule(lr){7-9}",
     r"Verifier & P & R & F1 & FPR & AUROC & P & R & F1 \\", r"\midrule"]
for nm, k1, k2 in det_rows:
    d = E1["rules"][k1]; a = A.get(k2)
    tail = f"{d['precision']:.2f} & {d['recall']:.2f} & {d['f1']:.2f} & {d['fpr']:.2f} & -- & "
    tail += f"{a['precision']:.2f} & {a['recall']:.2f} & {a['f1']:.2f}" if a else "-- & -- & --"
    L.append(f"{nm} & {tail} \\\\")
L.append(r"\midrule")
for nm, j, ann in [("Self-check, Qwen2.5-1.5B", "qwen1.5b", None), ("Self-check, Llama-3.2-1B", "llama1b", "llama1b"), ("LLM judge, Qwen2.5-7B", "qwen7b", "qwen7b")]:
    d = E1["models"].get(j); a = A.get(ann) if ann else None
    if d is None:
        continue
    tail = f"{d['precision']:.2f} & {d['recall']:.2f} & {d['f1']:.2f} & {d['fpr']:.2f} & {d['auroc']:.2f} & "
    tail += f"{a['precision']:.2f} & {a['recall']:.2f} & {a['f1']:.2f}" if a else "-- & -- & --"
    L.append(f"{nm} & {tail} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{TABLES}/detect.tex", "w").write("\n".join(L))
mac.update({"RannotVtwoF": f"{A['v2']['f1']:.2f}", "RannotVzeroF": f"{A['v0']['f1']:.2f}", "RannotVoneSF": f"{A['v1s']['f1']:.2f}",
            "RannotVzeroR": f"{A['v0']['recall']:.2f}", "RannotVtwoFPR": f"{A['v2']['fpr']:.2f}",
            "RannotN": str(A["n"]), "RannotUns": str(A["n_unsupported"]),
            "RcorVzeroF": f"{E1['rules']['v0']['f1']:.2f}", "RcorVoneSF": f"{E1['rules']['v1s']['f1']:.2f}",
            "RcorVzeroFPR": f"{E1['rules']['v0']['fpr']:.2f}",
            "RselfQwenAUC": f"{E1['models']['qwen1.5b']['auroc']:.2f}", "RselfLlamaAUC": f"{E1['models']['llama1b']['auroc']:.2f}"})
if "qwen7b" in E1["models"]:
    mac.update({"RjudgeCorAUC": f"{E1['models']['qwen7b']['auroc']:.2f}", "RjudgeCorF": f"{E1['models']['qwen7b']['f1']:.2f}"})
if "qwen7b" in A:
    mac.update({"RjudgeAnnF": f"{A['qwen7b']['f1']:.2f}", "RjudgeAnnAUC": f"{A['qwen7b']['auroc']:.2f}"})

# per-category flag rates
cats = ["correct_single", "correct_multi", "state_correct", "absence_correct", "value_small", "value_large", "metric_swap",
        "node_swap", "fabricated_node", "extra_fabricated", "state_false", "absence_false", "qualitative_false", "no_claim"]
cols = ["v0", "v1", "v1s", "v2"] + [j for j in ["qwen1.5b", "llama1b", "qwen7b"] if j in E1["models"]]
hdr = {"v0": "v0", "v1": "v1", "v1s": "v1s", "v2": "v2", "qwen1.5b": "self-Q", "llama1b": "self-L", "qwen7b": "judge-7B"}
L = [r"\begin{tabular}{ll" + "r" * len(cols) + "}", r"\toprule",
     "Category & Label & " + " & ".join(hdr[c] for c in cols) + r" \\", r"\midrule"]
for c in cats:
    v = E1["per_category"][c]
    lab = {"supported": "sup.", "unsupported": "unsup.", None: "n/a"}[v["label"]]
    L.append(c.replace("_", r"\_") + f" & {lab} & " + " & ".join(f"{v['flag_rate'][k]:.2f}" for k in cols) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{TABLES}/categories.tex", "w").write("\n".join(L))

# ------------------------------------------------------------------ Table: retrieval / prompt
L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
     r"Model & Evidence / prompt & Gold in $E$ & Acc. & v0 del.\ acc. & v2 cov. & v2 del.\ acc. \\", r"\midrule"]
for tag, name, ev in [("qwen1.5b_tfidf_original", "Qwen2.5-1.5B", "TF-IDF top-3 / original (as built)"),
                      ("qwen1.5b_tfidf_contract", "Qwen2.5-1.5B", "TF-IDF top-3 / contract"),
                      ("qwen1.5b_full_original", "Qwen2.5-1.5B", "full snapshot / original"),
                      ("qwen1.5b_full_contract", "Qwen2.5-1.5B$^\\dagger$", "full snapshot / contract"),
                      ("llama1b_tfidf_contract", "Llama-3.2-1B", "TF-IDF top-3 / contract"),
                      ("llama1b_full_contract", "Llama-3.2-1B", "full snapshot / contract")]:
    v = E2[tag]
    L.append(f"{name} & {ev} & {pct(v['gold_in_evidence'],0)} & {pct(v['acc'])} & {pct(v['gates']['v0']['delivered_acc'])} & "
             f"{pct(v['gates']['v2']['coverage'])} & {pct(v['gates']['v2']['delivered_acc'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{TABLES}/retrieval.tex", "w").write("\n".join(L))
mac.update({"RtfidfGold": pct(E2["qwen1.5b_tfidf_contract"]["gold_in_evidence"], 0) + r"\%",
            "RtfidfQwenVtwoAcc": pct(E2["qwen1.5b_tfidf_contract"]["gates"]["v2"]["delivered_acc"]) + r"\%",
            "RtfidfLlamaVtwoAcc": pct(E2["llama1b_tfidf_contract"]["gates"]["v2"]["delivered_acc"]) + r"\%",
            "RasbuiltAcc": pct(E2["qwen1.5b_tfidf_original"]["acc"]) + r"\%",
            "RasbuiltVzeroAcc": pct(E2["qwen1.5b_tfidf_original"]["gates"]["v0"]["delivered_acc"]) + r"\%",
            "RasbuiltVzeroCov": pct(E2["qwen1.5b_tfidf_original"]["gates"]["v0"]["coverage"]) + r"\%"})

# ------------------------------------------------------------------ Table: guards
L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
     r"Guard (predicate @ time) & $\Delta{=}0$ & 60\,s & 300\,s & 900\,s & 1800\,s \\", r"\midrule",
     r"\multicolumn{6}{l}{\emph{Unsafe executions, \% of shutdown requests on nodes busy at execution}} \\"]
gl = {"none": "none (whitelist only)", "as_built": "prototype (util, advisory)", "util@proposal": "util $\\le$10\\% @ proposal",
      "util@confirm": "util $\\le$10\\% @ confirmation", "sched@proposal": "running$=$0 @ proposal", "sched@confirm": "running$=$0 @ confirmation (ours)"}
ds = ["0", "60", "300", "900", "1800"]
for g, nm in gl.items():
    L.append(nm + " & " + " & ".join(pct(G[d][g]["unsafe_rate_among_busy"]) for d in ds) + r" \\")
L.append(r"\midrule")
L.append(r"\multicolumn{6}{l}{\emph{False blocks, \% of requests on nodes idle at execution}} \\")
for g in ["util@proposal", "sched@proposal", "sched@confirm"]:
    L.append(gl[g] + " & " + " & ".join(pct(G[d][g]["false_block_rate_among_idle"]) for d in ds) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{TABLES}/guards.tex", "w").write("\n".join(L))
g0 = G["0"]
mac.update({"RbusyLowUtil": pct(g0["busy_lowutil_at_exec"] / g0["busy_at_exec"]) + r"\%",
            "RguardN": f"{g0['n']:,}".replace(",", "{,}"),
            "RdriftFive": pct(D["300"]["drift_rate"]) + r"\%", "RdriftFiveCI": ci(D["300"]["ci"]),
            "RdriftFifteen": pct(D["900"]["drift_rate"]) + r"\%", "RdriftFifteenCI": ci(D["900"]["ci"]),
            "RdriftThirty": pct(D["1800"]["drift_rate"]) + r"\%", "RdriftThirtyCI": ci(D["1800"]["ci"]),
            "RdriftHour": pct(D["3600"]["drift_rate"]) + r"\%", "RdriftHourCI": ci(D["3600"]["ci"]),
            "RdriftN": f"{D['300']['n']:,}".replace(",", "{,}"),
            "RdriftMissedUtil": pct(D["1800"]["drift_missed_by_util"] / D["1800"]["n_drift"], 0) + r"\%"})

# ------------------------------------------------------------------ Table: LLM action proposals
AL = M["actions_llm"]
kinds = [("explicit_idle", "``Shut down Node-$k$'' ($k$ idle)"), ("explicit_busy", "``Shut down Node-$k$'' ($k$ busy)"),
         ("false_premise", "``Node-$k$ has been idle for hours, shut it down'' ($k$ busy)"),
         ("implicit_idle", "``Shut down the idle node\\ldots''"), ("missing_node", "``Shut down Node-$j$'' ($j$ absent)")]
L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
     r" & \multicolumn{3}{c}{Qwen2.5-1.5B} & \multicolumn{3}{c}{Llama-3.2-1B} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
     r"Instruction & Proposed & Unsafe (util) & Unsafe (ours) & Proposed & Unsafe (util) & Unsafe (ours) \\", r"\midrule"]
for k, nm in kinds:
    a, c = AL["qwen1.5b"][k], AL["llama1b"][k]
    L.append(f"{nm} & {pct(a['proposed_disruptive'],0)} & {pct(a['unsafe_util@confirm'],0)} & {pct(a['unsafe_sched@confirm'],0)} & "
             f"{pct(c['proposed_disruptive'],0)} & {pct(c['unsafe_util@confirm'],0)} & {pct(c['unsafe_sched@confirm'],0)} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{TABLES}/actions.tex", "w").write("\n".join(L))
mac.update({"RqwenBusyComply": pct(AL["qwen1.5b"]["explicit_busy"]["proposed_disruptive"], 0) + r"\%",
            "RqwenPremiseComply": pct(AL["qwen1.5b"]["false_premise"]["proposed_disruptive"], 0) + r"\%",
            "RllamaPremiseComply": pct(AL["llama1b"]["false_premise"]["proposed_disruptive"], 0) + r"\%",
            "RllamaParse": pct(AL["llama1b"]["explicit_busy"]["parse_rate"], 0) + r"\%"})

# ------------------------------------------------------------------ ablations
CA = M["checker_ablations"]; SE = M["shared_evidence"]
L = [r"\begin{tabular}{lrrrr}", r"\toprule",
     r"Variant (on v1s) & Controlled F1 & Controlled FPR & Real cov. & Real del.\ acc. \\", r"\midrule"]
for k, nm in [("tol_x1.0", "v1s (default $\\epsilon$)"), ("tol_x0.0", "$\\epsilon=0$ (rounding rule only)"), ("tol_x5.0", "$\\epsilon\\times5$"),
              ("strict_unbound", "unbound numbers rejected"), ("no_masking", "no masking of node ids/timestamps"),
              ("deliver_unverifiable", "deliver \\textsc{unverifiable} answers")]:
    v = CA[k]; c = v.get("corrupt")
    L.append(f"{nm} & {c['f1']:.2f} & {c['fpr']:.2f} & {pct(v['real_coverage'])} & {pct(v['real_delivered_acc'])} \\\\" if c else
             f"{nm} & -- & -- & {pct(v['real_coverage'])} & {pct(v['real_delivered_acc'])} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{TABLES}/ablations.tex", "w").write("\n".join(L))
L = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Checker reads live state after & 60\,s & 300\,s & 900\,s & 1800\,s \\", r"\midrule",
     "node records changed & " + " & ".join(pct(SE[d]["frac_node_records_changed"]) for d in ["60", "300", "900", "1800"]) + r" \\",
     "verdict changes & " + " & ".join(pct(SE[d]["verdict_flip_rate"]) for d in ["60", "300", "900", "1800"]) + r" \\",
     r"\quad of which: grounded $\to$ flagged & " + " & ".join(pct(SE[d]["grounded_answer_flagged_by_live"]) for d in ["60", "300", "900", "1800"]) + r" \\",
     r"\bottomrule", r"\end{tabular}"]
open(f"{TABLES}/shared.tex", "w").write("\n".join(L))
mac.update({"RflipFive": pct(SE["300"]["verdict_flip_rate"]) + r"\%", "RflipThirty": pct(SE["1800"]["verdict_flip_rate"]) + r"\%",
            "RnomaskFPR": f"{CA['no_masking']['corrupt']['fpr']:.2f}", "RnomaskCov": pct(CA["no_masking"]["real_coverage"]) + r"\%"})
if "checker_latency_us" in M:
    mac["RcheckUs"] = f"{M['checker_latency_us']['v2']:.0f}"

# ------------------------------------------------------------------ judge gating
JG = M.get("judge_gating", {})
if JG:
    for tag, short in [("qwen1.5b_full_contract", "Qwen"), ("llama1b_full_contract", "Llama")]:
        e = JG.get(tag, {})
        if "qwen7b" in e:
            j = e["qwen7b"]
            mac[f"RjudgeGate{short}Acc"] = pct(j["delivered_acc@matched"]) + r"\%"
            mac[f"RjudgeGate{short}AccCI"] = ci(j["delivered_acc@matched_ci"])
            mac[f"RjudgeGate{short}Lat"] = f"{j['latency_mean_s']:.1f}"
            mac[f"RruleGate{short}Acc"] = pct(e["rule_v2"]["delivered_acc"]) + r"\%"
            mac[f"RjudgeGate{short}AUC"] = f"{j['auroc_vs_incorrect']:.2f}"

with open(MACROS, "w") as f:
    for k, v in sorted(mac.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(mac), "macros")
