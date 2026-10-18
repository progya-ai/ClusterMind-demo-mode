"""
Deterministic answer checkers (no model call).

v0  : faithful port of the original ClusterMind ground_answer(): an answer is
      'grounded' iff it names (by substring) a node present in the evidence
      AND any number anywhere in the text is within +-1 of any rounded
      metric value of that node.
v1  : the evidence-contract checker. Claim-level:
      every extracted (node, metric, value) claim must be supported by the
      same frozen evidence snapshot E; node identifiers are whole tokens
      and are never treated as numeric claims.
v1s : v1 plus categorical state claims (idle / busy) checked against
      running_instances.
"""
import re

NUM_RE = re.compile(r"(?<![\w.])(\d+(?:,\d{3})*(?:\.\d+)?)")
NODE_RE = re.compile(r"\bnode[\s_-]?(\d+)\b", re.IGNORECASE)
TS_RE = re.compile(r"\bt\s*=\s*\d+(?:\.\d+)?\s*s?\b|\b\d{6,}\s*s\b|\(?\s*100\s*%\s*=\s*one GPU\s*\)?|\bin (?:the )?last (?:1 )?hour\b|\bpast (?:1 )?hour\b", re.IGNORECASE)
SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")

ABSTAIN_RE = re.compile(
    r"(not (?:present|found|listed|available|included|in the (?:telemetry|evidence|data))|"
    r"no (?:data|information|telemetry|record|such node)|does not exist|doesn't exist|"
    r"unknown node|cannot (?:find|determine)|can't (?:find|determine)|unable to (?:find|determine)|"
    r"there is no (?:node|information|data))", re.IGNORECASE)

# metric lexicon: (metric, regex) ; matched on a window around the number
METRIC_LEX = [
    ("failed_last_hour", re.compile(r"fail|crash|error", re.I)),
    ("gpu_mem_gb", re.compile(r"\bgb\b|gib|memory|\bmem\b|vram", re.I)),
    ("gpu_util_pct", re.compile(r"%|percent|utili[sz]ation|util\b|gpu usage|gpu load|busy", re.I)),
    ("running_instances", re.compile(r"instance|job|task|worker|running|process", re.I)),
]
UNIT_AFTER = [
    ("failed_last_hour", re.compile(r"^\s*(?:(?:instances?|jobs?|tasks?|workers?)\s+)?(?:that\s+)?(?:have\s+|had\s+)?(?:fail\w*|crash\w*)", re.I)),
    ("gpu_util_pct", re.compile(r"^\s*(%|percent)", re.I)),
    ("gpu_mem_gb", re.compile(r"^\s*(gb|gib|g\b)", re.I)),
    ("running_instances", re.compile(r"^\s*(running\s+)?(instances?|jobs?|tasks?|workers?)", re.I)),
]
KEY_BEFORE = [
    ("failed_last_hour", re.compile(r"(?:fail\w*|failures?)(?:\s+instances?)?(?:\s+(?:in|over|during))?(?:\s+(?:the\s+)?(?:last|past))?\s*(?:hour)?\s*(?:was|were|is|of|at|:|=)?\s*\(?\s*$", re.I)),
    ("running_instances", re.compile(r"(?:running\s+instances?|instances?\s+running|running)\s*(?:was|were|is|of|at|:|=)?\s*\(?\s*$", re.I)),
    ("gpu_util_pct", re.compile(r"utili[sz]ation(?:\s+(?:of|at|is|was))?\s*:?\s*(?:about|around|roughly|approximately)?\s*\(?\s*$", re.I)),
    ("gpu_mem_gb", re.compile(r"memory(?:\s+(?:usage|use|consumption))?(?:\s+(?:of|at|is|was))?\s*:?\s*(?:about|around|roughly|approximately)?\s*$", re.I)),
]
DEFAULT_TOL = {"gpu_util_pct": 1.0, "gpu_mem_gb": 0.5, "running_instances": 0.0, "failed_last_hour": 0.0}

IDLE_RE = re.compile(r"\bidle\b|not running any|nothing running|no (?:running )?(?:jobs|instances|tasks)|\b0 (?:running )?(?:jobs|instances|tasks)\b(?!\s+(?:that\s+)?(?:have\s+|had\s+)?fail)|running 0\b|unused", re.I)
BUSY_RE = re.compile(r"\bbusy\b|\b(?:is|are) running\b(?!\s+(?:0|no|any)\b)|\bactive\b|has [1-9]\d* (?:running )?(?:jobs|instances)|\b[1-9]\d* (?:running )?(?:jobs|instances) running", re.I)
NEG_RE = re.compile(r"\bnot\b|n't\b|\bno longer\b", re.I)


# --------------------------------------------------------------------- v0
def check_v0(answer, ev):
    recs = ev.records
    if not recs:
        return {"verdict": "unsupported", "reason": "no evidence"}
    low = answer.lower()
    nums = {round(float(n)) for n in re.findall(r"\d+(?:\.\d+)?", answer)}
    named = None
    for r in recs:
        if r["node"].lower() in low:
            named = r
            break
    if not named:
        return {"verdict": "unsupported", "reason": "no evidence node named"}
    sup = {round(named[m]) for m in ("running_instances", "gpu_util_pct", "gpu_mem_gb", "failed_last_hour")}
    if any(abs(a - s) <= 1 for a in nums for s in sup):
        return {"verdict": "supported", "reason": f"matched {named['node']}"}
    return {"verdict": "unsupported", "reason": "no number matches"}


# --------------------------------------------------------------------- v1
def _node_spans(text):
    return [(m.start(), m.end(), f"Node-{int(m.group(1))}") for m in NODE_RE.finditer(text)]


def _masked(text, spans):
    """Replace node identifiers and timestamps by spaces so their digits are never numeric claims."""
    chars = list(text)
    for a, b, _ in spans:
        for i in range(a, b):
            chars[i] = " "
    for m in TS_RE.finditer(text):
        for i in range(m.start(), m.end()):
            chars[i] = " "
    return "".join(chars)


def _bind_metric(sent, masked_sent, a, b):
    before = masked_sent[max(0, a - 60):a]
    for metric, rx in KEY_BEFORE:
        if rx.search(before):
            return metric
    after = masked_sent[b:b + 45]
    for metric, rx in UNIT_AFTER:
        if rx.search(after):
            return metric
    lo, hi = max(0, a - 40), min(len(sent), b + 25)
    window_before = masked_sent[lo:a]
    window_after = masked_sent[b:hi]
    best, best_d = None, 10 ** 9
    for metric, rx in METRIC_LEX:
        for m in rx.finditer(window_before):
            d = len(window_before) - m.end()
            if d < best_d:
                best, best_d = metric, d
        for m in rx.finditer(window_after):
            d = m.start()
            if d < best_d:
                best, best_d = metric, d
    return best


def _close(v, x, metric, tol, text_num):
    t = tol.get(metric, 0.0) if tol is not None else DEFAULT_TOL[metric]
    # accept the evidence value rounded to the precision the answer used
    dec = len(text_num.split(".")[1]) if "." in text_num else 0
    if round(x, dec) == round(v, dec):
        return True
    return abs(v - x) <= t + 1e-9


def extract_claims(answer, state_claims=False):
    """Returns list of claims: dict(node, metric, value, text_num, kind)."""
    claims = []
    sents = [s for s in SENT_SPLIT.split(answer) if s.strip()]
    last_node = None
    for s in sents:
        spans = _node_spans(s)
        ms = _masked(s, spans)
        for m in NUM_RE.finditer(ms):
            a, b = m.start(), m.end()
            raw = m.group(1).replace(",", "")
            # skip list markers such as "1." / "2)" at sentence start
            if ms[:a].strip() == "" and re.match(r"^\s*[.)]", ms[b:b + 2] or "") and "." not in raw:
                continue
            prev = [sp for sp in spans if sp[1] <= a]
            node = prev[-1][2] if prev else (spans[0][2] if spans else last_node)
            metric = _bind_metric(s, ms, a, b)
            claims.append({"node": node, "metric": metric, "value": float(raw), "text_num": raw, "kind": "numeric"})
        if state_claims and spans:
            neg = bool(NEG_RE.search(s))
            idle, busy = bool(IDLE_RE.search(s)), bool(BUSY_RE.search(s))
            if idle and busy and re.search(r"\bno (?:running )?(?:jobs|instances|tasks)\b|\b0 (?:running )?(?:jobs|instances|tasks)\b(?!\s+(?:that\s+)?(?:have\s+|had\s+)?fail)", s, re.I):
                busy = False
            if idle ^ busy:
                st = "idle" if idle else "busy"
                if neg and not re.search(r"not running any|nothing running|no (?:running )?(?:jobs|instances|tasks)", s, re.I):
                    st = "busy" if st == "idle" else "idle"
                for _, _, nd in spans:
                    claims.append({"node": nd, "metric": "state", "value": st, "text_num": None, "kind": "state"})
        if spans:
            last_node = spans[-1][2]
    return claims


def check_v1(answer, ev, tol=None, state_claims=False, strict_unbound=False):
    claims = extract_claims(answer, state_claims=state_claims)
    mentioned = {n for _, _, n in _node_spans(answer)}
    unknown = [n for n in mentioned if n.lower() not in ev.by_node]
    abstains = bool(ABSTAIN_RE.search(answer))
    results = []
    for c in claims:
        rec = ev.by_node.get(c["node"].lower()) if c["node"] else None
        if rec is None:
            ok, why = False, "claim about node not in evidence" if c["node"] else "numeric claim with no node"
        elif c["kind"] == "state":
            truth = "idle" if rec["running_instances"] == 0 else "busy"
            ok, why = (c["value"] == truth), f"state {c['value']} vs {truth}"
        elif c["metric"] is None:
            if strict_unbound:
                ok, why = False, "number not bound to a metric"
            else:
                ok = any(_close(c["value"], rec[m], m, tol, c["text_num"]) for m in DEFAULT_TOL)
                why = "unbound number, matched any metric" if ok else "unbound number matches no metric"
        else:
            ok = _close(c["value"], rec[c["metric"]], c["metric"], tol, c["text_num"])
            why = f"{c['metric']}={rec[c['metric']]} vs {c['value']}"
        results.append({**c, "supported": ok, "why": why})
    # an abstention that declares an EXISTING evidence node absent is itself a false claim
    for sent in SENT_SPLIT.split(answer):
        m = ABSTAIN_RE.search(sent)
        if m:
            for a, b, nd in _node_spans(sent[:m.start()]):
                if nd.lower() in ev.by_node:
                    results.append({"node": nd, "metric": "presence", "value": "absent", "text_num": None,
                                    "kind": "presence", "supported": False, "why": "existing node declared absent"})
    n_bad = sum(not r["supported"] for r in results)
    if unknown and not abstains:
        verdict = "unsupported"
    elif results and n_bad == 0:
        verdict = "supported"
    elif not results and abstains:
        verdict = "abstain"
    elif not results:
        verdict = "unverifiable"
    else:
        verdict = "unsupported"
    return {"verdict": verdict, "claims": results, "unknown_nodes": unknown, "n_claims": len(results), "n_unsupported": n_bad}


def check_v1s(answer, ev, tol=None):
    return check_v1(answer, ev, tol=tol, state_claims=True)


# --------------------------------------------------------------------- v2 (question-aware extension)
SUPER_RE = re.compile(r"\b(most|highest|busiest|maximum|largest|greatest|top|concentrated|heaviest)\b", re.I)
LEAST_RE = re.compile(r"\b(least|lowest|minimum|smallest|fewest)\b", re.I)
SUPER_METRIC = [
    ("failed_last_hour", re.compile(r"fail", re.I)),
    ("gpu_mem_gb", re.compile(r"memory|\bgb\b", re.I)),
    ("running_instances", re.compile(r"instances|jobs|tasks", re.I)),
    ("gpu_util_pct", re.compile(r"gpu|utili[sz]ation|usage|busiest|busy", re.I)),
]
SET_Q_RE = re.compile(r"\bwhich nodes\b|\blist (?:every|all)\b|\bany nodes\b|\bnodes? with nothing\b", re.I)
IDLE_Q_RE = re.compile(r"idle|nothing running|not running", re.I)


def _superlative_claims(answer, ev):
    out = []
    for s in [x for x in SENT_SPLIT.split(answer) if x.strip()]:
        spans = _node_spans(s)
        if not spans:
            continue
        sup, least = SUPER_RE.search(s), LEAST_RE.search(s)
        if not (sup or least):
            continue
        pos = (sup or least).start()
        cands = [(abs(mm.start() - pos), m) for m, rx in SUPER_METRIC for mm in rx.finditer(s)]
        if not cands:
            continue
        metric = min(cands)[1]
        node = spans[0][2]
        vals = [r[metric] for r in ev.records]
        target = max(vals) if sup else min(vals)
        winners = {r["node"] for r in ev.records if r[metric] == target}
        ok = node in winners and not (sup and target == 0)  # "most failures" is false when nothing failed
        out.append({"node": node, "metric": metric, "value": "max" if sup else "min", "text_num": None, "kind": "superlative",
                    "supported": ok, "why": f"{'argmax' if sup else 'argmin'} {metric} is {sorted(winners)}"})
    return out


FAILSTATE_RE = re.compile(r"recent failures|failed recently|recently failed|had (?:recent )?failures|with failures", re.I)
FAILNEG_RE = re.compile(r"\bno (?:recent )?fail|\bwithout (?:recent )?fail|\b0 (?:\w+ )?(?:that )?fail|\bno failed|\bwere no\b|\bzero\b", re.I)
ABSTAIN2_RE = re.compile(r"does not (?:provide|contain|include|report|have)|not provided|no (?:information|data) (?:about|for|on)", re.I)


def check_v2(answer, ev, question="", tol=None):
    r = check_v1(answer, ev, tol=tol, state_claims=True)
    extra = _superlative_claims(answer, ev)
    # failure-state claims ("busy with recent failures") -> failed_last_hour > 0
    for sent in SENT_SPLIT.split(answer):
        if FAILSTATE_RE.search(sent) and not NEG_RE.search(sent) and not FAILNEG_RE.search(sent):
            for _, _, nd in _node_spans(sent):
                rec = ev.by_node.get(nd.lower())
                if rec is not None:
                    extra.append({"node": nd, "metric": "fail_state", "value": ">0", "text_num": None, "kind": "state",
                                  "supported": rec["failed_last_hour"] > 0, "why": f"failed_last_hour={rec['failed_last_hour']}"})
    # question-aware target checks
    q_nodes = [n for _, _, n in _node_spans(question)] if question else []
    abst = bool(ABSTAIN_RE.search(answer) or ABSTAIN2_RE.search(answer))
    for tn in q_nodes:
        about_target = any(c["node"] == tn for c in r["claims"])
        if tn.lower() not in ev.by_node:
            ok = abst and not any(c["kind"] == "numeric" and c["node"] != tn for c in r["claims"])
            extra.append({"node": tn, "metric": "target_absent", "value": None, "text_num": None, "kind": "target",
                          "supported": ok, "why": "question names a node absent from E; answer must abstain"})
        else:
            ok = about_target and not abst
            extra.append({"node": tn, "metric": "target_present", "value": None, "text_num": None, "kind": "target",
                          "supported": ok, "why": "answer must make a checkable claim about the named node and not deny its data"})
    if question and SET_Q_RE.search(question) and IDLE_Q_RE.search(question):
        claimed = {c["node"] for c in r["claims"] if c["kind"] == "state" and c["value"] == "idle"}
        truth = {x["node"] for x in ev.records if x["running_instances"] == 0}
        extra.append({"node": None, "metric": "idle_set", "value": sorted(claimed), "text_num": None, "kind": "set",
                      "supported": claimed == truth, "why": f"idle set in E is {sorted(truth)}"})
    claims = r["claims"] + extra
    bad = sum(not c["supported"] for c in claims)
    verdict = r["verdict"]
    if extra:
        if r["unknown_nodes"] and not ABSTAIN_RE.search(answer):
            verdict = "unsupported"
        elif bad == 0:
            verdict = "supported"
        else:
            verdict = "unsupported"
    return {**r, "verdict": verdict, "claims": claims, "n_claims": len(claims), "n_unsupported": bad}
