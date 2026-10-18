"""Task-correctness scorer: compares an answer to the GOLD answer of its
question (not to the evidence). Deliberately implemented independently of
the checker: numbers count only when directly adjacent to the unit or
keyword of the gold metric."""
import re

NODE_RE = re.compile(r"\bnode[\s_-]?(\d+)\b", re.IGNORECASE)
ABSTAIN_RE = re.compile(
    r"(not (?:present|found|listed|available|included|in the (?:telemetry|evidence|data))|"
    r"no (?:data|information|telemetry|record|such node)|does not exist|doesn't exist|"
    r"unknown node|cannot (?:find|determine)|can't (?:find|determine)|unable to (?:find|determine)|"
    r"there is no (?:node|information|data))", re.IGNORECASE)
UNIT = {
    "gpu_util_pct": re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", re.I),
    "gpu_mem_gb": re.compile(r"(\d+(?:\.\d+)?)\s*(?:gb|gib)\b", re.I),
    "failed_last_hour": re.compile(r"(\d+)\s+(?:\w+\s+){0,2}(?:fail\w*)|(?:fail\w*)\D{0,60}?(\d+)", re.I),
    "running_instances": re.compile(r"(\d+)\s+(?:running\s+)?(?:instances?|jobs?|tasks?)", re.I),
}
TOL = {"gpu_util_pct": 1.0, "gpu_mem_gb": 0.5, "running_instances": 0.0, "failed_last_hour": 0.0}
WORDNUM = {"zero": 0, "no": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def nodes_in(text):
    return [f"Node-{int(m.group(1))}" for m in NODE_RE.finditer(text)]


def _strip_nodes(text):
    return NODE_RE.sub(" NODE ", text)


def _values(text, metric):
    t = _strip_nodes(text)
    vals = []
    for m in UNIT[metric].finditer(t):
        g = [x for x in m.groups() if x is not None]
        if g:
            vals.append(float(g[0]))
    if metric == "failed_last_hour":
        for w, v in WORDNUM.items():
            if re.search(rf"\b{w}\s+(?:\w+\s+)?(?:failed|failures?)", t, re.I):
                vals.append(float(v))
    return vals


def _close(v, g, metric):
    return abs(v - g) <= TOL[metric] + 1e-9 or round(v) == round(g) and metric in ("gpu_util_pct", "gpu_mem_gb") and abs(v - g) < 1


def score(q, answer):
    tpl, gold = q["template"], q["gold"]
    ns = nodes_in(answer)
    abst = bool(ABSTAIN_RE.search(answer))
    if tpl == "trap_missing":
        return abst
    if abst and tpl != "argmax_failed":
        return False
    if tpl in ("argmax_util", "argmax_failed"):
        if not gold["nodes"]:
            return (not ns) or bool(re.search(r"\bno (?:node|failures|failed)|none|zero|\b0 failed", answer, re.I))
        return bool(ns) and ns[0] in gold["nodes"]
    if tpl in ("value_util", "value_mem", "value_failed"):
        if ns and ns[0] != q["target_node"]:
            return False
        vals = _values(answer, gold["metric"])
        return any(_close(v, gold["value"], gold["metric"]) for v in vals)
    if tpl == "idle_set":
        sents = re.split(r"(?<=[.!?;])\s+|\n+", answer)
        idle_sents = [s for s in sents if re.search(r"\bidle\b|not running any|nothing running|no running|0 running", s, re.I)
                      and not re.search(r"\bnot idle\b|\bbusy\b", s, re.I)]
        pred = set(n for s in idle_sents for n in nodes_in(s)) if idle_sents else set(ns)
        if not gold["nodes"]:
            return not pred
        return pred == set(gold["nodes"])
    if tpl == "busy_yesno":
        a = re.sub(r"\bnot idle\b", "busy", answer.lower())
        yes = bool(re.search(r"^\s*yes\b|\bis (?:currently )?running\b(?!\s+(?:any|no|0)\b)|\bbusy\b|\bactive\b|\b[1-9]\d* (?:running )?(?:instances?|jobs?)", a))
        no = bool(re.search(r"^\s*no\b|\bidle\b|not running|nothing (?:is )?running|\b0 (?:running )?(?:instances?|jobs?)|no (?:running )?(?:instances|jobs)", a))
        if re.search(r"\bnot busy\b", a):
            yes = False
        if yes == no:
            return False
        return yes == gold["busy"]
    raise KeyError(tpl)
