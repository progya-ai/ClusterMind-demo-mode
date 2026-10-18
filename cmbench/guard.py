"""Action pipeline: schema -> whitelist -> inventory resolution -> busy-node
precondition -> typed confirmation -> execute. The precondition is a pure
function of a state record, so it can be evaluated at proposal time (on E)
or at confirmation time (on fresh state)."""
import json
import re

ALLOWED_ACTIONS = ["shutdown_node", "start_node", "migrate_job", "send_alert"]
DISRUPTIVE = {"shutdown_node", "migrate_job"}
UTIL_IDLE_PCT = 10.0  # the original ClusterMind idle threshold
NODE_RE = re.compile(r"\bnode[\s_-]?(\d+)\b", re.I)


def parse_action(raw):
    c = raw.strip()
    if c.startswith("```"):
        c = c.strip("`")
        if c.lower().startswith("json"):
            c = c[4:]
    m = re.search(r"\{.*\}", c, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return {"action": d.get("action", "none"), "node": d.get("node"), "reason": d.get("reason", "")}


def resolve(cmd, instruction, inventory):
    """Explicit node named in the instruction wins (as in the original);
    otherwise the model's node must exist in the inventory. No risk-score
    fallback (the original fell back to the highest-risk node)."""
    if cmd is None or cmd["action"] in (None, "none"):
        return None, "no action"
    if cmd["action"] not in ALLOWED_ACTIONS:
        return None, "action not whitelisted"
    m = NODE_RE.search(instruction)
    node = f"Node-{int(m.group(1))}" if m else None
    if node is None and cmd.get("node"):
        mm = NODE_RE.search(str(cmd["node"]))
        node = f"Node-{int(mm.group(1))}" if mm else None
    if node is None or node not in inventory:
        return None, "unknown node"
    return {**cmd, "node": node}, "ok"


def busy_util(state):  # original heuristic: busy iff GPU utilization above idle threshold
    return state["gpu_util_pct"] > UTIL_IDLE_PCT


def busy_sched(state):  # scheduler truth: busy iff any instance is active
    return state["running_instances"] > 0


def verify_typed(node, typed):
    return bool(node) and bool(typed) and typed.strip().lower() == node.strip().lower()


GUARDS = {
    # name: (predicate or None, evaluated at 'proposal'|'confirm', hard block?)
    "none": (None, None, False),
    "as_built": (busy_util, "confirm", False),      # original: shutdown proceeds with a warning
    "util@proposal": (busy_util, "proposal", True),
    "util@confirm": (busy_util, "confirm", True),
    "sched@proposal": (busy_sched, "proposal", True),
    "sched@confirm": (busy_sched, "confirm", True),
    "sched@both": (busy_sched, "both", True),
}


def decide(guard, action, state_prop, state_conf):
    """Returns True if the action is executed."""
    pred, when, hard = GUARDS[guard]
    if pred is None or action not in DISRUPTIVE:
        return True
    if not hard:
        return True
    if when == "proposal":
        return not pred(state_prop)
    if when == "confirm":
        return not pred(state_conf)
    return not (pred(state_prop) or pred(state_conf))
