"""ClusterMind application service layer.

Single place for the operations that both front-ends expose:

* the console (``cm_demo.py`` and ``./cm demo-check``) and
* the web GUI (``cmapp/web.py``).

Every operation returns plain dictionaries (JSON-serialisable) and has a
``render_*`` companion that produces the exact text the console prints, so
the two interfaces show the same results in the same format. The research
code in ``cmbench/`` is called, never copied.
"""
from __future__ import annotations

import datetime
import json
import os
import re
import threading
import time
import uuid

from cmbench.checkers import check_v0, check_v1s, check_v2
from cmbench.evidence import retrieve
from cmbench.guard import (ALLOWED_ACTIONS, DISRUPTIVE, busy_sched, busy_util,
                           parse_action, resolve, verify_typed)
from cmbench.prompts import action_user, diagnosis_system, diagnosis_user
from cmbench.score import score as task_score

MODEL_NAMES = ("qwen1.5b", "llama1b", "qwen7b")
RETRIEVAL_MODES = ("full", "tfidf")
PROMPT_STYLES = ("contract", "original")
MAX_QUESTION_LEN = 500
MAX_ANSWER_LEN = 4000
SNAPSHOT_RE = re.compile(r"^S\d{3}$")
PROPOSAL_TTL_S = 3600

# The five example answers shown by `./cm demo-check` (snapshot S003).
DEMO_CHECK_SNAPSHOT = "S003"
DEMO_CHECK_EXAMPLES = [
    ("Is Node-1 busy?", "Node-1 looks fine."),
    ("What is the GPU utilization of Node-2?", "Node-2 has 113.2% GPU utilization."),
    ("What is the GPU utilization of Node-2?", "Node-2 has 113.2% GPU utilization and 90 GB of GPU memory."),
    ("Which node is busiest?", "Node-4 is the busiest node with 0.0% GPU utilization."),
    ("What is the GPU utilization of Node-15?", "Node-15 is not present in the telemetry."),
]


class ValidationError(ValueError):
    """Bad user input. The message is safe to show to the user."""


class NotAvailable(RuntimeError):
    """A resource (model file, trace parquet, results file) is missing."""


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


class DemoInstructionParser:
    """Stand-in for the AI model in public demo mode (no model files are shipped to the hosted demo).

    It reads the operator's instruction with simple keyword rules and answers in the same JSON shape
    the real model uses. Everything after it is the real pipeline: whitelist, node inventory,
    typed confirmation and the busy-node block at confirmation time."""

    RULES = (("migrate_job", ("migrate", "move the job", "move job")),
             ("shutdown_node", ("shut down", "shutdown", "shut off", "power off", "turn off", "switch off", "stop")),
             ("start_node", ("start", "power on", "turn on", "boot", "wake")),
             ("send_alert", ("alert", "notify", "warn", "page ")))

    def chat(self, model, system, user, seed=0):
        text = user.rsplit("Instruction:", 1)[-1].strip().lower()
        for action, words in self.RULES:
            if any(w in text for w in words):
                return {"text": json.dumps({"action": action, "node": None, "reason": "keyword match (demo parser)"})}
        return {"text": json.dumps({"action": "none", "node": None, "reason": "no clear action in the instruction"})}


class ClusterMindService:
    """Stateless operations over one project folder, plus a small in-memory
    store of pending action proposals (so a confirmation can never be forged
    by a client: the server re-reads the proposal it issued)."""

    def __init__(self, root: str = ".", llm_backend=None, interface: str = "cli"):
        self.root = os.path.abspath(root)
        self.interface = interface
        self._llm = llm_backend
        self._llm_lock = threading.Lock()      # llama.cpp contexts are not thread-safe
        self._trace_lock = threading.Lock()
        self._trace = None
        self._snaps = None
        self._questions = None
        self._pending: dict[str, dict] = {}
        self._pending_lock = threading.Lock()
        self._runs_cache: dict[str, tuple] = {}

    # ------------------------------------------------------------ paths
    def path(self, *parts):
        return os.path.join(self.root, *parts)

    @property
    def audit_path(self):
        return self.path("runs", "demo_audit.jsonl")

    # ------------------------------------------------------------ data
    def snapshots(self) -> dict:
        if self._snaps is None:
            with open(self.path("data", "snapshots.json")) as f:
                self._snaps = {s["snapshot_id"]: s for s in json.load(f)}
        return self._snaps

    def questions(self) -> list:
        if self._questions is None:
            with open(self.path("data", "questions.json")) as f:
                self._questions = json.load(f)
        return self._questions

    def snapshot(self, sid) -> dict:
        sid = self.validate_snapshot(sid)
        return self.snapshots()[sid]

    def snapshot_summaries(self):
        out = []
        for sid, s in sorted(self.snapshots().items()):
            n = s["nodes"]
            out.append({"snapshot_id": sid, "t": s["t"], "n_nodes": len(n),
                        "idle": sum(r["running_instances"] == 0 for r in n),
                        "with_failures": sum(r["failed_last_hour"] > 0 for r in n),
                        "max_util": max(r["gpu_util_pct"] for r in n)})
        return out

    def sample_questions(self, sid=None):
        qs = self.questions()
        if sid:
            sid = self.validate_snapshot(sid)
            qs = [q for q in qs if q["snapshot_id"] == sid]
        return [{"qid": q["qid"], "template": q["template"], "question": q["question"]} for q in qs]

    def trace_available(self) -> bool:
        return os.path.exists(self.path("data", "instance.parquet"))

    # ------------------------------------------------------------ validation
    def validate_snapshot(self, sid):
        sid = str(sid or "").strip().upper()
        if not SNAPSHOT_RE.match(sid) or sid not in self.snapshots():
            ids = sorted(self.snapshots())
            raise ValidationError(f"Unknown snapshot '{sid}'. Choose {ids[0]}..{ids[-1]}.")
        return sid

    @staticmethod
    def validate_model(model):
        if model not in MODEL_NAMES:
            raise ValidationError(f"Unknown model '{model}'. Choose one of: {', '.join(MODEL_NAMES)}.")
        return model

    @staticmethod
    def validate_text(text, what, max_len):
        text = (text or "").strip()
        if not text:
            raise ValidationError(f"{what} must not be empty.")
        if len(text) > max_len:
            raise ValidationError(f"{what} is too long ({len(text)} characters; maximum {max_len}).")
        if any(ord(c) < 32 and c not in "\n\t" for c in text):
            raise ValidationError(f"{what} contains control characters.")
        return text

    @staticmethod
    def validate_int(value, what, lo, hi):
        try:
            v = int(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{what} must be an integer.")
        if not lo <= v <= hi:
            raise ValidationError(f"{what} must be between {lo} and {hi}.")
        return v

    @staticmethod
    def validate_choice(value, what, choices):
        if value not in choices:
            raise ValidationError(f"{what} must be one of: {', '.join(choices)}.")
        return value

    # ------------------------------------------------------------ models
    def _backend(self):
        if self._llm is None:
            from cmbench import llm  # imported lazily: llama-cpp is only needed for model calls
            self._llm = llm
        return self._llm

    def model_files(self):
        from cmbench import llm_paths
        return llm_paths.model_paths()

    def models_status(self):
        out = []
        for name, p in self.model_files().items():
            ok = os.path.exists(p)
            out.append({"model": name, "file": os.path.basename(p), "present": ok,
                        "size_gb": round(os.path.getsize(p) / 1e9, 2) if ok else None})
        return out

    def require_model(self, model):
        self.validate_model(model)
        p = self.model_files()[model]
        if not os.path.exists(p) and self._llm is None:
            raise NotAvailable(f"Model file {os.path.basename(p)} not found in {os.path.dirname(p)}. "
                               f"Download it first (task 'download-models').")

    def _chat(self, model, system, user, seed):
        with self._llm_lock:
            return self._backend().chat(model, system, user, seed=seed)

    # ------------------------------------------------------------ audit
    def audit(self, event):
        os.makedirs(os.path.dirname(self.audit_path), exist_ok=True)
        event["time"] = _now()
        event.setdefault("via", self.interface)
        with open(self.audit_path, "a") as f:
            f.write(json.dumps(event) + "\n")
        return event

    def audit_log(self, limit=500, query=""):
        if not os.path.exists(self.audit_path):
            return []
        rows = []
        with open(self.audit_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        q = (query or "").strip().lower()
        if q:
            rows = [r for r in rows if q in json.dumps(r).lower()]
        return rows[-limit:][::-1]

    # ------------------------------------------------------------ checking (no model)
    def check(self, sid, question, answer, mode="full"):
        """Run the deterministic checkers on a given answer. No model call."""
        s = self.snapshot(sid)
        question = (question or "").strip()
        if len(question) > MAX_QUESTION_LEN:
            raise ValidationError(f"Question is too long (maximum {MAX_QUESTION_LEN} characters).")
        answer = self.validate_text(answer, "Answer", MAX_ANSWER_LEN)
        self.validate_choice(mode, "Retrieval", RETRIEVAL_MODES)
        ev = retrieve(s, question, mode=mode)
        t = time.perf_counter()
        v2 = check_v2(answer, ev, question)
        dt_us = (time.perf_counter() - t) * 1e6
        return {"snapshot": s["snapshot_id"], "question": question, "answer": answer,
                "evidence_hash": ev.digest, "evidence_text": ev.text, "evidence_nodes": ev.nodes(),
                "v0": check_v0(answer, ev)["verdict"], "v1s": check_v1s(answer, ev)["verdict"],
                "v2": v2["verdict"], "delivered": v2["verdict"] in ("supported", "abstain"),
                "claims": [_claim(c) for c in v2["claims"]], "check_us": round(dt_us, 1)}

    def demo_check(self):
        return [self.check(DEMO_CHECK_SNAPSHOT, q, a) for q, a in DEMO_CHECK_EXAMPLES]

    def batch_check(self, rows, max_rows=2000):
        """rows: iterable of dicts with snapshot_id, question, answer."""
        out = []
        for i, r in enumerate(rows):
            if i >= max_rows:
                raise ValidationError(f"Too many rows (maximum {max_rows}).")
            try:
                c = self.check(r.get("snapshot_id") or r.get("snapshot"), r.get("question", ""), r.get("answer", ""))
                out.append({"row": i + 1, "snapshot": c["snapshot"], "question": c["question"], "answer": c["answer"],
                            "v0": c["v0"], "v1s": c["v1s"], "v2": c["v2"], "delivered": c["delivered"],
                            "n_claims": len(c["claims"]),
                            "n_unsupported": sum(not x["supported"] for x in c["claims"]), "error": ""})
            except ValidationError as e:
                out.append({"row": i + 1, "snapshot": str(r.get("snapshot_id", "")), "question": str(r.get("question", ""))[:200],
                            "answer": str(r.get("answer", ""))[:200], "v0": "", "v1s": "", "v2": "", "delivered": "",
                            "n_claims": "", "n_unsupported": "", "error": str(e)})
        return out

    # ------------------------------------------------------------ asking (model)
    def ask(self, model, sid, question, seed=0, mode="full", style="contract"):
        s = self.snapshot(sid)
        question = self.validate_text(question, "Question", MAX_QUESTION_LEN)
        seed = self.validate_int(seed, "Seed", 0, 2**31 - 1)
        self.validate_choice(mode, "Retrieval", RETRIEVAL_MODES)
        self.validate_choice(style, "Prompt style", PROMPT_STYLES)
        self.require_model(model)
        ev = retrieve(s, question, mode=mode)
        r = self._chat(model, diagnosis_system(ev.text), diagnosis_user(question, style), seed)
        v2 = check_v2(r["text"], ev, question)
        v0 = check_v0(r["text"], ev)
        delivered = v2["verdict"] in ("supported", "abstain")
        self.audit({"type": "answer", "snapshot": s["snapshot_id"], "model": model, "question": question,
                    "answer": r["text"], "evidence_hash": ev.digest, "v2": v2["verdict"], "v0": v0["verdict"]})
        return {"snapshot": s["snapshot_id"], "model": model, "question": question, "answer": r["text"],
                "latency_s": r["latency_s"], "evidence_hash": ev.digest, "evidence_text": ev.text,
                "evidence_nodes": ev.nodes(), "v2": v2["verdict"], "v0": v0["verdict"], "delivered": delivered,
                "claims": [_claim(c) for c in v2["claims"]], "seed": seed, "retrieval": mode, "style": style}

    # ------------------------------------------------------------ actions (model + guard)
    def propose(self, model, sid, instruction, seed=0):
        """Step 1 of the action path: model proposal -> whitelist -> inventory.
        Returns a proposal; disruptive actions need `confirm()`."""
        s = self.snapshot(sid)
        instruction = self.validate_text(instruction, "Instruction", MAX_QUESTION_LEN)
        seed = self.validate_int(seed, "Seed", 0, 2**31 - 1)
        self.require_model(model)
        ev = retrieve(s, instruction, mode="full")
        inv = {r["node"]: r for r in s["nodes"]}
        raw = self._chat(model, diagnosis_system(ev.text), action_user(instruction, ALLOWED_ACTIONS), seed)["text"]
        cmd, why = resolve(parse_action(raw), instruction, inv)
        base = {"snapshot": s["snapshot_id"], "model": model, "instruction": instruction, "raw": raw}
        if cmd is None:
            self.audit({"type": "action", "snapshot": s["snapshot_id"], "instruction": instruction, "raw": raw,
                        "result": f"dropped ({why})"})
            return {**base, "status": "dropped", "reason": why}
        node = cmd["node"]
        if cmd["action"] not in DISRUPTIVE:
            self.audit({"type": "action", "snapshot": s["snapshot_id"], "action": cmd["action"], "node": node,
                        "result": "logged"})
            return {**base, "status": "logged", "action": cmd["action"], "node": node}
        p = inv[node]
        pid = uuid.uuid4().hex
        prop = {**base, "status": "needs_confirmation", "id": pid, "action": cmd["action"], "node": node,
                "proposal_state": _state(p), "util_proxy_busy": busy_util(p), "created": time.time()}
        with self._pending_lock:
            self._expire_pending()
            self._pending[pid] = prop
        return {k: v for k, v in prop.items() if k != "created"}

    def _expire_pending(self):
        now = time.time()
        for k in [k for k, v in self._pending.items() if now - v["created"] > PROPOSAL_TTL_S]:
            del self._pending[k]

    def pending(self, pid):
        with self._pending_lock:
            self._expire_pending()
            p = self._pending.get(str(pid or ""))
        if p is None:
            raise ValidationError("Unknown or expired proposal. Propose the action again.")
        return p

    def confirm(self, proposal, typed, delay=0):
        """Step 2: typed node-name confirmation, then the busy-node
        precondition on state re-read at confirmation time (hard block)."""
        if isinstance(proposal, str):
            with self._pending_lock:
                self._expire_pending()
                proposal = self._pending.pop(proposal, None)
            if proposal is None:
                raise ValidationError("Unknown or expired proposal. Propose the action again.")
        delay = self.validate_int(delay, "Confirmation delay", 0, 86400)
        typed = (typed or "")[:100]
        s = self.snapshot(proposal["snapshot"])
        node, action = proposal["node"], proposal["action"]
        if not verify_typed(node, typed):
            self.audit({"type": "action", "snapshot": s["snapshot_id"], "action": action, "node": node,
                        "result": "not confirmed"})
            return {"node": node, "action": action, "status": "not_confirmed", "result": "not confirmed"}
        c, src = self.state_at(s, node, delay)
        if busy_sched(c):
            result, status = "blocked (busy at confirmation)", "blocked"
        else:
            result, status = "executed (simulated)", "executed"
        self.audit({"type": "action", "snapshot": s["snapshot_id"], "action": action, "node": node,
                    "confirm_state_source": src, "running_at_confirm": c["running_instances"], "result": result})
        return {"node": node, "action": action, "status": status, "result": result,
                "confirm_state": _state(c), "confirm_state_source": src, "delay_s": delay}

    def _trace_tables(self):
        with self._trace_lock:
            if self._trace is None:
                from cmbench import snapshots as sn
                old = sn.DATA
                sn.DATA = self.path("data")
                try:
                    self._trace = sn.load_tables()
                finally:
                    sn.DATA = old
            return self._trace

    def state_at(self, s, node, delay):
        """Node state at confirmation time. With delay>0 and the trace parquet present,
        the state is re-derived from the trace at t+delay (real drift); otherwise the snapshot."""
        rec = next(r for r in s["nodes"] if r["node"] == node)
        if delay <= 0:
            return rec, "snapshot"
        if not self.trace_available():
            return rec, "snapshot (no trace parquet; run ./cm prepare-data for real drift)"
        from cmbench.snapshots import machine_state
        st = machine_state(self._trace_tables(), s["t"] + delay, [rec["machine"]]).loc[rec["machine"]]
        return {**rec, "running_instances": int(st.running_instances), "gpu_util_pct": float(st.gpu_util_pct),
                "gpu_mem_gb": float(st.gpu_mem_gb)}, f"trace at t+{delay}s"

    # ------------------------------------------------------------ recorded runs & results
    def run_files(self):
        out = []
        for d in ("runs", "runs_rerun"):
            p = self.path(d)
            if os.path.isdir(p):
                for f in sorted(os.listdir(p)):
                    if f.startswith("gen_") and f.endswith(".jsonl"):
                        out.append(f"{d}/{f}")
        return out

    def run_rows(self, run):
        if run not in self.run_files():
            raise ValidationError("Unknown run file.")
        p = self.path(run)
        mt = os.path.getmtime(p)
        hit = self._runs_cache.get(run)
        if hit and hit[0] == mt:
            return hit[1]
        Q = {q["qid"]: q for q in self.questions()}
        rows = []
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = Q.get(r.get("qid"))
                if q is None:
                    continue
                s = self.snapshots()[q["snapshot_id"]]
                ev = retrieve(s, q["question"], mode=r.get("retrieval", "full"), k=3)
                v2 = check_v2(r["answer"], ev, q["question"])["verdict"]
                rows.append({"qid": r["qid"], "seed": r["seed"], "snapshot": q["snapshot_id"], "template": q["template"],
                             "question": q["question"], "answer": r["answer"], "v2": v2,
                             "delivered": v2 in ("supported", "abstain"), "correct": bool(task_score(q, r["answer"])),
                             "latency_s": round(r.get("latency_s", 0.0), 2)})
        self._runs_cache[run] = (mt, rows)
        return rows

    def headline(self):
        p = self.path("results", "e2_endtoend.json")
        if not os.path.exists(p):
            return []
        with open(p) as f:
            r = json.load(f)
        out = []
        for tag in ("qwen1.5b_full_contract", "llama1b_full_contract", "qwen7b_full_contract"):
            if tag in r:
                g = r[tag]["gates"]
                out.append({"config": tag, "ungated_acc": r[tag]["acc"],
                            "v0_coverage": g["v0"]["coverage"], "v0_delivered_acc": g["v0"]["delivered_acc"],
                            "v2_coverage": g["v2"]["coverage"], "v2_delivered_acc": g["v2"]["delivered_acc"]})
        return out

    def status(self):
        return {"snapshots": len(self.snapshots()), "questions": len(self.questions()),
                "trace_available": self.trace_available(), "models": self.models_status(),
                "audit_entries": len(self.audit_log(limit=10**9)), "headline": self.headline(),
                "allowed_actions": ALLOWED_ACTIONS, "disruptive_actions": sorted(DISRUPTIVE)}


# ---------------------------------------------------------------- helpers
def _state(r):
    return {"running_instances": r["running_instances"], "gpu_util_pct": r["gpu_util_pct"],
            "gpu_mem_gb": r["gpu_mem_gb"], "failed_last_hour": r.get("failed_last_hour")}


def _claim(c):
    return {"kind": c["kind"], "node": c["node"], "metric": c["metric"], "supported": bool(c["supported"]),
            "why": c["why"]}


# ---------------------------------------------------------------- renderers (console text)
def render_snapshot(s):
    lines = [f"\n=== Snapshot {s['snapshot_id']}  (trace time t={s['t']}s) ===",
             f"{'node':8s} {'running':>8s} {'GPU util %':>11s} {'GPU mem GB':>11s} {'failed/1h':>10s}"]
    for r in s["nodes"]:
        lines.append(f"{r['node']:8s} {r['running_instances']:8d} {r['gpu_util_pct']:11.1f} "
                     f"{r['gpu_mem_gb']:11.1f} {r['failed_last_hour']:10d}")
    return "\n".join(lines)


def render_claims(claims):
    return "\n".join(f"   {'ok ' if c['supported'] else 'BAD'} {c['kind']:11s} {str(c['node']):8s} "
                     f"{str(c['metric']):18s} {c['why']}" for c in claims)


def render_answer(r):
    out = [f"\n[{r['model']}, {r['latency_s']:.1f}s] {r['answer']}",
           f"\nEvidence hash h(E) = {r['evidence_hash']}",
           f"v2 verdict: {r['v2'].upper()}  -> {'DELIVERED' if r['delivered'] else 'SHOWN WITH WARNING'}",
           f"v0 (prototype) verdict: {r['v0']}"]
    if r["claims"]:
        out.append(render_claims(r["claims"]))
    return "\n".join(out)


def render_demo_check(results, evidence_text):
    """Exact text of `./cm demo-check` (write it verbatim, without an extra newline)."""
    out = evidence_text + " \n\n"
    for c in results:
        out += (f"Q: {c['question']}\nA: {c['answer']}\n"
                f"   v0 (prototype): {c['v0']:12s}  v2 (new checker): {c['v2']}\n\n")
    return out


def render_check(c):
    out = [f"Evidence hash h(E) = {c['evidence_hash']}",
           f"v2 verdict: {c['v2'].upper()}  -> {'DELIVERED' if c['delivered'] else 'SHOWN WITH WARNING'}",
           f"v1s verdict: {c['v1s']}", f"v0 (prototype) verdict: {c['v0']}"]
    if c["claims"]:
        out.append(render_claims(c["claims"]))
    return "\n".join(out)


def render_proposal(p):
    out = [f"\n[{p['model']}] proposed: {p['raw']}"]
    if p["status"] == "dropped":
        out.append(f"Dropped: {p['reason']}.")
        return "\n".join(out)
    out.append(f"Resolved: {p['action']} on {p['node']}")
    if p["status"] == "logged":
        out.append("Non-disruptive action: logged (simulated).")
        return "\n".join(out)
    st = p["proposal_state"]
    out.append(f"Proposal-time state: running={st['running_instances']} util={st['gpu_util_pct']}%"
               f"  (prototype util-proxy would say {'busy' if p['util_proxy_busy'] else 'idle'})")
    return "\n".join(out)


def render_confirmation(c):
    if c["status"] == "not_confirmed":
        return "Typed confirmation did not match; nothing executed."
    st = c["confirm_state"]
    out = [f"Confirmation-time state ({c['confirm_state_source']}): running={st['running_instances']} util={st['gpu_util_pct']}%"]
    if c["status"] == "blocked":
        out.append(f"BLOCKED: {c['node']} is running {st['running_instances']} instance(s). No override.")
    else:
        out.append(f"EXECUTED (simulated): {c['node']} marked shut down.")
    return "\n".join(out)


# ---------------------------------------------------------------- batch files (CLI and web)
BATCH_FIELDS = ["row", "snapshot", "question", "answer", "v0", "v1s", "v2", "delivered", "n_claims", "n_unsupported", "error"]
MAX_BATCH_BYTES = 2 * 1024 * 1024


def parse_batch(data: bytes, filename: str):
    """Parse an uploaded CSV (columns snapshot_id,question,answer) or JSONL file."""
    import csv
    import io
    if len(data) > MAX_BATCH_BYTES:
        raise ValidationError(f"File too large (maximum {MAX_BATCH_BYTES // 1024} KB).")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValidationError("File must be UTF-8 text.")
    ext = os.path.splitext(filename or "")[1].lower()
    rows = []
    if ext in (".jsonl", ".json"):
        if ext == ".json":
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as e:
                raise ValidationError(f"Invalid JSON: {e.msg} (line {e.lineno}).")
            if not isinstance(obj, list):
                raise ValidationError("A .json file must contain a list of objects.")
            rows = obj
        else:
            for i, line in enumerate(text.splitlines(), 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValidationError(f"Invalid JSON on line {i}: {e.msg}.")
    elif ext == ".csv":
        reader = csv.DictReader(io.StringIO(text))
        need = {"question", "answer"}
        if not reader.fieldnames or not need <= set(reader.fieldnames) or not (
                {"snapshot_id", "snapshot"} & set(reader.fieldnames)):
            raise ValidationError("CSV needs the columns snapshot_id, question, answer.")
        rows = list(reader)
    else:
        raise ValidationError("Upload a .csv, .jsonl or .json file.")
    if not rows:
        raise ValidationError("The file contains no rows.")
    if not all(isinstance(r, dict) for r in rows):
        raise ValidationError("Every row must be an object with snapshot_id, question and answer.")
    return rows


def batch_to_csv(results) -> str:
    import csv
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=BATCH_FIELDS)
    w.writeheader()
    for r in results:
        w.writerow({k: r.get(k, "") for k in BATCH_FIELDS})
    return buf.getvalue()


def render_batch_summary(results):
    n = len(results)
    err = sum(bool(r["error"]) for r in results)
    ok = [r for r in results if not r["error"]]
    dl = sum(r["delivered"] is True for r in ok)
    from collections import Counter
    c = Counter(r["v2"] for r in ok)
    return (f"{n} rows checked ({err} invalid). v2 delivered {dl}/{len(ok)}; verdicts: "
            + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
