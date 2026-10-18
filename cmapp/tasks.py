"""Safe command dispatcher and background job manager.

The browser can only start operations listed in COMMANDS. Each entry
validates its arguments and returns a fixed argument vector for the
existing console runner (`cm`), so web tasks run exactly the same code as
`./run.sh <command>`. No shell is involved: argv lists go to
subprocess.Popen with shell=False.

In-process jobs (ask / propose / confirm / demo-check) use the same job
manager so that their progress and output appear in the console panel.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque

from .service import MODEL_NAMES, PROMPT_STYLES, RETRIEVAL_MODES, ValidationError

SEEDS_RE = re.compile(r"^\d{1,2}(,\d{1,2}){0,4}$")
ITEMSET_RE = re.compile(r"^gen:(qwen1\.5b|llama1b|qwen7b)_(full|tfidf)_(contract|original):\d{1,2}$")
MAX_LOG_LINES = 20000


def _choice(args, key, choices, default=None):
    v = args.get(key, default)
    if v not in choices:
        raise ValidationError(f"'{key}' must be one of: {', '.join(choices)}.")
    return v


def _cm(*argv):
    return ["bash", "cm", *argv]


def _v_none(args):
    return []


def _v_download(args):
    models = args.get("models") or []
    if isinstance(models, str):
        models = [m for m in re.split(r"[\s,]+", models) if m]
    if not isinstance(models, list) or any(m not in MODEL_NAMES for m in models):
        raise ValidationError(f"'models' must be a subset of: {', '.join(MODEL_NAMES)}.")
    return list(dict.fromkeys(models))


def _v_generate(args):
    model = _choice(args, "model", MODEL_NAMES)
    retrieval = _choice(args, "retrieval", RETRIEVAL_MODES, "full")
    style = _choice(args, "style", PROMPT_STYLES, "contract")
    seeds = str(args.get("seeds", "0")).replace(" ", "")
    if not SEEDS_RE.match(seeds):
        raise ValidationError("'seeds' must be a comma-separated list of up to 5 integers 0-99, e.g. 0,1,2.")
    argv = [model, retrieval, style, seeds]
    limit = args.get("limit")
    if limit not in (None, "", 0, "0"):
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            raise ValidationError("'limit' must be an integer.")
        if not 1 <= limit <= 320:
            raise ValidationError("'limit' must be between 1 and 320 questions.")
        argv.append(str(limit))
    return argv


def _v_judge(args):
    judge = _choice(args, "judge", MODEL_NAMES)
    itemset = str(args.get("itemset", "")).strip()
    if itemset != "corrupt" and not ITEMSET_RE.match(itemset):
        raise ValidationError("'itemset' must be 'corrupt' or gen:<model>_<retrieval>_<style>:<seed>, "
                              "e.g. gen:llama1b_full_contract:0.")
    return [judge, itemset]


def _v_actions(args):
    return [_choice(args, "model", MODEL_NAMES)]


# name -> (label, description, validator, argv builder, heavy?)
COMMANDS = {
    "test": ("Unit tests", "Run the test suite (checkers, service, web API).", _v_none,
             lambda a: _cm("test"), False),
    "tables": ("Recompute result tables", "Recompute every number and table from the shipped raw outputs (runs/).",
               _v_none, lambda a: _cm("tables"), False),
    "prepare-data": ("Prepare data", "Extract the 366-machine subset from ../dataset, rebuild the test data, verify hashes.",
                     _v_none, lambda a: _cm("prepare-data"), True),
    "download-models": ("Download models", "Download GGUF models (about 7 GB for all three) and verify MD5.",
                        _v_download, lambda a: _cm("download-models", *a), True),
    "generate": ("Generate answers", "Answer the test questions with a model (resumable JSONL in runs/).",
                 _v_generate, lambda a: _cm("generate", *a), True),
    "judge": ("Model verifier baseline", "Run a self-check / 7B judge over an item set.",
              _v_judge, lambda a: _cm("judge", *a), True),
    "actions": ("LLM action proposals", "End-to-end action proposals from a model plus every guard.",
                _v_actions, lambda a: _cm("actions", *a), True),
    "reproduce-all": ("Reproduce all experiments", "Every model run, into runs_rerun/ (many hours; resumable).",
                      _v_none, lambda a: _cm("reproduce-all"), True),
    "tables-rerun": ("Tables from re-run", "Analyse runs_rerun/ into results_rerun/.",
                     _v_none, lambda a: _cm("tables-rerun"), False),
}


def build_argv(name, args):
    if name not in COMMANDS:
        raise ValidationError(f"Unknown command '{name}'.")
    if not isinstance(args, dict):
        raise ValidationError("Arguments must be an object.")
    _, _, validator, builder, _ = COMMANDS[name]
    return builder(validator(args))


def cli_equivalent(argv):
    """Human-readable command line (display only; never executed by a shell)."""
    if argv[:2] == ["bash", "cm"]:
        argv = ["./run.sh", *argv[2:]]
    return " ".join(a if re.match(r"^[\w.,:/=+-]+$", a) else "'" + a.replace("'", "'\\''") + "'" for a in argv)


class Job:
    def __init__(self, kind, name, title, command):
        self.id = uuid.uuid4().hex[:12]
        self.kind, self.name, self.title, self.command = kind, name, title, command
        self.status = "queued"
        self.created = time.time()
        self.started = self.finished = None
        self.returncode = None
        self.result = None
        self.error = None
        self.lines = deque(maxlen=MAX_LOG_LINES)
        self.first_line = 0          # index of lines[0] (after old lines were dropped)
        self.proc = None
        self.lock = threading.Lock()

    def log(self, text):
        with self.lock:
            for line in str(text).split("\n"):
                if len(self.lines) == self.lines.maxlen:
                    self.first_line += 1
                self.lines.append(line)

    def snapshot(self, since=0):
        with self.lock:
            start = max(0, since - self.first_line)
            lines = list(self.lines)[start:]
            nxt = self.first_line + len(self.lines)
        return {"id": self.id, "kind": self.kind, "name": self.name, "title": self.title, "command": self.command,
                "status": self.status, "created": self.created, "started": self.started, "finished": self.finished,
                "returncode": self.returncode, "error": self.error, "result": self.result,
                "lines": lines, "next": nxt}


class JobManager:
    def __init__(self, root, max_jobs=100):
        self.root = root
        self.jobs: dict[str, Job] = {}
        self.order = deque(maxlen=max_jobs)
        self.lock = threading.Lock()

    def _add(self, job):
        with self.lock:
            if len(self.order) == self.order.maxlen:
                old = self.order[0]
                if self.jobs.get(old) and self.jobs[old].status not in ("queued", "running"):
                    self.jobs.pop(old, None)
            self.order.append(job.id)
            self.jobs[job.id] = job
        return job

    def get(self, jid):
        j = self.jobs.get(str(jid))
        if j is None:
            raise ValidationError("Unknown job.")
        return j

    def list(self):
        with self.lock:
            ids = list(self.order)
        return [self.jobs[i].snapshot(since=10**12) for i in reversed(ids) if i in self.jobs]

    def running_heavy(self):
        return [j for j in self.jobs.values() if j.kind == "command" and j.status in ("queued", "running")
                and COMMANDS[j.name][4]]

    # ---------------- subprocess jobs
    def start_command(self, name, args):
        argv = build_argv(name, args)
        if COMMANDS[name][4] and self.running_heavy():
            raise ValidationError("Another long-running task is in progress. Wait for it or cancel it first.")
        job = self._add(Job("command", name, COMMANDS[name][0], cli_equivalent(argv)))
        threading.Thread(target=self._run_proc, args=(job, argv), daemon=True).start()
        return job

    def _run_proc(self, job, argv):
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        job.log(f"$ {job.command}")
        job.status, job.started = "running", time.time()
        try:
            job.proc = subprocess.Popen(argv, cwd=self.root, env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
                                        bufsize=1, shell=False, start_new_session=True)
            for line in job.proc.stdout:
                job.log(line.rstrip("\n"))
            job.returncode = job.proc.wait()
            if job.status == "cancelling":
                job.status = "cancelled"
            else:
                job.status = "succeeded" if job.returncode == 0 else "failed"
        except Exception as e:  # pragma: no cover - defensive
            job.status, job.error = "failed", str(e)
            job.log(f"[error] {e}")
        job.finished = time.time()
        job.log(f"[{job.status}] exit code {job.returncode} after {job.finished - job.started:.1f}s")

    def cancel(self, jid):
        job = self.get(jid)
        if job.kind != "command" or job.status != "running" or job.proc is None:
            raise ValidationError("Only running tasks can be cancelled.")
        job.status = "cancelling"
        try:
            os.killpg(job.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        return job

    # ---------------- in-process jobs
    def start_call(self, name, title, command, fn):
        """fn(job) -> result; runs in a thread; output via job.log."""
        job = self._add(Job("call", name, title, command))

        def run():
            job.log(f"$ {command}")
            job.status, job.started = "running", time.time()
            try:
                job.result = fn(job)
                job.status = "succeeded"
            except ValidationError as e:
                job.status, job.error = "failed", str(e)
                job.log(f"Error: {e}")
            except Exception as e:
                job.status, job.error = "failed", f"{type(e).__name__}: {e}"
                job.log(f"Error: {type(e).__name__}: {e}")
                job.log(traceback.format_exc(limit=3))
            job.finished = time.time()
            job.log(f"[{job.status}] {job.finished - job.started:.1f}s")

        threading.Thread(target=run, daemon=True).start()
        return job

    def wait(self, jid, timeout=60):  # used by tests
        job = self.get(jid)
        t = time.time()
        while job.status in ("queued", "running", "cancelling") and time.time() - t < timeout:
            time.sleep(0.05)
        return job


def python_argv(*args):
    return [sys.executable, *args]
