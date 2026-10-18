"""ClusterMind web GUI (Flask).

A thin HTTP layer over cmapp.service (the same logic the console uses) and
cmapp.tasks (an allowlisted dispatcher for the `cm` console commands).
Long operations run as background jobs; the page polls
/api/jobs/<id>?since=N to stream their output into the console panel.
"""
from __future__ import annotations

import ipaddress
import json
import os
import secrets
import time
from collections import deque

from flask import Flask, Response, abort, jsonify, render_template, request, send_file

from .service import (ClusterMindService, DemoInstructionParser, NotAvailable, ValidationError, batch_to_csv, parse_batch,
                      render_answer, render_batch_summary, render_check, render_confirmation, render_demo_check,
                      render_proposal, render_snapshot, retrieve, DEMO_CHECK_SNAPSHOT, MAX_BATCH_BYTES,
                      MODEL_NAMES, PROMPT_STYLES, RETRIEVAL_MODES)
from .tasks import COMMANDS, JobManager, cli_equivalent

# Folders whose files may be downloaded, and the file types allowed.
DOWNLOAD_ROOTS = {"results": "results", "results_rerun": "results_rerun", "tables": "results/tables",
                  "runs": "runs", "runs_rerun": "runs_rerun", "gui": "results_gui"}
DOWNLOAD_EXT = {".json", ".jsonl", ".txt", ".tex", ".csv"}
CSRF_HEADER = "X-CM-Request"


DEMO_BLOCKED = ("/api/ask", "/api/batch-check", "/api/tasks", "/api/files", "/api/audit", "/api/commands", "/download/")
DEMO_RATE_PER_MIN = 90        # POSTs per client address per minute in public demo mode


def create_app(root=".", service=None, token=None, allowed_hosts=None, demo=None):
    root = os.path.abspath(root)
    if demo is None:
        demo = os.environ.get("CM_DEMO", "").strip().lower() in ("1", "true", "yes", "on")
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_BATCH_BYTES + 64 * 1024
    app.config["JSON_SORT_KEYS"] = False
    svc = service or ClusterMindService(root, llm_backend=DemoInstructionParser() if demo else None, interface="web")
    jobs = JobManager(root)
    token = token if token is not None else os.environ.get("CM_WEB_TOKEN", "")
    hosts = {"localhost", "127.0.0.1", "::1"} | set(allowed_hosts or []) | {
        h.strip().lower() for h in os.environ.get("CM_WEB_ALLOWED_HOSTS", "").split(",") if h.strip()}
    app.extensions["cm"] = {"service": svc, "jobs": jobs, "demo": demo}

    hits: dict[str, deque] = {}

    def _rate_ok():
        ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or request.remote_addr or "?"
        now = time.time()
        q = hits.setdefault(ip, deque())
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= DEMO_RATE_PER_MIN:
            return False
        q.append(now)
        if len(hits) > 5000:
            hits.clear()
        return True

    # ------------------------------------------------------------ security
    @app.before_request
    def guard():
        host = (request.host or "").rsplit(":", 1)[0].strip("[]").lower()
        if not demo and host not in hosts and not _is_private_ip(host):
            abort(400, "Host not allowed (set CM_WEB_ALLOWED_HOSTS to add one).")
        if demo and (request.path.startswith(DEMO_BLOCKED) or request.path.rstrip("/") == "/api/jobs"):
            return jsonify(error="This feature is turned off in the public demo. Run ClusterMind locally to use it."), 403
        if demo and request.method == "POST" and not _rate_ok():
            return jsonify(error="Too many requests. Please wait a moment and try again."), 429
        if token:
            supplied = request.headers.get("X-CM-Token") or request.cookies.get("cm_token") or request.args.get("token")
            if not supplied or not secrets.compare_digest(supplied, token):
                if request.path.startswith("/api/") or request.path.startswith("/download/"):
                    return jsonify(error="Missing or wrong access token."), 401
                return Response("ClusterMind: open this page with ?token=<CM_WEB_TOKEN>.", 401, mimetype="text/plain")
        if request.method == "POST" and request.headers.get(CSRF_HEADER) != "1":
            return jsonify(error="Missing request header."), 403

    @app.after_request
    def headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if token and request.args.get("token") == token:
            resp.set_cookie("cm_token", token, httponly=True, samesite="Strict")
        return resp

    @app.errorhandler(ValidationError)
    def bad_input(e):
        return jsonify(error=str(e)), 400

    @app.errorhandler(NotAvailable)
    def missing(e):
        return jsonify(error=str(e)), 409

    @app.errorhandler(413)
    def too_big(e):
        return jsonify(error=f"Upload too large (maximum {MAX_BATCH_BYTES // 1024} KB)."), 413

    def body():
        d = request.get_json(silent=True)
        if not isinstance(d, dict):
            raise ValidationError("Expected a JSON object in the request body.")
        return d

    def job_response(job, code=202):
        return jsonify(job.snapshot()), code

    # ------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return render_template("index.html", models=MODEL_NAMES, retrievals=RETRIEVAL_MODES, styles=PROMPT_STYLES)

    @app.get("/healthz")
    def healthz():
        return jsonify(ok=True)

    # ------------------------------------------------------------ read-only data
    @app.get("/api/status")
    def status():
        st = svc.status()
        st["jobs_running"] = 0 if demo else sum(j["status"] == "running" for j in jobs.list())
        st["demo"] = demo
        if demo:
            st["models"] = []
            st["audit_entries"] = 0
        return jsonify(st)

    @app.get("/api/snapshots")
    def snapshots():
        return jsonify(svc.snapshot_summaries())

    @app.get("/api/snapshots/<sid>")
    def snapshot(sid):
        s = svc.snapshot(sid)
        ev = retrieve(s, "", "full")
        return jsonify({**s, "text": render_snapshot(s), "evidence_text": ev.text, "evidence_hash": ev.digest,
                        "questions": svc.sample_questions(s["snapshot_id"])})

    @app.get("/api/commands")
    def commands():
        return jsonify([{"name": k, "label": v[0], "description": v[1], "long": v[4]} for k, v in COMMANDS.items()])

    @app.get("/api/runs")
    def runs():
        return jsonify(svc.run_files())

    @app.get("/api/runs/rows")
    def run_rows():
        return jsonify(svc.run_rows(request.args.get("run", "")))

    @app.get("/api/audit")
    def audit():
        limit = svc.validate_int(request.args.get("limit", 500), "limit", 1, 5000)
        return jsonify(svc.audit_log(limit=limit, query=request.args.get("q", "")))

    # ------------------------------------------------------------ operations (background jobs)
    @app.post("/api/check")
    def check():
        d = body()
        sid, q, a, mode = d.get("snapshot"), d.get("question", ""), d.get("answer", ""), d.get("retrieval", "full")
        c = svc.check(sid, q, a, mode)          # validate + compute synchronously (microseconds)
        cmd = cli_equivalent(["./run.sh", "check", "--snapshot", c["snapshot"], "--question", c["question"],
                              "--answer", c["answer"]] + (["--retrieval", mode] if mode != "full" else []))
        job = jobs.start_call("check", "Check answer", cmd, lambda j: (j.log(render_check(c)), c)[1])
        return job_response(jobs.wait(job.id, 5), 200)

    @app.post("/api/demo-check")
    def demo_check():
        def run(j):
            res = svc.demo_check()
            j.log(render_demo_check(res, retrieve(svc.snapshot(DEMO_CHECK_SNAPSHOT), "", "full").text).rstrip("\n"))
            return res
        return job_response(jobs.start_call("demo-check", "Example checks", "./run.sh demo-check", run))

    @app.post("/api/ask")
    def ask():
        d = body()
        model = svc.validate_model(d.get("model"))
        sid = svc.validate_snapshot(d.get("snapshot"))
        q = svc.validate_text(d.get("question"), "Question", 500)
        seed = svc.validate_int(d.get("seed", 0), "Seed", 0, 2**31 - 1)
        mode = svc.validate_choice(d.get("retrieval", "full"), "Retrieval", RETRIEVAL_MODES)
        style = svc.validate_choice(d.get("style", "contract"), "Prompt style", PROMPT_STYLES)
        svc.require_model(model)
        argv = ["./run.sh", "demo", "--model", model, "--snapshot", sid, "--seed", str(seed), "--question", q]
        cmd = cli_equivalent(argv) + ("" if (mode, style) == ("full", "contract") else f"   # retrieval={mode}, prompt={style}")

        def run(j):
            j.log(f"loading {model} and generating (CPU; typically 5-60 s)...")
            r = svc.ask(model, sid, q, seed=seed, mode=mode, style=style)
            j.log(render_answer(r).lstrip("\n"))
            return r
        return job_response(jobs.start_call("ask", "Ask", cmd, run))

    @app.post("/api/actions/propose")
    def propose():
        d = body()
        model = svc.validate_model(d.get("model") or ("qwen1.5b" if demo else None))
        sid = svc.validate_snapshot(d.get("snapshot"))
        ins = svc.validate_text(d.get("instruction"), "Instruction", 500)
        seed = svc.validate_int(d.get("seed", 0), "Seed", 0, 2**31 - 1)
        svc.require_model(model)
        cmd = cli_equivalent(["./run.sh", "demo", "--model", model, "--snapshot", sid, "--seed", str(seed),
                              "--question", "act: " + ins])

        def run(j):
            j.log(f"loading {model} and proposing an action...")
            p = svc.propose(model, sid, ins, seed=seed)
            j.log(render_proposal(p).lstrip("\n"))
            if p["status"] == "needs_confirmation":
                j.log(f"Waiting for typed confirmation of {p['node']} in the GUI.")
            return p
        return job_response(jobs.start_call("propose", "Propose action", cmd, run))

    @app.post("/api/actions/confirm")
    def confirm():
        d = body()
        prop = svc.pending(d.get("proposal_id"))
        typed = str(d.get("typed", ""))[:100]
        delay = 0 if demo else svc.validate_int(d.get("delay", 0), "Confirmation delay", 0, 86400)
        cmd = f"confirm {prop['node']} (typed: {typed!r}, confirmation delay {delay}s)"

        def run(j):
            if delay > 0 and svc.trace_available():
                j.log("re-reading node state from the trace (first time loads the parquet, ~10 s)...")
            c = svc.confirm(prop["id"], typed, delay)
            j.log(render_confirmation(c))
            return c
        return job_response(jobs.start_call("confirm", "Confirm action", cmd, run))

    @app.post("/api/batch-check")
    def batch_check():
        f = request.files.get("file")
        if f is None or not f.filename:
            raise ValidationError("Choose a .csv or .jsonl file to upload.")
        name = os.path.basename(f.filename)
        rows = parse_batch(f.read(MAX_BATCH_BYTES + 1), name)

        def run(j):
            res = svc.batch_check(rows)
            os.makedirs(os.path.join(root, "results_gui"), exist_ok=True)
            out = f"batch_{j.id}.csv"                       # server-chosen name, never the upload's
            with open(os.path.join(root, "results_gui", out), "w", newline="") as fh:
                fh.write(batch_to_csv(res))
            j.log(render_batch_summary(res))
            j.log(f"Results: results_gui/{out}")
            return {"rows": res, "download": f"/download/gui/{out}"}
        job = jobs.start_call("batch-check", "Batch check", f"./run.sh check-batch {cli_equivalent([name])}", run)
        return job_response(job)

    @app.post("/api/tasks")
    def start_task():
        d = body()
        name = d.get("command")
        args = d.get("args") or {}
        return job_response(jobs.start_command(name, args))

    # ------------------------------------------------------------ jobs
    @app.get("/api/jobs")
    def list_jobs():
        return jsonify(jobs.list())

    @app.get("/api/jobs/<jid>")
    def get_job(jid):
        since = svc.validate_int(request.args.get("since", 0), "since", 0, 10**12)
        return jsonify(jobs.get(jid).snapshot(since))

    @app.post("/api/jobs/<jid>/cancel")
    def cancel_job(jid):
        return jsonify(jobs.cancel(jid).snapshot(10**12))

    # ------------------------------------------------------------ files
    @app.get("/api/files")
    def files():
        out = []
        for key, rel in DOWNLOAD_ROOTS.items():
            d = os.path.join(root, rel)
            if not os.path.isdir(d):
                continue
            for dirpath, _, names in os.walk(d):
                for n in sorted(names):
                    if os.path.splitext(n)[1] in DOWNLOAD_EXT:
                        p = os.path.join(dirpath, n)
                        sub = os.path.relpath(p, d)
                        out.append({"root": key, "path": sub, "label": f"{rel}/{sub}", "bytes": os.path.getsize(p),
                                    "mtime": os.path.getmtime(p), "url": f"/download/{key}/{sub}"})
        return jsonify(out)

    @app.get("/download/<key>/<path:sub>")
    def download(key, sub):
        if key not in DOWNLOAD_ROOTS:
            abort(404)
        base = os.path.realpath(os.path.join(root, DOWNLOAD_ROOTS[key]))
        p = os.path.realpath(os.path.join(base, sub))
        if not p.startswith(base + os.sep) or not os.path.isfile(p) or os.path.splitext(p)[1] not in DOWNLOAD_EXT:
            abort(404)
        return send_file(p, as_attachment=True, download_name=os.path.basename(p), mimetype="text/plain")

    if demo:
        svc.audit = lambda event: event            # a public demo keeps no shared log of visitors' actions
    return app


def _is_private_ip(host):
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private


def serve(root, host="127.0.0.1", port=8000, debug=False):
    app = create_app(root)
    shown = "localhost" if host in ("0.0.0.0", "127.0.0.1", "::") else host
    tok = os.environ.get("CM_WEB_TOKEN")
    print(f"ClusterMind web GUI on http://{shown}:{port}/" + (f"?token={tok}" if tok else ""), flush=True)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("Note: listening on a non-loopback interface. Keep it behind localhost port publishing "
              "or set CM_WEB_TOKEN.", flush=True)
    try:
        from waitress import serve as wserve
        wserve(app, host=host, port=port, threads=8, channel_timeout=120)
    except ImportError:
        app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)
