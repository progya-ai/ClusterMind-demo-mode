"""ClusterMind application entry point.

  python -m cmapp web [--host 127.0.0.1] [--port 8000]     # browser GUI
  python -m cmapp cli <command> [args]                      # any ./cm console command (same as ./cm <command>)
  python -m cmapp demo-check                                # checkers on the example answers
  python -m cmapp check --snapshot S003 --question Q --answer A [--json]
  python -m cmapp check-batch FILE.csv|.jsonl [--out results.csv]
"""
import argparse
import json
import os
import subprocess
import sys

from .service import (ClusterMindService, ValidationError, batch_to_csv, parse_batch, render_batch_summary,
                      render_check, render_demo_check, retrieve, DEMO_CHECK_SNAPSHOT, RETRIEVAL_MODES)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cmd_demo_check(a):
    svc = ClusterMindService(ROOT)
    ev = retrieve(svc.snapshot(DEMO_CHECK_SNAPSHOT), "", "full")
    sys.stdout.write(render_demo_check(svc.demo_check(), ev.text))
    return 0


def cmd_check(a):
    svc = ClusterMindService(ROOT)
    try:
        c = svc.check(a.snapshot, a.question or "", a.answer, a.retrieval)
    except ValidationError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(c, indent=1) if a.json else render_check(c))
    return 0


def cmd_check_batch(a):
    svc = ClusterMindService(ROOT)
    try:
        with open(a.file, "rb") as f:
            rows = parse_batch(f.read(), a.file)
        res = svc.batch_check(rows)
    except (OSError, ValidationError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    out = batch_to_csv(res)
    if a.out:
        with open(a.out, "w", newline="") as f:
            f.write(out)
        print(f"wrote {a.out}")
    else:
        sys.stdout.write(out)
    print(render_batch_summary(res), file=sys.stderr)
    return 0


def cmd_cli(a):
    # Structured argv, no shell: forwards to the existing console runner.
    return subprocess.call(["bash", os.path.join(ROOT, "cm"), *a.args], cwd=ROOT)


def cmd_web(a):
    from .web import serve
    serve(ROOT, host=a.host, port=a.port, debug=a.debug)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cmapp", description="ClusterMind console and web interfaces")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("web", help="start the web GUI")
    w.add_argument("--host", default=os.environ.get("CM_WEB_HOST", "127.0.0.1"))
    w.add_argument("--port", type=int, default=int(os.environ.get("CM_WEB_PORT", "8000")))
    w.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)
    w.set_defaults(fn=cmd_web)
    c = sub.add_parser("cli", help="run a console command (same as ./cm <command>)")
    c.add_argument("args", nargs=argparse.REMAINDER)
    c.set_defaults(fn=cmd_cli)
    d = sub.add_parser("demo-check", help="run the checkers on the example answers")
    d.set_defaults(fn=cmd_demo_check)
    k = sub.add_parser("check", help="check one answer against a snapshot (no model)")
    k.add_argument("--snapshot", required=True)
    k.add_argument("--question", default="")
    k.add_argument("--answer", required=True)
    k.add_argument("--retrieval", default="full", choices=RETRIEVAL_MODES)
    k.add_argument("--json", action="store_true")
    k.set_defaults(fn=cmd_check)
    b = sub.add_parser("check-batch", help="check a CSV/JSONL file of answers (no model)")
    b.add_argument("file")
    b.add_argument("--out")
    b.set_defaults(fn=cmd_check_batch)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
