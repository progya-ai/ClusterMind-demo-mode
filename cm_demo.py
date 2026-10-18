"""ClusterMind interactive demo (CPU, local GGUF model).

Ask questions about a real cluster snapshot and see the evidence, the
model's answer, and the deterministic verdicts (v0 = original prototype, v2 = new checker).
Prefix an instruction with "act:" to go through the guarded action path.

  python cm_demo.py                      # interactive, qwen1.5b, snapshot S000
  python cm_demo.py --model llama1b --snapshot S012
  python cm_demo.py --question "Which nodes are idle?"          # one-shot
  python cm_demo.py --question "act: Shut down Node-3" --yes     # one-shot action (auto-types node name)

The logic lives in cmapp/service.py, which the web GUI (./cm web) uses too;
this file only handles console input and output.
"""
import argparse, json

from cmapp.service import (ClusterMindService, ValidationError, NotAvailable, render_snapshot, render_answer,
                           render_proposal, render_confirmation, MODEL_NAMES)

SVC = ClusterMindService(".", interface="cli")
SNAPS = SVC.snapshots()
AUDIT = SVC.audit_path


def show_snapshot(s):
    print(render_snapshot(s))


def ask(model, s, question, seed):
    print(render_answer(SVC.ask(model, s["snapshot_id"], question, seed=seed)))


def act(model, s, instruction, seed, delay, auto_yes):
    p = SVC.propose(model, s["snapshot_id"], instruction, seed=seed)
    print(render_proposal(p))
    if p["status"] != "needs_confirmation":
        return
    node = p["node"]
    typed = node if auto_yes else input(f"Type the node name to confirm ({node}): ")
    print(render_confirmation(SVC.confirm(p["id"], typed, delay)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen1.5b", choices=list(MODEL_NAMES))
    ap.add_argument("--snapshot", default="S000", choices=sorted(SNAPS))
    ap.add_argument("--question")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confirm-delay", type=int, default=0,
                    help="seconds between proposal and confirmation (re-reads trace state if parquet present)")
    ap.add_argument("--yes", action="store_true", help="auto-type the node name (non-interactive)")
    a = ap.parse_args()
    s = SNAPS[a.snapshot]
    show_snapshot(s)
    def handle(q):
        try:
            if q.lower().startswith("act:"):
                act(a.model, s, q[4:].strip(), a.seed, a.confirm_delay, a.yes)
            else:
                ask(a.model, s, q, a.seed)
        except (ValidationError, NotAvailable) as e:
            print(f"Error: {e}")
    if a.question:
        handle(a.question); return
    print("\nAsk a question, 'act: <instruction>', 'snap <ID>' to switch snapshot, or 'quit'.")
    while True:
        try:
            q = input("\nclustermind> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("quit", "exit"):
            break
        if q.startswith("snap "):
            sid = q.split()[1].upper()
            if sid in SNAPS:
                s = SNAPS[sid]; show_snapshot(s)
            else:
                print("unknown snapshot; choose S000..S039")
            continue
        handle(q)


if __name__ == "__main__":
    main()
