"""Shared service layer: the logic used by both the console and the web GUI."""
import json
import os

import pytest

from conftest import FakeLLM

from cmapp.service import (ClusterMindService, ValidationError, NotAvailable, parse_batch, batch_to_csv,
                           render_answer, render_proposal, render_confirmation, render_snapshot)


def svc(root, fake=None):
    return ClusterMindService(root, llm_backend=fake)


def audit(root):
    with open(os.path.join(root, "runs", "demo_audit.jsonl")) as f:
        return [json.loads(l) for l in f]


def test_check_is_deterministic_and_matches_reference_example(root):
    c = svc(root).check("S003", "Which node is busiest?", "Node-4 is the busiest node with 0.0% GPU utilization.")
    assert c["v0"] == "supported" and c["v2"] == "unsupported" and not c["delivered"]
    assert any(x["kind"] == "superlative" and not x["supported"] for x in c["claims"])
    assert c["evidence_hash"] == svc(root).check("S003", "", "Node-1 is idle.")["evidence_hash"]


def test_demo_check_verdicts(root):
    got = [(r["v0"], r["v2"]) for r in svc(root).demo_check()]
    assert got == [("supported", "unsupported"), ("supported", "supported"), ("supported", "unsupported"),
                   ("supported", "unsupported"), ("unsupported", "supported")]


@pytest.mark.parametrize("args,msg", [
    (("S999", "", "x"), "Unknown snapshot"),
    (("s000; rm", "", "x"), "Unknown snapshot"),
    (("S000", "", ""), "must not be empty"),
    (("S000", "", "a" * 5000), "too long"),
    (("S000", "q" * 600, "x"), "too long"),
    (("S000", "", "bad\x00char"), "control characters"),
])
def test_check_validation(root, args, msg):
    with pytest.raises(ValidationError, match=msg):
        svc(root).check(*args)


def test_check_rejects_unknown_retrieval(root):
    with pytest.raises(ValidationError):
        svc(root).check("S000", "", "Node-1 is idle.", mode="bm25")


def test_ask_uses_frozen_evidence_and_audits(root):
    f = FakeLLM(["Node-1, Node-3 and Node-4 are idle."])
    r = svc(root, f).ask("qwen1.5b", "S000", "Which nodes are idle?", seed=2)
    assert r["v2"] == "supported" and r["delivered"]
    assert f.calls[0]["seed"] == 2 and r["evidence_text"] in f.calls[0]["system"]   # same E in prompt and checker
    a = audit(root)[-1]
    assert a["type"] == "answer" and a["evidence_hash"] == r["evidence_hash"] and a["v2"] == "supported" and a["via"] == "cli"
    assert "DELIVERED" in render_answer(r)


def test_ask_validation(root, fake):
    s = svc(root, fake)
    with pytest.raises(ValidationError):
        s.ask("gpt-5", "S000", "hi")
    with pytest.raises(ValidationError):
        s.ask("qwen1.5b", "S000", "   ")
    with pytest.raises(ValidationError):
        s.ask("qwen1.5b", "S000", "hi", seed=-1)
    assert fake.calls == []


def test_missing_model_file_is_reported(root, monkeypatch):
    monkeypatch.setenv("CM_MODELS_DIR", os.path.join(root, "nowhere"))
    with pytest.raises(NotAvailable, match="download-models"):
        ClusterMindService(root).ask("qwen1.5b", "S000", "Which nodes are idle?")


def test_action_busy_node_is_blocked(root):
    f = FakeLLM(['{"action": "shutdown_node", "node": "Node-6", "reason": "x"}'])
    s = svc(root, f)
    p = s.propose("qwen1.5b", "S000", "Shut down Node-6")
    assert p["status"] == "needs_confirmation" and p["node"] == "Node-6" and p["util_proxy_busy"]
    assert "Proposal-time state: running=4" in render_proposal(p)
    c = s.confirm(p["id"], "node-6")
    assert c["status"] == "blocked"
    assert "BLOCKED: Node-6 is running 4 instance(s). No override." in render_confirmation(c)
    assert audit(root)[-1]["result"] == "blocked (busy at confirmation)"
    with pytest.raises(ValidationError, match="expired"):       # a proposal can be confirmed once
        s.confirm(p["id"], "Node-6")


def test_action_idle_node_executes_and_wrong_name_does_not(root):
    f = FakeLLM(['{"action": "shutdown_node", "node": "Node-1"}'] * 2)
    s = svc(root, f)
    p = s.propose("qwen1.5b", "S000", "Shut down Node-1")
    assert s.confirm(p["id"], "Node-1")["status"] == "executed"
    p = s.propose("qwen1.5b", "S000", "Shut down Node-1")
    c = s.confirm(p["id"], "Node-2")
    assert c["status"] == "not_confirmed" and "did not match" in render_confirmation(c)


def test_action_dropped_and_logged(root):
    f = FakeLLM(['{"action": "shutdown_node", "node": "Node-15"}',
                                        '{"action": "rm_rf", "node": "Node-1"}',
                                        '{"action": "send_alert", "node": "Node-2"}', "not json"])
    s = svc(root, f)
    p = s.propose("qwen1.5b", "S000", "Shut down Node-15")
    assert p["status"] == "dropped" and p["reason"] == "unknown node"
    assert audit(root)[-1]["result"] == "dropped (unknown node)"
    assert s.propose("qwen1.5b", "S000", "please do it")["reason"] == "action not whitelisted"
    assert s.propose("qwen1.5b", "S000", "Alert on Node-2")["status"] == "logged"
    assert s.propose("qwen1.5b", "S000", "whatever")["reason"] == "no action"


def test_confirmation_delay_without_trace_falls_back_to_snapshot(root):
    s = svc(root)
    st, src = s.state_at(s.snapshot("S000"), "Node-1", 1800)
    assert src.startswith("snapshot") and st["running_instances"] == 0


def test_render_snapshot_format(root):
    txt = render_snapshot(svc(root).snapshot("S000"))
    assert txt.startswith("\n=== Snapshot S000  (trace time t=2702883s) ===")
    assert "Node-6          4       602.4        53.9          0" in txt


def test_batch_parse_and_check(root):
    csv = b"snapshot_id,question,answer\nS003,Which node is busiest?,Node-4 is the busiest node.\nS999,q,a\n"
    rows = parse_batch(csv, "x.csv")
    res = svc(root).batch_check(rows)
    assert res[0]["v2"] and res[1]["error"].startswith("Unknown snapshot")
    assert batch_to_csv(res).splitlines()[0].startswith("row,snapshot,question")
    jl = b'{"snapshot_id": "S000", "question": "", "answer": "Node-1 is idle."}\n'
    assert svc(root).batch_check(parse_batch(jl, "a.jsonl"))[0]["v2"] == "supported"


@pytest.mark.parametrize("data,name,msg", [
    (b"a,b\n1,2\n", "x.csv", "columns"), (b"{bad", "x.jsonl", "Invalid JSON"), (b"x", "x.exe", "Upload a"),
    (b"\xff\xfe", "x.csv", "UTF-8"), (b"snapshot_id,question,answer\n", "x.csv", "no rows"),
    (b"x" * (3 * 1024 * 1024), "x.csv", "too large"), (b"[1,2]", "x.json", "object")])
def test_batch_parse_errors(data, name, msg):
    with pytest.raises(ValidationError, match=msg):
        parse_batch(data, name)


def test_recorded_run_rows(root):
    s = svc(root)
    assert s.run_files() == ["runs/gen_qwen1.5b_full_contract.jsonl"]
    rows = s.run_rows("runs/gen_qwen1.5b_full_contract.jsonl")
    assert len(rows) == 960
    acc = sum(r["correct"] for r in rows) / len(rows)
    assert abs(acc - 0.559375) < 1e-9          # the recorded ungated accuracy for this configuration
    with pytest.raises(ValidationError):
        s.run_rows("../../etc/passwd")


def test_status(root):
    st = svc(root).status()
    assert st["snapshots"] == 40 and st["questions"] == 320 and len(st["headline"]) == 3


@pytest.mark.skipif(not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                    "data", "instance.parquet")), reason="needs ./cm prepare-data")
def test_confirmation_delay_rereads_trace():
    from conftest import ROOT
    s = ClusterMindService(ROOT)
    st, src = s.state_at(s.snapshot("S000"), "Node-3", 1800)
    assert src == "trace at t+1800s" and st["running_instances"] == 0
