"""Web API: routes call the shared service; security checks."""
import io
import time

import pytest

from conftest import FakeLLM
from cmapp.service import ClusterMindService
from cmapp.web import create_app

H = {"X-CM-Request": "1"}


def make(root, replies=None, token=""):
    fake = FakeLLM(replies)
    app = create_app(root, service=ClusterMindService(root, llm_backend=fake, interface="web"), token=token)
    app.testing = True
    return app.test_client(), fake


def wait(c, job):
    for _ in range(200):
        j = c.get(f"/api/jobs/{job['id']}").get_json()
        if j["status"] not in ("queued", "running"):
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_index_and_security_headers(root):
    c, _ = make(root)
    r = c.get("/")
    assert r.status_code == 200 and b"ClusterMind" in r.data and b'id="jobs-list"' in r.data and b'id="con"' not in r.data
    assert "default-src 'self'" in r.headers["Content-Security-Policy"] and r.headers["X-Frame-Options"] == "DENY"
    assert c.get("/static/app.js").status_code == 200


def test_post_requires_header_and_json(root):
    c, _ = make(root)
    assert c.post("/api/check", json={"snapshot": "S000", "answer": "x"}).status_code == 403
    assert c.post("/api/check", data="nope", headers=H).status_code == 400


def test_foreign_host_rejected(root):
    c, _ = make(root)
    assert c.get("/api/status", headers={"Host": "evil.example"}).status_code == 400
    assert c.get("/api/status", headers={"Host": "localhost:8000"}).status_code == 200


def test_token(root):
    c, _ = make(root, token="s3cret")
    assert c.get("/api/status").status_code == 401
    assert c.get("/api/status", headers={"X-CM-Token": "s3cret"}).status_code == 200
    r = c.get("/?token=s3cret")
    assert r.status_code == 200 and "cm_token=s3cret" in r.headers.get("Set-Cookie", "")


def test_status_snapshots(root):
    c, _ = make(root)
    st = c.get("/api/status").get_json()
    assert st["snapshots"] == 40 and len(st["models"]) == 3
    s = c.get("/api/snapshots/S000").get_json()
    assert len(s["nodes"]) == 8 and s["questions"] and s["text"].startswith("\n=== Snapshot S000")
    assert c.get("/api/snapshots/S999").status_code == 400


def test_check_endpoint(root):
    c, _ = make(root)
    r = c.post("/api/check", json={"snapshot": "S003", "question": "Which node is busiest?",
                                   "answer": "Node-4 is the busiest node with 0.0% GPU utilization."}, headers=H)
    j = r.get_json()
    assert r.status_code == 200 and j["status"] == "succeeded" and j["result"]["v2"] == "unsupported"
    assert j["command"].startswith("./run.sh check --snapshot S003")
    assert any("SHOWN WITH WARNING" in l for l in j["lines"])
    bad = c.post("/api/check", json={"snapshot": "S003", "answer": ""}, headers=H)
    assert bad.status_code == 400 and "empty" in bad.get_json()["error"]


def test_ask_job(root):
    c, fake = make(root, ["Node-6 is the busiest node with 602.4% GPU utilization."])
    r = c.post("/api/ask", json={"model": "qwen1.5b", "snapshot": "S000", "seed": 1,
                                 "question": "Which node is the busiest?"}, headers=H)
    assert r.status_code == 202
    j = wait(c, r.get_json())
    assert j["status"] == "succeeded" and j["result"]["v2"] == "supported" and fake.calls[0]["seed"] == 1
    assert j["command"] == "./run.sh demo --model qwen1.5b --snapshot S000 --seed 1 --question 'Which node is the busiest?'"
    audit = c.get("/api/audit?q=busiest").get_json()
    assert audit and audit[0]["via"] == "web"
    for bad in ({"model": "x", "snapshot": "S000", "question": "q"}, {"model": "qwen1.5b", "snapshot": "S000", "question": ""},
                {"model": "qwen1.5b", "snapshot": "S000", "question": "q", "seed": "abc"},
                {"model": "qwen1.5b", "snapshot": "S000", "question": "q", "retrieval": "x"}):
        assert c.post("/api/ask", json=bad, headers=H).status_code == 400


def test_action_propose_confirm(root):
    c, _ = make(root, ['{"action": "shutdown_node", "node": "Node-6"}'])
    j = wait(c, c.post("/api/actions/propose", json={"model": "qwen1.5b", "snapshot": "S000",
                                                     "instruction": "Shut down Node-6"}, headers=H).get_json())
    p = j["result"]
    assert p["status"] == "needs_confirmation"
    # a client cannot confirm something the server did not propose
    assert c.post("/api/actions/confirm", json={"proposal_id": "forged", "typed": "Node-6"}, headers=H).status_code == 400
    assert c.post("/api/actions/confirm", json={"proposal_id": p["id"], "typed": "Node-6", "delay": "-5"},
                  headers=H).status_code == 400
    k = wait(c, c.post("/api/actions/confirm", json={"proposal_id": p["id"], "typed": "Node-6", "delay": 0}, headers=H).get_json())
    assert k["result"]["status"] == "blocked" and any(l.startswith("BLOCKED") for l in k["lines"])
    assert c.post("/api/actions/confirm", json={"proposal_id": p["id"], "typed": "Node-6"}, headers=H).status_code == 400


def test_batch_upload_and_download(root):
    c, _ = make(root)
    data = {"file": (io.BytesIO(b"snapshot_id,question,answer\nS000,,Node-1 is idle.\n"), "../../evil.csv")}
    r = c.post("/api/batch-check", data=data, headers=H, content_type="multipart/form-data")
    j = wait(c, r.get_json())
    assert j["status"] == "succeeded" and j["result"]["rows"][0]["v2"] == "supported"
    d = c.get(j["result"]["download"])
    assert d.status_code == 200 and d.data.startswith(b"row,snapshot")
    bad = c.post("/api/batch-check", data={"file": (io.BytesIO(b"x"), "a.exe")}, headers=H, content_type="multipart/form-data")
    assert bad.status_code == 400
    big = c.post("/api/batch-check", data={"file": (io.BytesIO(b"x" * (3 * 1024 * 1024)), "a.csv")}, headers=H,
                 content_type="multipart/form-data")
    assert big.status_code == 413


def test_downloads_are_confined(root):
    c, _ = make(root)
    files = c.get("/api/files").get_json()
    assert any(f["label"] == "results/e2_endtoend.json" for f in files)
    assert c.get("/download/results/e2_endtoend.json").status_code == 200
    for url in ("/download/results/../data/snapshots.json", "/download/results/%2e%2e/%2e%2e/etc/passwd",
                "/download/etc/passwd", "/download/runs/../../conftest.py"):
        assert c.get(url).status_code == 404


def test_tasks_api(root):
    c, _ = make(root)
    cmds = {x["name"] for x in c.get("/api/commands").get_json()}
    assert {"test", "tables", "generate", "judge", "download-models"} <= cmds and "shell" not in cmds
    assert c.post("/api/tasks", json={"command": "shell", "args": {}}, headers=H).status_code == 400
    assert c.post("/api/tasks", json={"command": "generate", "args": {"model": "x"}}, headers=H).status_code == 400
    assert c.get("/api/jobs/doesnotexist").status_code == 400


def test_runs_api(root):
    c, _ = make(root)
    runs = c.get("/api/runs").get_json()
    rows = c.get(f"/api/runs/rows?run={runs[0]}").get_json()
    assert len(rows) == 960 and {"qid", "v2", "correct"} <= set(rows[0])
    assert c.get("/api/runs/rows?run=../x").status_code == 400
