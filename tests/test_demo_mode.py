"""Public demo mode (CM_DEMO=1): risky routes are closed, the keyword parser replaces the model,
and the real guard (typed confirmation + busy-node block) still decides."""
import os
import time

import pytest

from cmapp.web import DEMO_RATE_PER_MIN, create_app

H = {"X-CM-Request": "1"}


@pytest.fixture
def demo(root):
    app = create_app(root, demo=True)
    app.testing = True
    return app.test_client(), root


def wait(c, job):
    for _ in range(300):
        j = c.get(f"/api/jobs/{job['id']}").get_json()
        if j["status"] not in ("queued", "running"):
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def propose(c, text, sid="S000"):
    r = c.post("/api/actions/propose", json={"snapshot": sid, "instruction": text}, headers=H)
    assert r.status_code == 202, r.get_json()
    j = wait(c, r.get_json())
    assert j["status"] == "succeeded", j
    return j["result"]


def confirm(c, p, typed):
    r = c.post("/api/actions/confirm", json={"proposal_id": p["id"], "typed": typed, "delay": 900}, headers=H)
    assert r.status_code == 202, r.get_json()
    return wait(c, r.get_json())["result"]


def test_status_reports_demo_and_hides_server_details(demo):
    c, _ = demo
    st = c.get("/api/status").get_json()
    assert st["demo"] is True and st["models"] == [] and st["audit_entries"] == 0


@pytest.mark.parametrize("method,path", [("post", "/api/ask"), ("post", "/api/tasks"), ("post", "/api/batch-check"),
                                         ("get", "/api/files"), ("get", "/api/audit"), ("get", "/api/jobs"),
                                         ("get", "/api/commands"), ("get", "/download/results/e2_endtoend.json")])
def test_risky_routes_are_closed(demo, method, path):
    c, _ = demo
    r = getattr(c, method)(path, headers=H)
    assert r.status_code == 403 and "public demo" in r.get_json()["error"]


def test_any_host_is_accepted_in_demo(demo):
    c, _ = demo
    assert c.get("/", headers={"Host": "clustermind-demo.onrender.com"}).status_code == 200
    assert c.get("/healthz", headers={"Host": "clustermind-demo.onrender.com"}).status_code == 200


def test_busy_node_is_blocked_and_idle_node_executes(demo):
    c, _ = demo
    s = c.get("/api/snapshots/S000").get_json()
    busy = next(n["node"] for n in s["nodes"] if n["running_instances"] > 0)
    idle = next(n["node"] for n in s["nodes"] if n["running_instances"] == 0)
    p = propose(c, f"Shut down {busy}")
    assert p["status"] == "needs_confirmation" and p["node"] == busy and p["action"] == "shutdown_node"
    assert confirm(c, p, "wrong-name")["status"] == "not_confirmed"
    p = propose(c, f"Shut down {busy}")
    assert confirm(c, p, busy)["status"] == "blocked"
    p = propose(c, f"Please turn off {idle}")
    assert confirm(c, p, idle)["status"] == "executed"


def test_demo_confirmation_ignores_delay(demo):
    c, _ = demo
    p = propose(c, "Shut down Node-1")
    assert confirm(c, p, "Node-1")["delay_s"] == 0          # the request asked for 900 s; no trace is shipped


def test_non_disruptive_and_unknown_instructions(demo):
    c, _ = demo
    assert propose(c, "Send an alert about Node-2")["status"] == "logged"
    assert propose(c, "Shut down Node-99")["status"] == "dropped"
    assert propose(c, "What a lovely day")["status"] == "dropped"


def test_demo_writes_no_shared_audit_log(demo):
    c, root = demo
    propose(c, "Send an alert about Node-2")
    assert not os.path.exists(os.path.join(root, "runs", "demo_audit.jsonl"))


def test_check_still_works_and_posts_are_rate_limited(demo):
    c, _ = demo
    r = c.post("/api/check", json={"snapshot": "S000", "question": "Is Node-1 busy?", "answer": "Node-1 looks fine."}, headers=H)
    assert r.status_code == 200 and r.get_json()["result"]["v2"] == "unsupported"
    codes = [c.post("/api/actions/confirm", json={"proposal_id": "x"}, headers=H).status_code
             for _ in range(DEMO_RATE_PER_MIN + 5)]
    assert codes[-1] == 429 and 429 not in codes[:DEMO_RATE_PER_MIN - 3]
