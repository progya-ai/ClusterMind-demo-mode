"""Allowlisted command dispatcher and job manager."""
import sys

import pytest

from cmapp import tasks
from cmapp.service import ValidationError


def test_argv_is_a_fixed_list_for_each_command():
    assert tasks.build_argv("test", {}) == ["bash", "cm", "test"]
    assert tasks.build_argv("generate", {"model": "llama1b", "retrieval": "tfidf", "style": "original", "seeds": "0, 1,2", "limit": "10"}) == \
        ["bash", "cm", "generate", "llama1b", "tfidf", "original", "0,1,2", "10"]
    assert tasks.build_argv("judge", {"judge": "qwen7b", "itemset": "gen:llama1b_full_contract:0"}) == \
        ["bash", "cm", "judge", "qwen7b", "gen:llama1b_full_contract:0"]
    assert tasks.build_argv("download-models", {"models": ["qwen1.5b", "qwen1.5b", "llama1b"]}) == \
        ["bash", "cm", "download-models", "qwen1.5b", "llama1b"]
    assert tasks.build_argv("download-models", {}) == ["bash", "cm", "download-models"]


@pytest.mark.parametrize("name,args", [
    ("shell", {}), ("rm", {}), ("generate", {"model": "llama1b; rm -rf /"}),
    ("generate", {"model": "llama1b", "seeds": "0;id"}), ("generate", {"model": "llama1b", "seeds": "1,2,3,4,5,6"}),
    ("generate", {"model": "llama1b", "limit": "999"}), ("generate", {"model": "llama1b", "retrieval": "all"}),
    ("judge", {"judge": "qwen7b", "itemset": "gen:../../x:0"}), ("judge", {"judge": "qwen7b", "itemset": "$(id)"}),
    ("download-models", {"models": ["qwen1.5b", "--help"]}), ("actions", {"model": ""}), ("test", "notadict"),
])
def test_rejects_unknown_commands_and_bad_arguments(name, args):
    with pytest.raises(ValidationError):
        tasks.build_argv(name, args)


def test_cli_equivalent_quotes_for_display():
    assert tasks.cli_equivalent(["bash", "cm", "demo-check"]) == "./run.sh demo-check"
    assert tasks.cli_equivalent(["./run.sh", "demo", "--question", "it's idle?"]) == "./run.sh demo --question 'it'\\''s idle?'"


def test_subprocess_job_streams_output_and_can_be_cancelled(tmp_path, monkeypatch):
    monkeypatch.setitem(tasks.COMMANDS, "echo", ("Echo", "", tasks._v_none,
                        lambda a: [sys.executable, "-c", "print('hello'); print('world')"], False))
    monkeypatch.setitem(tasks.COMMANDS, "sleep", ("Sleep", "", tasks._v_none,
                        lambda a: [sys.executable, "-c", "import time; print('start', flush=True); time.sleep(30)"], True))
    jm = tasks.JobManager(str(tmp_path))
    j = jm.wait(jm.start_command("echo", {}).id, 20)
    snap = j.snapshot()
    assert snap["status"] == "succeeded" and "hello" in snap["lines"] and "world" in snap["lines"]
    assert j.snapshot(since=snap["next"])["lines"] == []           # incremental reads
    s = jm.start_command("sleep", {})
    with pytest.raises(ValidationError, match="Another long-running task"):
        jm.start_command("sleep", {})
    for _ in range(100):
        if s.status == "running" and s.proc is not None:
            break
        __import__("time").sleep(0.05)
    jm.cancel(s.id)
    assert jm.wait(s.id, 20).status == "cancelled"


def test_call_job_reports_validation_errors(tmp_path):
    jm = tasks.JobManager(str(tmp_path))

    def boom(job):
        raise ValidationError("bad input")
    j = jm.wait(jm.start_call("x", "X", "x", boom).id)
    assert j.status == "failed" and j.error == "bad input"
    j = jm.wait(jm.start_call("y", "Y", "y", lambda job: (job.log("hi"), 42)[1]).id)
    assert j.status == "succeeded" and j.result == 42 and "hi" in j.snapshot()["lines"]
