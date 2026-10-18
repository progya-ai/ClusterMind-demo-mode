import os
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class FakeLLM:
    """Stands in for llama.cpp: returns scripted replies, records calls."""

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls = []

    def chat(self, model, system, user, seed=0, **kw):
        self.calls.append({"model": model, "system": system, "user": user, "seed": seed})
        text = self.replies.pop(0) if self.replies else "Node-1 is idle."
        return {"text": text, "latency_s": 0.01, "prompt_tokens": 1, "completion_tokens": 1}


@pytest.fixture
def root(tmp_path):
    """A throw-away project root: real data/ (read-only use), fresh runs/ with one recorded run."""
    (tmp_path / "data").mkdir()
    for f in os.listdir(os.path.join(ROOT, "data")):
        if f.endswith(".json"):                      # test-data files only (no trace parquet)
            os.symlink(os.path.join(ROOT, "data", f), tmp_path / "data" / f)
    (tmp_path / "runs").mkdir()
    shutil.copy(os.path.join(ROOT, "runs", "gen_qwen1.5b_full_contract.jsonl"), tmp_path / "runs")
    (tmp_path / "results").mkdir()
    shutil.copy(os.path.join(ROOT, "results", "e2_endtoend.json"), tmp_path / "results")
    return str(tmp_path)


@pytest.fixture
def fake():
    return FakeLLM()
