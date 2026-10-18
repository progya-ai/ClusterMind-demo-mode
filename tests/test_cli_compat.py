"""The console interface is unchanged: same commands, same arguments, same output."""
import os
import subprocess
import sys

from conftest import ROOT

ORIGINAL_COMMANDS = ["test", "tables", "demo-check", "prepare-data", "download-models", "demo", "generate",
                     "judge", "actions", "reproduce-all", "tables-rerun", "shell"]


def run(*argv):
    return subprocess.run(list(argv), cwd=ROOT, capture_output=True, text=True, timeout=120)


def test_help_lists_every_original_command_and_the_new_ones():
    out = run("bash", "cm", "help").stdout
    for c in ORIGINAL_COMMANDS + ["web", "check", "check-batch"]:
        assert f"   {c}" in out, c


def test_demo_check_output_is_byte_identical():
    with open(os.path.join(ROOT, "tests", "golden", "demo_check.txt")) as f:
        assert run("bash", "cm", "demo-check").stdout == f.read()


def test_demo_arguments_unchanged():
    out = run(sys.executable, "cm_demo.py", "--help").stdout
    for a in ("--model {qwen1.5b,llama1b,qwen7b}", "--snapshot", "--question", "--seed", "--confirm-delay", "--yes"):
        assert a in out


def test_unknown_command_still_fails():
    r = run("bash", "cm", "no-such-command")
    assert r.returncode == 1 and "unknown command" in r.stdout


def test_check_command_and_python_entry_point():
    r = run("bash", "cm", "check", "--snapshot", "S003", "--question", "Which node is busiest?",
            "--answer", "Node-4 is the busiest node with 0.0% GPU utilization.")
    assert r.returncode == 0 and "v2 verdict: UNSUPPORTED  -> SHOWN WITH WARNING" in r.stdout
    r = run(sys.executable, "-m", "cmapp", "check", "--snapshot", "S999", "--answer", "x")
    assert r.returncode == 2 and "Unknown snapshot" in r.stderr
    r = run(sys.executable, "-m", "cmapp", "cli", "help")
    assert r.returncode == 0 and "Web GUI" in r.stdout


def test_check_batch_command(tmp_path):
    f = tmp_path / "in.csv"
    f.write_text("snapshot_id,question,answer\nS000,,Node-1 is idle.\n")
    out = tmp_path / "out.csv"
    r = run("bash", "cm", "check-batch", str(f), "--out", str(out))
    assert r.returncode == 0 and out.read_text().splitlines()[1].split(",")[6] == "supported"
