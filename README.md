# ClusterMind: check before you trust, AI for GPU clusters

ClusterMind lets small local AI models answer questions about a GPU cluster and handle risky actions such as node shutdowns, with a fast rule-based checker that tests every answer against the same cluster data the operator sees. It has two front-ends over one code base:

* **Console mode.** The original commands (`./run.sh <command>` or `./cm <command>`), with their arguments and output unchanged.
* **Web GUI mode.** The same operations in a browser (`./run.sh web`), as a single-page interface with tabs.

Both modes call the same service layer (`cmapp/service.py`), which in turn calls the core logic in `cmbench/`. They read the same data, write the same audit log (`runs/demo_audit.jsonl`), and print results in the same format.

## Contents

| Path | What it is |
|---|---|
| `cmbench/` | Core logic: typed cluster state, frozen evidence, test questions, answer checkers (v0/v1/v1s/v2), scorer, action guards, prompts, local inference |
| `cmapp/service.py` | **Shared application layer**: ask, check, batch check, propose/confirm actions, snapshots, recorded runs, audit log, and text renderers used by both interfaces |
| `cmapp/tasks.py` | **Allowlisted command dispatcher** and background job manager (web tasks run the same `cm` commands) |
| `cmapp/web.py`, `cmapp/templates/`, `cmapp/static/` | Web GUI (Flask; plain HTML/CSS/JS, no CDN) |
| `cmapp/__main__.py` | `python -m cmapp web / cli / check / check-batch / demo-check` |
| `cm`, `run.sh` | Console runner inside Docker / Mac-side wrapper |
| `cm_demo.py` | Interactive console demo (now a thin front-end over `cmapp/service.py`) |
| `build_*.py`, `run_*.py`, `analyze*.py`, `eval_*.py`, `make_tables.py` | Test-data construction, model runs, analysis scripts |
| `tests/` | Checker tests, service tests, dispatcher tests, web API tests, and console-compatibility tests |

## Installation

**With Docker (recommended on macOS).** Docker Desktop → Settings → Resources: at least 4 CPUs and 8 GB memory.

```bash
cd ClusterMindNew/clustermind-app
chmod +x run.sh cm
cp .env.example .env        # optional: edit to reuse models you already downloaded (CM_MODELS_HOST)
./run.sh help               # first call builds the image (5-10 min: compiles llama.cpp)
```

**Without Docker** (Python 3.10+):

```bash
pip install -r requirements.txt
./cm help
```

Models (about 7 GB) come from `./run.sh download-models`, or from the web GUI's Tasks tab. To reuse a folder that already holds the GGUF files, set `CM_MODELS_HOST=../clustermind-docker/models` in `.env`.

## Public demo (hosted)

Setting `CM_DEMO=1` starts a read-mostly demo that needs no AI model: Dashboard, Check answers, Guarded action (a keyword
parser stands in for the model; the typed confirmation and busy-node block are the real code) and Recorded runs. Ask, Tasks,
Files and uploads are turned off. See [DEPLOY.md](DEPLOY.md) for hosting it on Render.

## Console mode

```bash
./run.sh test                                   # unit tests
./run.sh demo-check                             # v0 vs v2 verdicts on example answers (no model)
./run.sh tables                                 # recompute every table and number from runs/
./run.sh prepare-data                           # rebuild the test data from ../dataset/*.tar.gz, check hashes
./run.sh download-models [qwen1.5b llama1b qwen7b]
./run.sh demo [--model M] [--snapshot S012] [--confirm-delay 1800] [--question "..."] [--yes]
./run.sh generate llama1b full contract 0,1,2   # and: judge, actions, reproduce-all, tables-rerun, shell
```

These are new console commands; the web GUI exposes the same features:

```bash
./run.sh check --snapshot S003 --question "Which node is busiest?" --answer "Node-4 is the busiest node with 0.0% GPU utilization."
./run.sh check-batch answers.csv --out checked.csv     # CSV columns: snapshot_id,question,answer (or JSONL)
```

## Web GUI mode

```bash
./run.sh web          # then open http://localhost:8000   (Ctrl-C stops it)
./run.sh web -d       # run in the background;  ./run.sh web-stop  stops it
```

Without Docker, use `./cm web --host 127.0.0.1 --port 8000` or `python -m cmapp web --port 8000`. Any console command can also be run as `python -m cmapp cli <command> [args]`.

### What the interface offers

The page has a context bar (snapshot, model, seed) and eight tabs:

| Tab | What it does | Console equivalent |
|---|---|---|
| Dashboard | System status (models present, trace parquet, audit entries); the headline results table; the node table of the current snapshot (click a node to ask about it); the evidence text r(E) | `./run.sh tables` (headline), `demo` (snapshot table) |
| Ask | Question with sample questions; evidence (full / TF-IDF top-3); prompt style; shows the answer, the v2 verdict (delivered or shown with warning), v0, h(E), every checked claim, and the evidence | `./run.sh demo --question "..."` |
| Check answers | Checks a pasted answer without a model (v0, v1s, v2, claims, checker time); example checks; **batch upload** of a CSV/JSONL file with a filterable results table and a CSV download | `check`, `demo-check`, `check-batch` |
| Guarded action | Instruction → model proposal → whitelist/inventory → **typed node-name confirmation** → state re-read at confirmation (optional delay: 5 min to 1 h, re-derived from the trace) → busy nodes are hard-blocked | `./run.sh demo --question "act: ..."` |
| Recorded runs | Browse `runs/gen_*.jsonl` (the recorded model outputs), re-checked with v2 and scored against gold; filter by seed, template, verdict, correctness or text; accuracy / coverage / delivered accuracy of the filtered set; download | `analyze.py` |
| Tasks | Forms for the long commands: `test`, `tables`, `prepare-data`, `download-models`, `generate`, `judge`, `actions`, `reproduce-all`, `tables-rerun`. A *Recent tasks* list shows each task's status and duration, with Cancel, an expandable live log and a log download | `./run.sh <command>` |
| Files | Filterable list of `results/`, `results/tables/`, `runs/`, `results_gui/` with download buttons | — |
| Audit log | Searchable table of `runs/demo_audit.jsonl` (entries from both interfaces, marked `via: cli/web`), with a download button | — |

Results appear in cards on the page. On screens narrower than 640 px the layout adapts to a single narrow column.

Operations that need a model (Ask, Guarded action) and every Task run in the background, so the page stays responsive. Model calls run one at a time, because llama.cpp is single-context. Long tasks also run one at a time, since two 7B runs would exhaust Docker's memory.

### Example workflows

1. **See the checker at work without models.** Open Check answers and click *Run example checks*. Then click a row, change a number in the answer, and click *Check*.
2. **Ask and gate.** Pick snapshot S000 and the model `qwen1.5b`. On Ask, choose a sample question and click *Ask*. The result card shows the answer, the verdict, the evidence hash and each claim's result.
3. **Guarded shutdown.** On Guarded action, click the chip *Shut down Node-6* (a busy node) and then *Propose*. Type `Node-6` and confirm: the action is **BLOCKED**. Next, pick an idle node, set *Confirm* to *after 30 min*, and confirm. The state is re-read from the trace 30 minutes later. This needs `prepare-data`, which creates the trace parquet.
4. **Audit the headline numbers.** On Recorded runs, choose `runs/gen_llama1b_full_contract.jsonl` and leave the filters empty. Accuracy (38.5%) and v2 coverage (24.5%) match the headline table. Delivered accuracy here is pooled over the three runs (98.7%), whereas the headline table reports the mean over runs (98.5%). Filter by *v2 verdict = supported* and *correct = no* to read the few wrong answers that v2 still delivers.
5. **Reproduce.** Tasks → *Recompute result tables* → Run. When it finishes, download `results/tables/endtoend.tex` from Files.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CM_MODELS_HOST` (compose) | `./models` | Host folder with the GGUF files, mounted at `/app/models` |
| `CM_MODELS_DIR` | `models` | Model folder as seen by the code |
| `CM_THREADS` | all CPUs | llama.cpp threads |
| `CM_WEB_PORT` (compose) | `8000` | Port on your Mac (published on 127.0.0.1 only) |
| `CM_WEB_HOST` / `--host` | `127.0.0.1` | Interface the server listens on (the container uses 0.0.0.0 internally) |
| `CM_WEB_TOKEN` | empty | If set, every request needs the token: open `http://localhost:8000/?token=...` once and a cookie is set |
| `CM_WEB_ALLOWED_HOSTS` | empty | Extra `Host` header names to accept (localhost and IP addresses are always accepted) |
| `CM_RUNS_DIR`, `CM_RESULTS_DIR`, `CM_TABLES_DIR` | `runs`, `results`, `results/tables` | Output folders |

## Security considerations

- **No shell in the browser.** The web GUI can only start the commands listed in `cmapp/tasks.py:COMMANDS`.
  - Every argument is validated: model names against an allowlist; seeds, limits and item sets against strict patterns.
  - Each command becomes a fixed argument list, passed to `subprocess.Popen(..., shell=False)`. No string ever reaches a shell. Task logs in the GUI are output-only.
- **Actions cannot be forged.**
  - A confirmation refers to a proposal ID that the server issued and stored; the proposal is used once, and it expires after an hour.
  - The typed node name, the busy check and the state re-read all happen on the server. Execution is simulated.
- **Uploads.**
  - Size is capped at 2 MB and 2,000 rows.
  - Files must be UTF-8 CSV/JSONL/JSON and are parsed as data only.
  - The uploaded filename is never used on disk; result files get server-chosen names in `results_gui/`.
- **Downloads.** Only files under `results/`, `results_rerun/`, `results/tables/`, `runs/`, `runs_rerun/` and `results_gui/`, and only with the extensions `.json`, `.jsonl`, `.txt`, `.tex` and `.csv`. Paths are resolved and confined to those folders.
- **Network exposure.**
  - Docker publishes the port on `127.0.0.1` only.
  - The server rejects foreign `Host` headers, which protects against DNS rebinding.
  - POST requests must carry a custom header (`X-CM-Request`). A cross-site form cannot send one, so this blocks CSRF.
  - Responses carry a strict Content-Security-Policy (no inline or third-party scripts), `X-Frame-Options: DENY` and `nosniff`.
  - Set `CM_WEB_TOKEN` if the machine is shared.
- **No secrets are needed.** Models are local files and nothing is sent to external services. Model downloads use Hugging Face without a token.
- The server is `waitress` (pure Python). It is meant for local, single-user use and should not be exposed to the internet.

## Reproducing the results

See `README_DOCKER.md` for step-by-step timings. In short:

```bash
./run.sh tables            # all numbers from the shipped runs/ (minutes)
./run.sh prepare-data      # rebuild the test data from the Alibaba tarballs
./run.sh reproduce-all && ./run.sh tables-rerun   # re-run every model experiment (many hours)
```

Data: `pai_instance_table.tar.gz` and `pai_sensor_table.tar.gz` from https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020, placed in `../dataset/`. Models: `Qwen/Qwen2.5-1.5B-Instruct-GGUF` (q4_k_m), `bartowski/Llama-3.2-1B-Instruct-GGUF` (Q8_0), `bartowski/Qwen2.5-7B-Instruct-GGUF` (Q4_K_M); MD5 checksums are in `models/MD5SUMS`.

## Tests

```bash
./run.sh test        # or: python -m pytest -q tests
```

| File | What it covers |
|---|---|
| `test_checkers.py` | Checker behaviour |
| `test_service.py` | Shared logic: verdicts, validation, ask/propose/confirm with a fake model, audit, batch parsing, recorded runs; the ungated accuracy is recomputed exactly |
| `test_tasks.py` | Allowlist, argument validation (including injection attempts), streaming output, cancellation, one-long-task rule |
| `test_web.py` | Every API endpoint, the CSRF header, the Host check, the token, forged or expired confirmations, upload limits, path traversal on downloads |
| `test_cli_compat.py` | All original commands are still listed; `demo-check` output is byte-identical to the pre-GUI version; `cm_demo.py` arguments are unchanged; the new console commands work |
