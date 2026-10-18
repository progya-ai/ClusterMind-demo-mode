# ClusterMind in Docker (macOS, Intel or Apple Silicon)

Everything runs on CPU inside one container. This folder is mounted into the
container at `/app`, so all outputs (runs/, results/, results/tables/) appear here on your Mac.

## 0. One-time setup
Docker Desktop → Settings → Resources: give Docker **at least 4 CPUs and 8 GB memory**
(the 7B judge needs about 6 GB; the 1–1.5B models need about 2 GB).

```bash
cd "ClusterMindNew/clustermind-app"
chmod +x run.sh cm
./run.sh help            # first call builds the image (5–10 min: compiles llama.cpp)
```

## 1. Quick checks (no models, no raw data; ~2 minutes)
```bash
./run.sh test            # checker unit tests
./run.sh demo-check      # v0 (prototype) vs v2 (new checker) verdicts on example answers
./run.sh tables          # recompute every table and number from runs/
```
`tables` rewrites `results/tables/*.tex` and `results/macros.tex`; they should be unchanged except the checker-latency figure (`\RcheckUs`, microseconds), which depends on your CPU.

## 2. Rebuild the test data from the Alibaba tarballs (~5–10 minutes)
The tarballs are read from `../dataset/` (already in your ClusterMindNew folder).
```bash
./run.sh prepare-data    # extracts the 366-machine subset, rebuilds the test data, checks SHA-256 against the reference copies
```
All six files should print `OK`. This also enables the “re-read live state” analysis in `tables`
and real trace drift in the demo.

## 3. Download the models (~7 GB)
```bash
./run.sh download-models              # all three
./run.sh download-models qwen1.5b     # just the primary model (1.1 GB) for the demo
```

## 4. Interactive demo
```bash
./run.sh demo                                    # Qwen2.5-1.5B, snapshot S000
./run.sh demo --model llama1b --snapshot S012
./run.sh demo --confirm-delay 1800               # confirmation 30 min after the proposal (state re-read from the trace)
```
At the prompt:
* `Which nodes are idle?` → answer, evidence hash, v2 and v0 verdicts, and every checked claim
* `act: Shut down Node-3` → model proposal → whitelist/inventory → type the node name → busy-node check at confirmation → blocked or (simulated) executed
* `snap S021` to switch snapshot, `quit` to exit. Everything is appended to `runs/demo_audit.jsonl`.

## 4b. Web GUI
```bash
./run.sh web                                     # http://localhost:8000  (Ctrl-C stops it)
./run.sh web -d  /  ./run.sh web-stop            # background / stop
```
The page covers Dashboard, Ask, Check answers
(including batch upload), Guarded action, Recorded runs, Tasks (every long command below), Files and the Audit log.
See README.md for details and security notes. If your models are in another folder, put
`CM_MODELS_HOST=../clustermind-docker/models` in `.env`.

## 5. Re-run the experiments (long)
```bash
./run.sh generate llama1b full contract 0        # one configuration (~20–40 min on a laptop)
./run.sh reproduce-all                           # everything, into runs_rerun/ (many hours; resumable)
./run.sh tables-rerun                            # analyse runs_rerun/ into results_rerun/
```
The shipped `runs/` (the recorded model outputs) is never overwritten by `reproduce-all`.
Sampling with the same seeds on a different CPU can give slightly different text, so re-run
numbers may differ a little; the headline numbers come from `runs/`.

## Useful
* `./run.sh shell` – a shell inside the container.
* `CM_THREADS=4 ./run.sh demo` – limit CPU threads.
