"""Download the three GGUF models used by ClusterMind into $CM_MODELS_DIR (default ./models) and verify MD5."""
import hashlib, os, sys
from huggingface_hub import hf_hub_download

MODELS = [
    ("qwen1.5b", "Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    ("llama1b", "bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf"),
    ("qwen7b", "bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
]
d = os.environ.get("CM_MODELS_DIR", "models")
want = sys.argv[1:] or [m[0] for m in MODELS]
ref = {}
if os.path.exists(os.path.join(d, "MD5SUMS")):
    for line in open(os.path.join(d, "MD5SUMS")):
        h, f = line.split()
        ref[os.path.basename(f)] = h
for key, repo, fn in MODELS:
    if key not in want:
        continue
    path = os.path.join(d, fn)
    if not os.path.exists(path):
        print(f"downloading {repo}/{fn} ...", flush=True)
        hf_hub_download(repo, fn, local_dir=d)
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
    ok = ref.get(fn) in (None, md5.hexdigest())
    print(f"{key:9s} {fn}  md5={md5.hexdigest()}  {'OK' if ok else 'MISMATCH with MD5SUMS'}")
