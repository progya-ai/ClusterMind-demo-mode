"""Local GGUF inference via llama-cpp-python (CPU). Mirrors the original
ClusterMind Ollama settings: temperature 0.3, top_p 0.9, max 180 tokens."""
import os
import time

from cmbench.llm_paths import model_paths, models_dir

MODELS_DIR = models_dir()
N_THREADS = int(os.environ.get("CM_THREADS", "0")) or max(1, (os.cpu_count() or 2))
MODELS = model_paths()
# The recorded runs used n_threads=2 on a 2-vCPU machine. Thread count does not
# change greedy/seeded outputs in principle, but floating-point reduction order
# can differ across CPUs, so regenerated text may differ slightly from runs/.

_cache = {}


def get(name, n_ctx=2048):
    if not os.path.exists(MODELS[name]):
        raise FileNotFoundError(f"{MODELS[name]} not found. Run: ./cm download-models")
    if name not in _cache:
        from llama_cpp import Llama  # imported here so that importing this module is cheap
        _cache.clear()
        _cache[name] = Llama(model_path=MODELS[name], n_ctx=n_ctx, n_threads=N_THREADS,
                             n_batch=256, verbose=False, seed=0, logits_all=False)
    return _cache[name]


def chat(name, system, user, seed=0, temperature=0.3, top_p=0.9, max_tokens=180, logprobs=None):
    m = get(name)
    t = time.time()
    r = m.create_chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed,
        logprobs=logprobs is not None, top_logprobs=logprobs)
    dt = time.time() - t
    ch = r["choices"][0]
    return {"text": ch["message"]["content"].strip(), "latency_s": dt,
            "prompt_tokens": r["usage"]["prompt_tokens"], "completion_tokens": r["usage"]["completion_tokens"],
            "logprobs": ch.get("logprobs")}
