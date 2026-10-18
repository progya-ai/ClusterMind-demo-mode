"""Model file locations (no heavy imports, so the web GUI and the checkers
work even where llama-cpp-python is not installed)."""
import os

MODEL_FILES = {
    "qwen1.5b": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "llama1b": "Llama-3.2-1B-Instruct-Q8_0.gguf",
    "qwen7b": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
}


def models_dir():
    return os.environ.get("CM_MODELS_DIR", "models")


def model_paths():
    d = models_dir()
    return {k: os.path.join(d, f) for k, f in MODEL_FILES.items()}
