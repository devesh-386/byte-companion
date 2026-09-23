import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Working name for the character; rename freely (retrain the persona adapter to match its voice).
NAME = "Byte"
USER_NAME = "Devesh"

LLAMA_SERVER = ROOT / "llama.cpp-bin" / "runtime" / "llama-server.exe"
PERSONA_LORA = ROOT / "artifacts" / "persona" / "persona-lora-f16.gguf"
EMOTION_DIR = ROOT / "artifacts" / "emotion_bge"   # v2 (fine-tuned bge-small); v1 from scratch: artifacts/emotion
EMBEDDER_DIR = ROOT / "models" / "bge-small-en-v1.5"
VOICE_MODEL = ROOT / "models" / "piper" / "en_US-lessac-medium.onnx"
VOICE_SPEED = 1.08
WHISPER_MODEL = "base.en"
WHISPER_DIR = ROOT / "models" / "whisper"
LOG_DIR = ROOT / "logs"
ACTION_LOG = LOG_DIR / "actions.jsonl"

# User data lives outside the code folder so reinstalling the code never wipes memories.
DATA_HOME = Path(os.environ.get("COMPANION_HOME", Path.home() / ".companion"))
MEMORY_DB = DATA_HOME / "memory.db"
TASKS_FILE = DATA_HOME / "tasks.yaml"

HOST = "127.0.0.1"
PORT = 8765
# The model file, context size, KV cache and vision settings live in companion/profiles.py.
GPU_LAYERS = 99

# Rough chars-per-token for English; used to keep history under the context window.
CHARS_PER_TOKEN = 3.5
HISTORY_BUDGET_TOKENS = 6000

MAX_AGENT_STEPS = 12  # UI tasks (observe, click, observe, type, save) need more than 6; Ctrl+Alt+Esc still stops it
TOOL_OUTPUT_MAX_CHARS = 4000
TOOLS_PER_TURN = 6   # tool retrieval: how many tools the model sees per message (0 = all of them)


# Set by the phase-2 bake-off (evals/results/bakeoff-20260923-180822.json): the 4B beat the 8B and 9B on
# tool calls (92% valid) and speed (83 tok/s) in 4.3 GB. See companion/profiles.py for the others.
DEFAULT_PROFILE = "qwen3.5-4b-byte"   # + Byte's adapter: 99% valid / 88% right tool vs 88% / 76% (bakeoff-20260923-204051)


def active_profile():
    from .profiles import PROFILES
    name = os.environ.get("COMPANION_MODEL", DEFAULT_PROFILE)
    if name not in PROFILES:
        raise ValueError(f"unknown model profile {name!r}; choose from {', '.join(PROFILES)}")
    return PROFILES[name]


def active_lora() -> Path | None:
    if os.environ.get("COMPANION_NO_LORA"):
        return None
    lora = active_profile().lora
    return lora if lora and lora.exists() else None
