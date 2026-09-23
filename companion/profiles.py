"""One ModelProfile per brain Byte can run on. The bake-off and the app both start the server from these,
so what gets measured is exactly what runs.

Pick one with the COMPANION_MODEL environment variable, or change DEFAULT_PROFILE in config.py.
Big model files live on D: (D:\\byte-models); the small original 3B stays in the repo.
"""
from dataclasses import dataclass, field
from pathlib import Path

D_MODELS = Path("D:/byte-models")
ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelProfile:
    name: str
    gguf: Path
    mmproj: Path | None = None       # vision projector; None = text only
    ctx: int = 16384                 # shared by the server's slots (chat + memory extraction)
    kv_type: str = "q8_0"            # quantised KV cache halves its VRAM (P3)
    lora: Path | None = None         # persona adapter trained for THIS base model only
    thinking: bool = False           # D5: off by default
    image_max_tokens: int = 1024     # caps what one screenshot costs (P1)
    context_in_user: bool = True     # D3 layout; the old 3B adapter was trained with context in the system prompt
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @property
    def vision(self) -> bool:
        return self.mmproj is not None

    def available(self) -> bool:
        return (self.gguf.exists() and (self.mmproj is None or self.mmproj.exists())
                and (self.lora is None or self.lora.exists()))

    def server_args(self) -> list[str]:
        """Everything after `-m model --host --port -ngl`, i.e. what differs between models."""
        args = ["-c", str(self.ctx)]
        if self.kv_type != "f16":
            args += ["-fa", "on", "-ctk", self.kv_type, "-ctv", self.kv_type]  # quantised V needs flash attention
        if self.mmproj:
            args += ["--mmproj", str(self.mmproj), "--image-max-tokens", str(self.image_max_tokens)]
        else:
            args += ["--no-mmproj"]
        args += ["--reasoning", "on" if self.thinking else "off"]
        return args + list(self.extra_args)


PROFILES: dict[str, ModelProfile] = {p.name: p for p in (
    ModelProfile("qwen2.5-3b", ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf", kv_type="f16",
                 lora=ROOT / "artifacts" / "persona" / "persona-lora-f16.gguf", context_in_user=False),
    ModelProfile("qwen3-vl-8b", D_MODELS / "Qwen3-VL-8B-Instruct-GGUF" / "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
                 mmproj=D_MODELS / "Qwen3-VL-8B-Instruct-GGUF" / "mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf"),
    ModelProfile("qwen3.5-9b", D_MODELS / "Qwen3.5-9B-GGUF" / "Qwen3.5-9B-Q4_K_M.gguf",
                 mmproj=D_MODELS / "Qwen3.5-9B-GGUF" / "mmproj-F16.gguf"),
    ModelProfile("qwen3.5-4b", D_MODELS / "Qwen3.5-4B-GGUF" / "Qwen3.5-4B-Q4_K_M.gguf",
                 mmproj=D_MODELS / "Qwen3.5-4B-GGUF" / "mmproj-F16.gguf"),
    # The 4B plus Byte's own adapter (personality + tool use), trained by training/agent + train_lora.py.
    ModelProfile("qwen3.5-4b-byte", D_MODELS / "Qwen3.5-4B-GGUF" / "Qwen3.5-4B-Q4_K_M.gguf",
                 mmproj=D_MODELS / "Qwen3.5-4B-GGUF" / "mmproj-F16.gguf",
                 lora=ROOT / "artifacts" / "agent_4b" / "byte-4b-lora-f16.gguf"),
)}
