"""Convert a trained PEFT adapter to GGUF so llama-server can load it with --lora.
Uses llama.cpp's own converter from llama.cpp-src/ (same build as our binaries).

    python training/persona/export_lora.py                       the old 3B persona adapter
    python training/persona/export_lora.py --adapter artifacts/agent_4b/adapter \
        --base D:/byte-models/Qwen3.5-4B-hf --outfile artifacts/agent_4b/byte-4b-lora-f16.gguf
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(ROOT / "artifacts" / "persona" / "adapter"))
    ap.add_argument("--base", default=str(ROOT / "models" / "Qwen2.5-3B-Instruct"))
    ap.add_argument("--outfile", default=str(ROOT / "artifacts" / "persona" / "persona-lora-f16.gguf"))
    args = ap.parse_args()
    out = Path(args.outfile)
    cmd = [sys.executable, str(ROOT / "llama.cpp-src" / "convert_lora_to_gguf.py"), args.adapter,
           "--base", args.base, "--outtype", "f16", "--outfile", str(out)]
    print(" ".join(cmd))
    code = subprocess.call(cmd)
    if code == 0:
        print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MB)")
    return code


if __name__ == "__main__":
    sys.exit(main())
