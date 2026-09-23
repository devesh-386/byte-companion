"""Base model vs base+persona adapter on prompts never used in training.

Checks what the adapter was for (identity, voice, brevity) AND what it must not break (tool calling).
Run:  .venv/Scripts/python.exe training/persona/eval_persona.py
Out:  artifacts/persona/eval.md
"""
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from companion import config  # noqa: E402
from companion.agent import Agent  # noqa: E402
from companion.hooks import EmotionHook  # noqa: E402
from companion.llm import LlamaClient  # noqa: E402
from companion.memory.embedder import Embedder  # noqa: E402
from companion.memory.hook import MemoryHook  # noqa: E402
from companion.plugins import Deps, load_plugins  # noqa: E402
from companion.memory.store import MemoryStore  # noqa: E402
from companion.ml.emotion import EmotionClassifier  # noqa: E402
from companion.persona import build_persona  # noqa: E402
from companion.protocol import build_system_prompt  # noqa: E402
from companion.server import LlamaServer  # noqa: E402
from companion.tools import ToolRegistry  # noqa: E402
from generate_data import BANNED, SEED_FACTS, read_conversations  # noqa: E402

PORT = 8766
EXPECTED_TOOL = {
    "what's 1234 times 56": "calculate",
    "what's the time": "get_current_time",
    "list what's in my documents folder": "list_directory",
    "open vs code": "launch_app",
    "search the web for the latest pytorch version": "web_search",
    "how much free disk space do I have": "system_info",
    "remind me in 15 minutes to stretch": "set_reminder",
}
IDENTITY = {"what are you exactly?", "is this chatgpt?"}


def run_all(lora: Path | None, prompts: list[str]) -> list[dict]:
    profile = config.active_profile()
    server = LlamaServer(config.LLAMA_SERVER, profile.gguf, config.HOST, PORT, profile.ctx,
                         config.GPU_LAYERS, config.LOG_DIR, lora=lora, profile=profile)
    server.log_path = config.LOG_DIR / f"eval-server-{'lora' if lora else 'base'}.log"
    server.start()
    try:
        store = MemoryStore(Path(tempfile.mkdtemp()) / "m.db", Embedder(config.EMBEDDER_DIR))
        for f in SEED_FACTS:
            store.add(f)
        reg = ToolRegistry(max_output_chars=config.TOOL_OUTPUT_MAX_CHARS, dry_run=True)
        load_plugins(reg, Deps(store=store))
        agent = Agent(LlamaClient(server.base_url), reg, build_system_prompt(build_persona(), reg.schemas()),
                      hooks=[EmotionHook(EmotionClassifier(config.EMOTION_DIR)), MemoryHook(store)],
                      confirm=lambda c, s: True)
        results = []
        for p in prompts:
            agent.reset()
            tools: list[str] = []
            answer = agent.run(p, on_event=lambda e: tools.append(e.data["name"]) if e.kind == "tool_call" else None)
            results.append({"prompt": p, "answer": answer, "tools": tools})
        return results
    finally:
        server.stop()


def score(results: list[dict]) -> dict:
    banned = sum(any(b in r["answer"].lower() for b in BANNED) for r in results)
    ident = [r for r in results if r["prompt"] in IDENTITY]
    ident_ok = sum("byte" in r["answer"].lower() and "alibaba" not in r["answer"].lower() for r in ident)
    tool_rows = [r for r in results if r["prompt"] in EXPECTED_TOOL]
    tool_ok = sum(EXPECTED_TOOL[r["prompt"]] in r["tools"] for r in tool_rows)
    chat = [r for r in results if r["prompt"] not in EXPECTED_TOOL]
    avg_len = sum(len(r["answer"]) for r in chat) / max(1, len(chat))
    questions = sum(r["answer"].rstrip().endswith("?") for r in chat)
    return {"identity": f"{ident_ok}/{len(ident)}", "tools": f"{tool_ok}/{len(tool_rows)}",
            "help-desk phrases": banned, "avg chat reply chars": round(avg_len),
            "chat replies ending in '?'": f"{questions}/{len(chat)}"}


def main() -> None:
    prompts = [c[0] for c in read_conversations(Path(__file__).resolve().parent / "heldout.txt")]
    lora = config.PERSONA_LORA
    if not lora.exists():
        sys.exit(f"No adapter at {lora}; run train_lora.py and export_lora.py first.")
    base = run_all(None, prompts)
    tuned = run_all(lora, prompts)
    sb, st = score(base), score(tuned)

    lines = ["# Persona adapter evaluation (held-out prompts)", "",
             "| metric | base | base + persona LoRA |", "|---|---|---|"]
    lines += [f"| {k} | {sb[k]} | {st[k]} |" for k in sb]
    lines += ["", "## Side by side", ""]
    for b, t in zip(base, tuned):
        clean = lambda s: re.sub(r"\s+", " ", s).strip()[:400]
        lines += [f"**{b['prompt']}**", "",
                  f"- base{(' [' + ', '.join(b['tools']) + ']') if b['tools'] else ''}: {clean(b['answer'])}",
                  f"- lora{(' [' + ', '.join(t['tools']) + ']') if t['tools'] else ''}: {clean(t['answer'])}", ""]
    out = config.PERSONA_LORA.parent / "eval.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:4 + len(sb)]))
    print(f"\nfull report: {out}")


if __name__ == "__main__":
    os.environ.pop("COMPANION_NO_LORA", None)
    main()
