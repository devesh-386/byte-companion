"""Build the personality training set by context distillation.

A "teacher" (base model + the long character bible in character.md) answers every prompt through the
real agent pipeline: real emotion model, real memory, real read-only tools, sandboxed action tools.
Every LLM call is recorded, and its system prompt is swapped for the SHORT production one. Training on
that teaches the adapter to behave like the teacher while only paying for the short prompt.

Run:  .venv/Scripts/python.exe training/persona/generate_data.py
Out:  data/persona/train.jsonl, data/persona/rejected.jsonl
"""
import json
import os
import random
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

os.environ["COMPANION_NO_LORA"] = "1"  # the teacher is always the plain base model
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from companion import config  # noqa: E402
from companion.agent import Agent  # noqa: E402
from companion.app import make_server  # noqa: E402
from companion.hooks import EmotionHook  # noqa: E402
from companion.llm import LlamaClient  # noqa: E402
from companion.memory.embedder import Embedder  # noqa: E402
from companion.memory.hook import MemoryHook  # noqa: E402
from companion.plugins import Deps, load_plugins  # noqa: E402
from companion.memory.store import MemoryStore  # noqa: E402
from companion.ml.emotion import EmotionClassifier  # noqa: E402
from companion.persona import RULES, build_persona  # noqa: E402
from companion.protocol import build_system_prompt, parse_tool_calls  # noqa: E402
from companion.tools import ToolRegistry  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = ROOT / "data" / "persona"
SEED_FACTS = [
    "The user's name is Devesh.",
    "The user is a computer science student at SRM.",
    "The user is building an AI companion called Byte as a learning project.",
    "The user's laptop has an RTX 5050 GPU.",
]
GOLD_REPEATS = 2

BANNED = [
    "alibaba", "language model", "as an ai", "i'm just a program", "how can i assist", "how may i assist",
    "i hope this helps", "feel free to", "is there anything else", "certainly!", "absolutely!",
    "<tool_response>", "i am an ai", "i'm an ai assistant", "created by alibaba",
]
_CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")


def teacher_persona() -> str:
    bible = (HERE / "character.md").read_text(encoding="utf-8")
    bible = bible.replace("{name}", config.NAME).replace("{user}", config.USER_NAME)
    today = datetime.now().strftime("%A %d %B %Y")
    return f"{bible}\n\n{RULES}\n\nToday is {today}."


def reject_reason(output: str) -> str | None:
    text = output.strip()
    if not text:
        return "empty"
    low = text.lower()
    for phrase in BANNED:
        if phrase in low:
            return f"banned phrase: {phrase}"
    if "qwen" in low and "byte" not in low:
        return "calls itself qwen"
    if _CJK.search(text):
        return "non-english script"
    if _EMOJI.search(text):
        return "emoji"
    calls, errors = parse_tool_calls(text)
    if errors:
        return "malformed tool call"
    if not calls and len(text) > 1100:
        return "too long"
    return None


class RecordingLLM:
    """Wraps the real client; remembers every (messages, output) pair the agent produces."""

    def __init__(self, llm: LlamaClient, teacher_base: str, production_base: str):
        self.llm = llm
        self.teacher_base = teacher_base
        self.production_base = production_base
        self.records: list[dict] = []

    def chat(self, messages, **kw):
        out = self.llm.chat(messages, **kw)
        system = messages[0]["content"]
        assert system.startswith(self.teacher_base)
        swapped = [{"role": "system", "content": self.production_base + system[len(self.teacher_base):]}]
        swapped += [dict(m) for m in messages[1:]]
        self.records.append({"messages": swapped + [{"role": "assistant", "content": out}]})
        return out


def read_conversations(path: Path) -> list[list[str]]:
    convs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            convs.append([t.strip() for t in line.split("||") if t.strip()])
    return convs


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N conversations (smoke test)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(7)
    server = make_server()
    server.start()
    try:
        tmp = Path(tempfile.mkdtemp(prefix="persona-gen-"))
        store = MemoryStore(tmp / "memory.db", Embedder(config.EMBEDDER_DIR))
        for fact in SEED_FACTS:
            store.add(fact)

        # Dry run: tools that open or change things only say what they would do (each plugin's `dry`).
        registry = ToolRegistry(max_output_chars=config.TOOL_OUTPUT_MAX_CHARS, dry_run=True)
        load_plugins(registry, Deps(store=store))

        schemas = registry.schemas()
        teacher_base = build_system_prompt(teacher_persona(), schemas)
        production_base = build_system_prompt(build_persona(), schemas)
        llm = RecordingLLM(LlamaClient(server.base_url, temperature=0.8, max_tokens=500),
                           teacher_base, production_base)
        agent = Agent(llm, registry, teacher_base, max_steps=config.MAX_AGENT_STEPS,
                      history_budget_chars=int(config.HISTORY_BUDGET_TOKENS * config.CHARS_PER_TOKEN),
                      hooks=[EmotionHook(EmotionClassifier(config.EMOTION_DIR)), MemoryHook(store)],
                      confirm=lambda call, summary: True)

        conversations = read_conversations(HERE / "prompts.txt")
        if args.limit:
            conversations = conversations[: args.limit]
        kept, rejected = [], []
        t0 = time.time()
        for i, turns in enumerate(conversations, 1):
            for attempt in range(2):
                llm.records.clear()
                agent.reset()
                for turn in turns:
                    agent.run(turn)
                reasons = [reject_reason(r["messages"][-1]["content"]) for r in llm.records]
                good = [r for r, why in zip(llm.records, reasons) if why is None]
                bad = [{"why": why, **r} for r, why in zip(llm.records, reasons) if why]
                rejected += bad
                if not bad or attempt == 1:
                    kept += [{"source": "teacher", **r} for r in good]
                    break
            if i % 10 == 0 or i == len(conversations):
                print(f"[{i}/{len(conversations)}] kept {len(kept)} rejected {len(rejected)} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        for line in (HERE / "gold.jsonl").read_text(encoding="utf-8").splitlines():
            ex = json.loads(line)
            msgs = [{"role": "system", "content": production_base},
                    {"role": "user", "content": ex["user"]},
                    {"role": "assistant", "content": ex["assistant"]}]
            kept += [{"source": "gold", "messages": msgs}] * GOLD_REPEATS

        random.shuffle(kept)
        with open(out_dir / "train.jsonl", "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(out_dir / "rejected.jsonl", "w", encoding="utf-8") as f:
            for r in rejected:
                f.write(json.dumps({"why": r["why"], "output": r["messages"][-1]["content"],
                                    "user": next(m["content"] for m in reversed(r["messages"]) if m["role"] == "user")},
                                   ensure_ascii=False) + "\n")
        print(f"DONE: {len(kept)} training samples, {len(rejected)} rejected -> {out_dir}")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
