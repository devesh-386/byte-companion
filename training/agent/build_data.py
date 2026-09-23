"""Training data for Byte's 4B adapter: personality + tool use, in exactly the layout Byte runs with.

    teacher (the 4B + the long character sheet) ─┬─ prompts.txt conversations      -> Byte's voice
                                                 └─ tool_prompts.tsv (expected tool) -> Byte's own tools,
                                                    kept ONLY if the right tool was called cleanly
    Hermes function-calling (multi-turn)  ─┐
    Glaive function-calling v2 (sample)   ─┴─ converted to <tool_call> format; only the tool-call turns
                                             are graded (their prose is generic-assistant, not Byte)

Every sample is (messages, graded last assistant turn). The system prompt is always Byte's short production
persona + the tools of that example, per-turn context rides in the user message (D3), and tool results come
back as <tool_response> user turns: the same bytes llama-server will see at run time.

Run:  .venv/Scripts/python.exe training/agent/build_data.py [--teacher-limit N]
Out:  data/agent_4b/{train,val}.jsonl
"""
import argparse
import json
import random
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training" / "persona"))

from companion import config  # noqa: E402
from companion.persona import build_persona  # noqa: E402
from companion.protocol import build_system_prompt, format_tool_response, parse_tool_calls  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = ROOT / "data" / "agent_4b"
HERMES = Path("D:/byte-data/hermes/func-calling.json")
GLAIVE = Path("D:/byte-data/glaive/glaive-function-calling-v2.json")
GENERIC = re.compile(r"\b(certainly|absolutely|i hope this helps|feel free|how can i assist|as an ai)\b", re.I)
FAKE_WINDOW = ("Window: Byte Test\n[1] Text 'This is a practice window.'\n[2] Edit 'Message'\n"
               "[3] Button 'OK'\n[4] Button 'Cancel'")


def call_block(name: str, arguments: dict) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False) + "\n</tool_call>"


def schema(fn: dict) -> dict:
    fn = fn.get("function", fn)
    return {"type": "function", "function": {"name": fn["name"], "description": fn.get("description", ""),
                                             "parameters": fn.get("parameters", {"type": "object", "properties": {}})}}


# --- Hermes ---------------------------------------------------------------------------------------
def hermes_samples(persona: str, limit: int, rng: random.Random) -> list[dict]:
    data = json.loads(HERMES.read_text(encoding="utf-8"))
    rng.shuffle(data)
    out = []
    for ex in data:
        tools = ex["tools"] if isinstance(ex["tools"], list) else json.loads(ex["tools"])
        system = build_system_prompt(persona, [schema(t) for t in tools])
        msgs = [{"role": "system", "content": system}]
        for turn in ex["conversations"]:
            role, text = turn["from"], turn["value"].strip()
            if role == "system":
                continue
            if role == "human":
                msgs.append({"role": "user", "content": text})
            elif role == "tool":
                msgs.append({"role": "user", "content": text})
            elif role == "gpt":
                calls, errors = parse_tool_calls(text)
                if errors:
                    break
                if calls:
                    content = "\n".join(call_block(c.name, c.arguments) for c in calls)
                    out.append({"source": "hermes", "messages": msgs + [{"role": "assistant", "content": content}]})
                msgs.append({"role": "assistant", "content": text})
        if len(out) >= limit:
            break
    return out[:limit]


# --- Glaive ---------------------------------------------------------------------------------------
_FN_CALL = re.compile(r"<functioncall>\s*(\{.*?\})\s*(?:<\|endoftext\|>|$)", re.S)


def _glaive_functions(system: str) -> list[dict]:
    body = system.split("-", 1)[1] if "Use them if required" in system else ""
    fns, depth, start = [], 0, None
    for i, ch in enumerate(body):  # the functions are JSON objects written one after another
        if ch == "{":
            depth += 1
            start = i if depth == 1 else start
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    fns.append(json.loads(body[start:i + 1]))
                except json.JSONDecodeError:
                    return []
    return fns


def _glaive_call(text: str) -> tuple[str, dict] | None:
    m = _FN_CALL.search(text)
    if not m:
        return None
    raw = m.group(1)
    # arguments arrive as a single-quoted JSON *string*: "arguments": '{"a": 1}'
    raw = re.sub(r"'(\{.*\})'", lambda g: json.dumps(g.group(1)), raw, flags=re.S)
    try:
        obj = json.loads(raw)
        args = obj.get("arguments", {})
        args = json.loads(args) if isinstance(args, str) else args
        return obj["name"], args
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def glaive_samples(persona: str, limit: int, rng: random.Random) -> list[dict]:
    data = json.loads(GLAIVE.read_text(encoding="utf-8"))
    rng.shuffle(data)
    out = []
    for ex in data:
        fns = _glaive_functions(ex["system"])
        if not fns:
            continue
        system = build_system_prompt(persona, [schema(f) for f in fns])
        msgs = [{"role": "system", "content": system}]
        parts = re.split(r"\n*(USER|ASSISTANT|FUNCTION RESPONSE):\s*", ex["chat"])
        ok = True
        for role, text in zip(parts[1::2], parts[2::2]):
            text = text.replace("<|endoftext|>", "").strip()
            if role == "USER":
                msgs.append({"role": "user", "content": text})
            elif role == "FUNCTION RESPONSE":
                name = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), "")
                fn = re.search(r'"name":\s*"([^"]+)"', name)
                msgs.append({"role": "user", "content": format_tool_response(fn.group(1) if fn else "tool", text)})
            else:
                call = _glaive_call(text)
                if call is None and "<functioncall>" in text:
                    ok = False
                    break
                if call:
                    content = call_block(*call)
                    out.append({"source": "glaive", "messages": msgs + [{"role": "assistant", "content": content}]})
                    msgs.append({"role": "assistant", "content": content})
                else:
                    msgs.append({"role": "assistant", "content": text})
        if not ok:
            continue
        if len(out) >= limit:
            break
    return out[:limit]


# --- the teacher (Byte's own tools and voice) ----------------------------------------------------
def teacher_samples(limit: int) -> list[dict]:
    from generate_data import SEED_FACTS, reject_reason, read_conversations, teacher_persona

    from companion.agent import Agent
    from companion.app import make_server
    from companion.hooks import EmotionHook
    from companion.llm import LlamaClient
    from companion.memory.embedder import Embedder
    from companion.memory.hook import MemoryHook
    from companion.memory.store import MemoryStore
    from companion.ml.emotion import EmotionClassifier
    from companion.plugins import Deps, load_plugins
    from companion.toolpick import ToolPicker
    from companion.tools import ToolRegistry

    server = make_server()
    server.start()
    embedder = Embedder(config.EMBEDDER_DIR)
    store = MemoryStore(Path(tempfile.mkdtemp(prefix="agent-gen-")) / "m.db", embedder)
    for fact in SEED_FACTS:
        store.add(fact)
    reg = ToolRegistry(max_output_chars=config.TOOL_OUTPUT_MAX_CHARS, dry_run=True)
    load_plugins(reg, Deps(store=store))  # no vision: screenshots of the real screen never enter training data
    # Window tools read real app contents; a fake practice window keeps private chats out of the data.
    reg.override("observe_window", lambda title="": FAKE_WINDOW)
    reg.override("list_open_windows", lambda: "- Byte Test\n- Visual Studio Code\n- Spotify")
    picker = ToolPicker(embedder, reg)
    teacher, production = teacher_persona(), build_persona()
    records: list[dict] = []

    class Recorder:
        def __init__(self, llm):
            self.llm = llm

        def chat(self, messages, **kw):
            out = self.llm.chat(messages, **kw)
            system = messages[0]["content"].replace(teacher, production)  # teach the short prompt the long one's manner
            records.append({"messages": [{"role": "system", "content": system}] + [dict(m) for m in messages[1:]]
                            + [{"role": "assistant", "content": out}]})
            return out

    llm = Recorder(LlamaClient(server.base_url, temperature=0.7, max_tokens=600))
    agent = Agent(llm, reg, "", max_steps=8, tool_picker=picker,
                  prompt_for_tools=lambda schemas: build_system_prompt(teacher, schemas),
                  hooks=[EmotionHook(EmotionClassifier(config.EMOTION_DIR)), MemoryHook(store)],
                  confirm=lambda c, s: True, context_in_user_turn=True)

    kept, stats = [], {"voice_kept": 0, "tool_kept": 0, "tool_dropped": 0, "rejected_turns": 0}
    convs = read_conversations(ROOT / "training" / "persona" / "prompts.txt")
    tools = [l.split("\t", 1) for l in (HERE / "tool_prompts.tsv").read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    if limit:
        convs, tools = convs[:limit], tools[:limit]
    t0 = time.time()
    for i, turns in enumerate(convs, 1):
        records.clear()
        agent.reset()
        for turn in turns:
            agent.run(turn)
        good = [r for r in records if reject_reason(r["messages"][-1]["content"]) is None]
        stats["rejected_turns"] += len(records) - len(good)
        kept += [{"source": "voice", **r} for r in good]
        stats["voice_kept"] += len(good)
        if i % 20 == 0:
            print(f"  voice {i}/{len(convs)} kept {stats['voice_kept']} ({time.time() - t0:.0f}s)", flush=True)
    for want, text in tools:
        for attempt in range(3):  # rejection sampling: up to 3 tries to get it right
            records.clear()
            agent.reset()
            called = []
            agent.run(text, on_event=lambda e: called.append((e.kind, e.data.get("name")))
                      if e.kind in ("tool_call", "tool_error") else None)
            names = [n for k, n in called if k == "tool_call"]
            errors = [n for k, n in called if k == "tool_error"]
            clean = all(reject_reason(r["messages"][-1]["content"]) is None for r in records)
            if want in names and not errors and clean:
                kept += [{"source": "tools", **r} for r in records]
                stats["tool_kept"] += 1
                break
        else:
            stats["tool_dropped"] += 1
            print(f"  dropped (never right in 3 tries): {want}: {text}", flush=True)
    print(f"teacher: {stats} in {time.time() - t0:.0f}s")
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-limit", type=int, default=0)
    ap.add_argument("--hermes", type=int, default=1200)
    ap.add_argument("--glaive", type=int, default=1500)
    ap.add_argument("--teacher-repeat", type=int, default=3, help="Byte's own data counts this many times")
    args = ap.parse_args()
    rng = random.Random(7)
    persona = build_persona()
    OUT.mkdir(parents=True, exist_ok=True)

    ext = hermes_samples(persona, args.hermes, rng) + glaive_samples(persona, args.glaive, rng)
    ext = [s for s in ext if not GENERIC.search(s["messages"][-1]["content"])]
    print(f"external: {sum(s['source'] == 'hermes' for s in ext)} hermes + {sum(s['source'] == 'glaive' for s in ext)} glaive")
    own = teacher_samples(args.teacher_limit)

    rng.shuffle(own)
    n_val = max(10, len(own) // 12)
    val = own[:n_val] + ext[:40]
    train = own[n_val:] * args.teacher_repeat + ext[40:]
    rng.shuffle(train)
    for name, rows in (("train", train), ("val", val)):
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"DONE train {len(train)}  val {len(val)}  (Byte's own {len(own)} x{args.teacher_repeat}) -> {OUT}")


if __name__ == "__main__":
    main()
