"""Phase 2 model bake-off: which brain should Byte run on this laptop (RTX 5050, 8 GB)?

For each model profile, with the same persona prompt and the same tool retrieval Byte uses:
  VRAM      peak GPU memory while it works (nvidia-smi, sampled every 0.25 s)
  speed     generation tokens/s and prompt tokens/s (llama-server's own timings)
  tools     70 one-shot requests: 60 need a specific tool, 10 are chat and need none.
            valid = parseable call to a real tool with valid arguments; right = the expected tool
  vision    3 staged screenshots (terminal traceback, dialog, editor) -> must name the error
  agent     5 multi-step tasks run through the real Agent loop in dry-run mode (nothing changes)

Run (Byte's own server must be stopped first so they don't fight over VRAM):
  .venv\\Scripts\\python.exe -m evals.bakeoff                         every downloaded profile
  .venv\\Scripts\\python.exe -m evals.bakeoff --only qwen3.5-4b --thinking
Writes evals/results/bakeoff-<stamp>.json and prints a table.
"""
import argparse
import datetime as dt
import io
import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from companion import config
from companion.agent import Agent
from companion.llm import LlamaClient
from companion.memory.embedder import Embedder
from companion.memory.store import MemoryStore
from companion.persona import build_persona
from companion.plugins import Deps, load_plugins
from companion.profiles import PROFILES
from companion.protocol import RESPONSE_OPEN, build_system_prompt, parse_tool_calls, strip_tool_calls
from companion.server import LlamaServer
from companion.toolpick import ToolPicker
from companion.tools import CHANGE, ToolError, ToolImage, ToolRegistry
from evals.toolpick_eval import load_queries

PORT = 8767
RESULTS = Path(__file__).resolve().parent / "results"
CHAT_ONLY = ["hey byte, how's it going?", "tell me a joke about gamers", "I'm bored lol",
             "what's your favourite game?", "thanks, that helped a lot", "good night!",
             "explain what a transformer model is in two lines", "are you an AI?",
             "I'm feeling stressed about exams", "say something encouraging"]


# --- VRAM sampling ----------------------------------------------------------------------------------
class VramPeak:
    def __init__(self):
        self.peak = 0
        self._stop = threading.Event()

    def _read(self) -> int:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        return int(out.stdout.strip().splitlines()[0])

    def __enter__(self):
        self.base = self._read()
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self._read())
            except (OSError, ValueError, IndexError):
                pass
            time.sleep(0.25)

    def __exit__(self, *a):
        self._stop.set()


# --- staged screenshots -----------------------------------------------------------------------------
def staged_images() -> list[tuple[str, bytes, list[str]]]:
    """(description, png, words the answer must contain one of)."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        mono = ImageFont.truetype("consola.ttf", 17)
        ui = ImageFont.truetype("segoeui.ttf", 18)
    except OSError:
        mono = ui = ImageFont.load_default()

    def png(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    term = Image.new("RGB", (1100, 420), (12, 12, 12))
    d = ImageDraw.Draw(term)
    lines = ["C:\\Users\\dev\\project> python server_config.py", "Traceback (most recent call last):",
             '  File "server_config.py", line 10, in <module>',
             "    load_settings('{\"host\": \"localhost\", \"port\": \"eighty\"}')",
             '  File "server_config.py", line 6, in load_settings',
             "    raise ValueError(f\"bad config value 'port': {settings['port']!r}\")",
             "ValueError: bad config value 'port': 'eighty'", "", "C:\\Users\\dev\\project>"]
    for i, l in enumerate(lines):
        d.text((14, 14 + i * 26), l, fill=(204, 204, 204), font=mono)

    dialog = Image.new("RGB", (700, 260), (243, 243, 243))
    d = ImageDraw.Draw(dialog)
    d.rectangle((0, 0, 700, 36), fill=(255, 255, 255))
    d.text((12, 8), "Destination Folder Access Denied", fill=(0, 0, 0), font=ui)
    d.text((80, 70), "You need permission to perform this action", fill=(0, 51, 153), font=ui)
    d.text((80, 110), "You require permission from TrustedInstaller to make changes to this folder.",
           fill=(0, 0, 0), font=ui)
    d.rectangle((470, 200, 570, 232), outline=(0, 120, 215), width=2)
    d.text((495, 204), "Try Again", fill=(0, 0, 0), font=ui)
    d.rectangle((585, 200, 685, 232), outline=(160, 160, 160))
    d.text((612, 204), "Cancel", fill=(0, 0, 0), font=ui)

    editor = Image.new("RGB", (1100, 380), (30, 30, 30))
    d = ImageDraw.Draw(editor)
    code = ["def total(items):", "    count = 0", "    for it in items:", "        count += it.price",
            "    return cuont"]
    for i, l in enumerate(code):
        d.text((50, 20 + i * 26), f"{i + 1:>2}  {l}", fill=(212, 212, 212), font=mono)
    d.line((175, 150, 225, 150), fill=(241, 76, 76), width=2)
    d.rectangle((0, 250, 1100, 380), fill=(24, 24, 24))
    d.text((14, 260), "PROBLEMS  1", fill=(255, 255, 255), font=ui)
    d.text((14, 295), "(X) \"cuont\" is not defined  Pylance(reportUndefinedVariable)  [Ln 5, Col 12]",
           fill=(241, 76, 76), font=mono)
    return [("terminal traceback", png(term), ["valueerror", "eighty"]),
            ("permission dialog", png(dialog), ["permission", "access denied", "trustedinstaller"]),
            ("editor problem", png(editor), ["cuont", "not defined", "undefined", "typo"])]


# --- agent tasks (dry-run, so nothing on the laptop changes) --------------------------------------
def agent_tasks(scratch: Path) -> list[tuple[str, callable]]:
    downloads = Path.home() / "Downloads"
    files = [p for p in downloads.iterdir() if p.is_file()] if downloads.exists() else []
    newest = max(files, key=lambda p: p.stat().st_mtime).name.lower() if files else "<none>"
    (scratch / "todo.txt").write_text("- buy milk\n- finish the CogniHire report\n", encoding="utf-8")
    (scratch / "notes.txt").write_text("Exam: Computer Networks unit 3 on Friday 3 October.\n", encoding="utf-8")

    def called(log, tool, **must):
        return any(c["name"] == tool and all(str(v).lower() in json.dumps(c["arguments"]).lower()
                                             for v in must.values()) for c in log)

    return [
        ("What's the newest file in my Downloads folder?",
         lambda ans, log: newest in ans.lower() or newest.rsplit(".", 1)[0] in ans.lower()),
        (f"Add 'study CN unit 3' to my todo list at {scratch}\\todo.txt.",
         lambda ans, log: any(c["name"] == "write_text_file" and "study cn unit 3" in json.dumps(c["arguments"]).lower()
                              and (c["arguments"].get("append") in (True, "true")
                                   or "buy milk" in json.dumps(c["arguments"]).lower()) for c in log)),
        (f"When is my exam? It's in {scratch}\\notes.txt", lambda ans, log: "friday" in ans.lower() or "3 oct" in ans.lower()),
        ("What's 17.5% of 2480, rounded to 2 decimals?", lambda ans, log: "434" in ans),
        ("Open Spotify and remind me in 25 minutes to stretch.",
         lambda ans, log: called(log, "launch_app", name="spotify") and called(log, "set_reminder", minutes="25")),
    ]


# --- one profile ------------------------------------------------------------------------------------
def evaluate(profile, embedder, thinking: bool, quick: bool) -> dict:
    import dataclasses
    if thinking:
        profile = dataclasses.replace(profile, thinking=True)
    server = LlamaServer(config.LLAMA_SERVER, profile.gguf, config.HOST, PORT, profile.ctx, config.GPU_LAYERS,
                         config.LOG_DIR, lora=profile.lora, profile=profile)
    server.log_path = config.LOG_DIR / f"bakeoff-{profile.name}.log"
    row = {"profile": profile.name, "thinking": thinking, "vision": profile.vision}
    with VramPeak() as vram:
        t0 = time.monotonic()
        server.start(timeout=400)
        row["load_s"] = round(time.monotonic() - t0, 1)
        try:
            llm = LlamaClient(server.base_url, temperature=0.2, max_tokens=1500 if thinking else 500)
            scratch = Path(tempfile.mkdtemp(prefix="bakeoff-"))
            store = MemoryStore(scratch / "m.db", embedder)
            for f in ("The user's name is Devesh.", "The user plays Valorant."):
                store.add(f)
            reg = ToolRegistry(dry_run=True)
            load_plugins(reg, Deps(store=store, vision=profile.vision))
            picker = ToolPicker(embedder, reg)
            persona = build_persona()

            # tools, one shot
            items = [(q["q"], q["tool"]) for q in load_queries()] + [(c, None) for c in CHAT_ONLY]
            if quick:
                items = items[::4]
            valid = right = no_tool_ok = 0
            gen_speed, prompt_speed = [], []
            for text, want in items:
                shown = set(picker.pick(text))
                system = build_system_prompt(persona, reg.schemas(CHANGE, only=shown))
                out = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": text}],
                               stop=[RESPONSE_OPEN])
                t = llm.last_timings
                if t.get("predicted_per_second"):
                    gen_speed.append(t["predicted_per_second"])
                if t.get("prompt_per_second") and t.get("prompt_n", 0) > 200:
                    prompt_speed.append(t["prompt_per_second"])
                calls, errors = parse_tool_calls(out)
                if want is None:
                    no_tool_ok += not calls and not errors
                    continue
                if calls and not errors:
                    try:
                        reg.validate(calls[0].name, calls[0].arguments)
                        valid += 1
                        right += calls[0].name == want
                    except ToolError:
                        pass
            n_tool = sum(w is not None for _, w in items)
            n_chat = len(items) - n_tool
            row.update(tool_valid=round(valid / n_tool, 3), tool_right=round(right / n_tool, 3),
                       chat_no_tool=round(no_tool_ok / max(n_chat, 1), 3),
                       gen_tok_s=round(sum(gen_speed) / max(len(gen_speed), 1), 1),
                       prompt_tok_s=round(sum(prompt_speed) / max(len(prompt_speed), 1), 1))

            # vision
            if profile.vision:
                ok, answers = 0, []
                for desc, png, words in staged_images():
                    msg = [{"role": "system", "content": persona},
                           {"role": "user", "content": [
                               {"type": "text", "text": "Here's my screen. What's the problem shown? One or two sentences."},
                               {"type": "image_url", "image_url": {"url": ToolImage(png, "").data_url()}}]}]
                    ans = llm.chat(msg)
                    hit = any(w in ans.lower() for w in words)
                    ok += hit
                    answers.append({"image": desc, "ok": hit, "answer": ans[:300]})
                row.update(vision_ok=f"{ok}/3", vision_answers=answers)

            # multi-step agent
            ok, details = 0, []
            for prompt, check in agent_tasks(scratch):
                log = []
                agent = Agent(llm, reg, "", tool_picker=picker, confirm=lambda c, s: True,
                              prompt_for_tools=lambda schemas: build_system_prompt(persona, schemas),
                              context_in_user_turn=True)
                t1 = time.monotonic()
                ans = agent.run(prompt, on_event=lambda e: log.append(e.data) if e.kind == "tool_call" else None)
                passed = bool(check(strip_tool_calls(ans), log))
                ok += passed
                details.append({"task": prompt[:60], "ok": passed, "tools": [c["name"] for c in log],
                                "seconds": round(time.monotonic() - t1, 1), "answer": ans[:200]})
            row.update(agent_ok=f"{ok}/5", agent_details=details)
        finally:
            server.stop()
    row["vram_peak_mb"] = vram.peak
    row["vram_used_mb"] = vram.peak - vram.base
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated profile names")
    ap.add_argument("--thinking", action="store_true", help="also run each profile with thinking on (D5)")
    ap.add_argument("--quick", action="store_true", help="a quarter of the tool requests (smoke test)")
    args = ap.parse_args()
    names = args.only.split(",") if args.only else [n for n, p in PROFILES.items() if p.available()]
    embedder = Embedder(config.EMBEDDER_DIR)
    rows = []
    for name in names:
        for thinking in ([False, True] if args.thinking else [False]):
            print(f"== {name}{' +thinking' if thinking else ''}", flush=True)
            try:
                row = evaluate(PROFILES[name], embedder, thinking, args.quick)
            except Exception as e:  # one broken model must not lose the others' results
                row = {"profile": name, "thinking": thinking, "error": f"{type(e).__name__}: {e}"}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if not k.endswith(("answers", "details"))}), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"bakeoff-{dt.datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'profile':16} {'think':5} {'VRAM MB':>8} {'gen t/s':>8} {'valid':>6} {'right':>6} {'chat':>5} "
          f"{'vision':>6} {'agent':>5}")
    for r in rows:
        if "error" in r:
            print(f"{r['profile']:16} ERROR {r['error'][:80]}")
            continue
        print(f"{r['profile']:16} {str(r['thinking'])[0]:5} {r['vram_used_mb']:>8} {r['gen_tok_s']:>8} "
              f"{r['tool_valid']:>6.0%} {r['tool_right']:>6.0%} {r['chat_no_tool']:>5.0%} "
              f"{r.get('vision_ok', '-'):>6} {r['agent_ok']:>5}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
