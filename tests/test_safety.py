"""Phase 0 safety rails: tiers, the action log, cancelling (incl. a real HTTP stream), and the hotkey."""
import http.server
import json
import threading
import time
from pathlib import Path

import pytest

from companion.actionlog import MemoryActionLog
from companion.agent import STOPPED_ANSWER, Agent
from companion.cancel import STOP, Cancelled, CancelToken
from companion.llm import LlamaClient
from companion.tools import CHANGE, LOOK, NEVER, OPEN, ToolRegistry

ROOT = Path(__file__).resolve().parents[1]


class ScriptLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, *, stop=None, on_token=None, cancel=None, **kw):
        self.seen.append(messages)
        if cancel:
            cancel.check()
        return self.replies.pop(0)


def call(name, **args):
    return f'<tool_call>{{"name": "{name}", "arguments": {json.dumps(args)}}}</tool_call>'


def make(replies, confirm=None, max_tier=CHANGE):
    reg = ToolRegistry()
    ran = []
    for name, tier in (("peek", LOOK), ("launch", OPEN), ("write", CHANGE), ("pay", NEVER)):
        def fn(_n=name) -> str:
            ran.append(_n)
            return f"{_n} done"
        fn.__name__, fn.__doc__ = name, f"{name} tool"
        reg.register(fn, tier=tier)
    log = MemoryActionLog("test")
    agent = Agent(ScriptLLM(replies), reg, "S", confirm=confirm, max_tier=max_tier, action_log=log)
    return agent, reg, ran, log


# --- tiers ----------------------------------------------------------------------------------------
def test_look_and_open_run_without_asking():
    asked = []
    agent, _, ran, log = make([call("peek"), call("launch"), "ok"], confirm=lambda c, s: asked.append(s))
    agent.run("go")
    assert ran == ["peek", "launch"] and asked == []
    assert [(e["tool"], e["decision"], e["outcome"]) for e in log.entries] == \
        [("peek", "auto", "ok"), ("launch", "auto", "ok")]


def test_change_asks_and_respects_the_answer():
    agent, _, ran, log = make([call("write"), "ok"], confirm=lambda c, s: False)
    agent.run("go")
    assert ran == [] and log.entries[0]["decision"] == "denied" and log.entries[0]["outcome"] == "skipped"
    agent, _, ran, log = make([call("write"), "ok"], confirm=lambda c, s: True)
    agent.run("go")
    assert ran == ["write"] and log.entries[0]["decision"] == "allowed"


def test_broken_change_call_never_reaches_the_user():
    asked = []
    agent, _, ran, log = make([call("write", junk=1), "ok"], confirm=lambda c, s: asked.append(s) or True)
    agent.run("go")
    assert asked == [] and ran == [] and log.entries[0]["outcome"] == "error"


def test_change_without_a_confirm_function_is_denied():
    agent, _, ran, _ = make([call("write"), "ok"], confirm=None)
    agent.run("go")
    assert ran == []


def test_never_is_refused_even_if_user_would_allow():
    agent, _, ran, log = make([call("pay"), "ok"], confirm=lambda c, s: True)
    agent.run("go")
    assert ran == [] and log.entries[0]["decision"] == "blocked"


def test_max_tier_blocks_and_schemas_hide_higher_tiers():
    agent, reg, ran, log = make([call("launch"), call("peek"), "ok"], max_tier=LOOK)
    agent.run("go")
    assert ran == ["peek"] and log.entries[0]["decision"] == "blocked"
    assert [s["function"]["name"] for s in reg.schemas(max_tier=LOOK)] == ["peek"]
    assert "pay" in [s["function"]["name"] for s in reg.schemas()]


def test_unknown_tier_rejected():
    with pytest.raises(ValueError):
        ToolRegistry().register(lambda: "x", tier=7)


def test_file_action_log(tmp_path):
    from companion.actionlog import ActionLog
    log = ActionLog(tmp_path / "a.jsonl", "chat")
    log.write(tool="peek", tier=0, arguments={"p": 1}, decision="auto", outcome="ok", detail="x" * 900)
    entry = json.loads((tmp_path / "a.jsonl").read_text(encoding="utf-8"))
    assert entry["context"] == "chat" and entry["tool"] == "peek" and len(entry["detail"]) == 500


# --- cancelling -----------------------------------------------------------------------------------
def test_cancel_token_callbacks_run_once_and_late_registration_fires():
    t = CancelToken()
    hits = []
    unregister = t.on_cancel(lambda: hits.append("a"))
    t.on_cancel(lambda: hits.append("b"))
    unregister()
    t.cancel("x")
    t.cancel("again")
    assert hits == ["b"] and t.reason == "x"
    t.on_cancel(lambda: hits.append("late"))
    assert hits == ["b", "late"]
    with pytest.raises(Cancelled):
        t.check()


def test_stop_all_reaches_a_run_that_passed_no_token():
    started = threading.Event()

    class SlowLLM:
        def chat(self, messages, *, cancel=None, **kw):
            started.set()
            cancel.wait(5)
            cancel.check()
            return "never"

    agent = Agent(SlowLLM(), ToolRegistry(), "S")
    result = {}
    th = threading.Thread(target=lambda: result.setdefault("answer", agent.run("hi")))
    th.start()
    started.wait(2)
    assert STOP.stop_all() >= 1
    th.join(2)
    assert result["answer"] == STOPPED_ANSWER
    assert agent.turns[-1][-1] == {"role": "assistant", "content": STOPPED_ANSWER}


class _SlowSSE(http.server.BaseHTTPRequestHandler):
    """Imitates llama-server: silent while 'reading the prompt', then a slow token stream."""
    header_delay = 5.0

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        time.sleep(self.header_delay)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(50):
                chunk = {"choices": [{"delta": {"content": f"t{i} "}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.2)
        except OSError:
            pass

    def log_message(self, *a):
        pass


@pytest.fixture
def slow_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SlowSSE)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.mark.parametrize("cancel_after,header_delay", [(0.5, 5.0), (0.8, 0.0)])
def test_real_http_stream_cancels_in_under_a_second(slow_server, cancel_after, header_delay):
    """Done-when for phase 0: a stop lands in < 1 s, both while the prompt is being read (no bytes yet)
    and while tokens are streaming."""
    _SlowSSE.header_delay = header_delay
    token = CancelToken()
    client = LlamaClient(slow_server, timeout=30)
    threading.Timer(cancel_after, token.cancel, args=("test",)).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        client.chat([{"role": "user", "content": "hi"}], cancel=token)
    assert time.monotonic() - t0 - cancel_after < 1.0


def test_uncancelled_stream_still_returns_text(slow_server):
    _SlowSSE.header_delay = 0.0
    client = LlamaClient(slow_server, timeout=30)
    token = CancelToken()
    got = []
    threading.Timer(0.5, token.cancel).start()
    with pytest.raises(Cancelled):
        client.chat([{"role": "user", "content": "hi"}], cancel=token, on_token=got.append)
    assert got and got[0] == "t0 "


# --- the real Windows hotkey ----------------------------------------------------------------------
def test_global_hotkey_fires_and_conflicts_are_reported():
    import ctypes
    from companion.hotkey import MOD_ALT, MOD_CONTROL, MOD_SHIFT, VK_F12, GlobalHotkey, HotkeyError
    fired = threading.Event()
    mods = MOD_CONTROL | MOD_ALT | MOD_SHIFT  # an unused combo, so this test never fights Byte's Ctrl+Alt+Esc
    hk = GlobalHotkey(fired.set, modifiers=mods, vk=VK_F12, hotkey_id=0xBEEF)
    hk.start()
    try:
        with pytest.raises(HotkeyError):
            GlobalHotkey(lambda: None, modifiers=mods, vk=VK_F12, hotkey_id=0xBEF0).start()
        user32 = ctypes.windll.user32
        keys = (0x11, 0x12, 0x10, VK_F12)  # Ctrl, Alt, Shift, F12
        for k in keys:
            user32.keybd_event(k, 0, 0, 0)
        for k in reversed(keys):
            user32.keybd_event(k, 0, 2, 0)
        assert fired.wait(2), "hotkey callback did not fire"
    finally:
        hk.stop()


# --- the task list stays in sync with the code ----------------------------------------------------
def test_tasks_doc_matches_code():
    import re
    from evals.tasks import TASKS
    doc = (ROOT / "docs" / "tasks.md").read_text(encoding="utf-8")
    in_doc = re.findall(r"^## (T\d\d) ([\w-]+)$", doc, re.M)
    assert in_doc == [(t.id, t.slug) for t in TASKS]
    for t in TASKS:
        section = doc.split(f"## {t.id} ")[1].split("\n## ")[0]
        assert t.ask.split("{")[0].strip().rstrip(".") in section, f"{t.id} ask differs from docs"
