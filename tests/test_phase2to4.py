"""Phases 2-4 plumbing: model profiles, thinking / XML tool calls, images in history, UIA helpers."""
from pathlib import Path

import pytest

from companion.agent import IMAGE_REMOVED, Agent, content_chars, drop_old_images
from companion.profiles import ModelProfile
from companion.protocol import VisibleTextFilter, parse_tool_calls, strip_tool_calls
from companion.tools import LOOK, ToolImage, ToolRegistry
from companion import uia


# --- profiles -------------------------------------------------------------------------------------
def test_profile_server_args():
    p = ModelProfile("x", Path("m.gguf"), mmproj=Path("v.gguf"), ctx=8192)
    args = p.server_args()
    assert args[:2] == ["-c", "8192"]
    assert ["-ctk", "q8_0"] == args[args.index("-ctk"):args.index("-ctk") + 2] and "-fa" in args
    assert "--mmproj" in args and "--image-max-tokens" in args
    assert args[args.index("--reasoning") + 1] == "off"
    text_only = ModelProfile("t", Path("m.gguf"), kv_type="f16", thinking=True).server_args()
    assert "--no-mmproj" in text_only and "-ctk" not in text_only and "on" in text_only


def test_server_uses_profile_args(tmp_path):
    from companion.server import LlamaServer
    p = ModelProfile("x", tmp_path / "m.gguf")
    s = LlamaServer(tmp_path / "srv.exe", p.gguf, "127.0.0.1", 9, p.ctx, 99, tmp_path, lora=tmp_path / "l.gguf", profile=p)
    args = s.args()
    assert args[args.index("-m") + 1].endswith("m.gguf") and "--lora" in args and "-ctk" in args


# --- thinking + qwen3.5 style tool calls ------------------------------------------------------------
def test_think_blocks_are_hidden_while_streaming_and_after():
    f = VisibleTextFilter()
    shown = "".join(f.feed(c) for c in ["Hi <thi", "nk>secret plan", " more</th", "ink>\nthere!"]) + f.flush()
    assert shown == "Hi there!"
    assert strip_tool_calls("<think>x</think>Answer") == "Answer"


def test_unclosed_think_never_leaks():
    f = VisibleTextFilter()
    assert f.feed("<think>still thinking...") == "" and f.flush() == ""


def test_tool_call_after_thinking_still_hidden_and_parsed():
    raw = '<think>use calc</think>Sure. <tool_call>{"name": "calculate", "arguments": {"expression": "2+2"}}</tool_call>'
    f = VisibleTextFilter()
    assert "".join(f.feed(c) for c in raw) + f.flush() == "Sure. "
    calls, errors = parse_tool_calls(raw)
    assert calls[0].name == "calculate" and not errors


def test_xml_style_tool_calls():
    raw = ("<tool_call>\n<function=set_volume>\n<parameter=percent>\n30\n</parameter>\n</function>\n</tool_call>"
           "<tool_call><function=web_search><parameter=query>rtx 5050 review</parameter></function></tool_call>")
    calls, errors = parse_tool_calls(raw)
    assert not errors
    assert (calls[0].name, calls[0].arguments) == ("set_volume", {"percent": 30})
    assert (calls[1].name, calls[1].arguments) == ("web_search", {"query": "rtx 5050 review"})


# --- images ---------------------------------------------------------------------------------------
def img_msg(tag):
    return {"role": "user", "content": [{"type": "text", "text": tag},
                                        {"type": "image_url", "image_url": {"url": "data:x"}}]}


def test_only_the_newest_image_survives():
    turns = [[img_msg("a")], [img_msg("b")], [img_msg("c")]]
    drop_old_images(turns)
    kinds = [[p["type"] for p in t[0]["content"]] for t in turns]
    assert kinds == [["text", "text"], ["text", "text"], ["text", "image_url"]]
    assert turns[0][0]["content"][1]["text"] == IMAGE_REMOVED
    assert content_chars(img_msg("")["content"]) == 7000


def test_tool_image_reaches_the_model_as_an_image():
    reg = ToolRegistry()

    def look_at_screen() -> str:
        """look"""
        return ToolImage(b"\x89PNG", "Screenshot of the window 'x'.")
    reg.register(look_at_screen, tier=LOOK)
    seen = []

    class LLM:
        replies = ['<tool_call>{"name": "look_at_screen", "arguments": {}}</tool_call>', "It's a ValueError."]

        def chat(self, messages, **kw):
            seen.append(messages)
            return self.replies.pop(0)
    ans = Agent(LLM(), reg, "S").run("what's on my screen")
    assert ans == "It's a ValueError."
    last_user = seen[1][-1]
    assert last_user["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_context_in_user_turn_keeps_system_prompt_stable():
    class Hook:
        def before_turn(self, text, emit):
            return f"mood: {text}"

        def after_turn(self, *a):
            pass
    systems, users = [], []

    class LLM:
        def chat(self, messages, **kw):
            systems.append(messages[0]["content"])
            users.append(messages[-1]["content"])
            return "ok"
    agent = Agent(LLM(), ToolRegistry(), "SYS", hooks=[Hook()], context_in_user_turn=True)
    agent.run("one")
    agent.run("two")
    assert systems == ["SYS", "SYS"]
    assert users[1].startswith("[Context]\nmood: two") and users[1].endswith("two")
    assert agent.turns[1][0]["content"] == "two"  # stored history stays clean


# --- UI automation helpers ------------------------------------------------------------------------
@pytest.mark.parametrize("combo,keys", [("ctrl+s", "{Ctrl}s"), ("alt+f4", "{Alt}{F4}"), ("enter", "{Enter}"),
                                        ("ctrl+shift+esc", "{Ctrl}{Shift}{Esc}")])
def test_combo_to_sendkeys(combo, keys):
    assert uia.combo_to_sendkeys(combo) == keys


def test_bad_key_and_escaping():
    with pytest.raises(uia.UIAError):
        uia.combo_to_sendkeys("ctrl+banana")
    assert uia.escape_keys("a{b}") == "a{{}b{}}"


def test_diff_reports_changes():
    assert uia.diff({"Button 'OK'"}, {"Button 'OK'"}) == "Nothing visible changed in that window."
    d = uia.diff({"Button 'OK'", "Text 'Hi'"}, {"Text 'Hi'", "Edit 'Name'"})
    assert "appeared: Edit 'Name'" in d and "gone: Button 'OK'" in d


def test_claimed_action_without_a_tool_gets_one_nudge():
    from companion.agent import UNBACKED_CLAIM, claims_action
    assert claims_action("Got it! I've clicked OK on the dialog.")
    assert claims_action("Done, I opened Spotify for you")
    assert not claims_action("I think you should click OK.")
    assert claims_action("Quest accepted! I'll open Notepad, type hello, and save it. Ready?")  # suite run 3, T10
    assert not claims_action("I'll be here if you need me.")
    assert claims_action("Got it, Devesh! Clicking OK on that dialog for you.")  # suite run 4, T09
    assert not claims_action("Try clicking the OK button yourself.")
    reg = ToolRegistry()

    def click() -> str:
        """click"""
        return "clicked"
    reg.register(click)

    class LLM:
        replies = ["I've clicked OK on the dialog.", '<tool_call>{"name": "click", "arguments": {}}</tool_call>',
                   "Clicked it for real."]
        seen = []

        def chat(self, messages, **kw):
            self.seen.append(messages[-1]["content"])
            return self.replies.pop(0)
    llm = LLM()
    assert Agent(llm, reg, "S").run("click OK") == "Clicked it for real."
    assert llm.seen[1] == UNBACKED_CLAIM


def test_honesty_nudge_happens_only_once():
    reg = ToolRegistry()

    def t() -> str:
        """t"""
        return "x"
    reg.register(t)

    class LLM:
        def chat(self, messages, **kw):
            return "I've opened it."
    assert Agent(LLM(), reg, "S").run("open it") == "I've opened it."


def test_repeating_the_same_look_is_short_circuited():
    from companion.agent import REPEATED_LOOK
    reg = ToolRegistry()
    runs = []

    def observe_window() -> str:
        """observe"""
        runs.append(1)
        return "[1] Button 'OK'"
    reg.register(observe_window, tier=LOOK)
    call = '<tool_call>{"name": "observe_window", "arguments": {}}</tool_call>'

    class LLM:
        replies = [call, call, "ok"]
        seen = []

        def chat(self, messages, **kw):
            self.seen.append(messages[-1]["content"])
            return self.replies.pop(0)
    llm = LLM()
    Agent(llm, reg, "S").run("look")
    assert len(runs) == 1 and REPEATED_LOOK in llm.seen[2]


def test_black_screenshots_are_explained_not_sent(monkeypatch):
    from PIL import Image
    from companion.plugins import screen
    black, normal = Image.new("RGB", (800, 600)), Image.new("RGB", (800, 600), (40, 90, 160))
    monkeypatch.setattr(screen, "capture", lambda whole_screen=False: (normal if whole_screen else black, "the window 'Netflix'"))
    out = screen.look_at_screen()
    assert isinstance(out, ToolImage) and "protected content" in out.text   # fell back to the whole screen
    monkeypatch.setattr(screen, "capture", lambda whole_screen=False: (black, "the window 'Netflix'"))
    out = screen.look_at_screen()
    assert isinstance(out, str) and "black" in out and "Netflix" in out
