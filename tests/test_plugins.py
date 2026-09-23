"""Phase 1: plugin discovery, dry-run mode, tool retrieval (fake + the real bge model)."""
import json
import textwrap

import numpy as np
import pytest

from companion import config
from companion.agent import Agent
from companion.plugins import Deps, load_plugins
from companion.tools import CHANGE, LOOK, OPEN, ToolRegistry
from companion.toolpick import ToolPicker


def write_plugin(folder, name, body):
    (folder / f"{name}.py").write_text(textwrap.dedent(body), encoding="utf-8")


# --- discovery ------------------------------------------------------------------------------------
def test_fake_plugin_folder_loads_in_order_and_skips_bad_ones(tmp_path):
    write_plugin(tmp_path, "b_late", """
        ORDER = 50
        def register(reg, deps):
            def late() -> str:
                "late tool"
                return "late"
            reg.register(late)
    """)
    write_plugin(tmp_path, "a_early", """
        ORDER = 10
        def register(reg, deps):
            def early() -> str:
                "early tool"
                return "early"
            reg.register(early)
    """)
    write_plugin(tmp_path, "broken_import", "import does_not_exist_anywhere\n")
    write_plugin(tmp_path, "no_register", "X = 1\n")
    write_plugin(tmp_path, "half_done", """
        def register(reg, deps):
            def good() -> str:
                "fine"
                return "x"
            reg.register(good)
            raise RuntimeError("boom after one tool")
    """)
    write_plugin(tmp_path, "dupe", """
        ORDER = 60
        def register(reg, deps):
            def early() -> str:
                "same name as a_early's tool"
                return "x"
            reg.register(early)
    """)
    write_plugin(tmp_path, "_private", "raise SystemExit('never imported')\n")

    reg = ToolRegistry()
    report = load_plugins(reg, Deps(), folder=tmp_path)
    assert reg.names() == ["early", "late"]            # ORDER, not file name, decides
    assert set(report.failed) == {"broken_import", "no_register", "half_done", "dupe"}
    assert "good" not in reg.names()                     # a failed plugin leaves nothing behind
    assert "already registered" in report.failed["dupe"]


def test_builtin_plugins_load_all_tools_with_tiers():
    reg = ToolRegistry()
    report = load_plugins(reg, Deps(store=object()))
    assert not report.failed
    assert set(reg.names()) == {
        "get_current_time", "calculate", "system_info", "set_reminder", "list_directory", "read_text_file",
        "open_folder", "open_file", "write_text_file", "web_search", "fetch_webpage", "launch_app",
        "remember", "recall", "forget", "memory_overview",
        "list_open_windows", "find_folder", "observe_window", "media_control", "set_volume", "focus_window",
        "open_in_vscode", "play_spotify", "click", "type_text", "press_keys", "close_window",
        "search_own_code", "read_own_code", "edit_own_code", "write_plugin", "undo_self_edit", "restart_byte"}
    assert reg.tier("open_file") == CHANGE and reg.tier("launch_app") == OPEN and reg.tier("recall") == LOOK
    assert {n for n in reg.names() if reg.tier(n) == CHANGE} == {
        "open_file", "write_text_file", "forget", "click", "type_text", "press_keys", "close_window",
        "edit_own_code", "write_plugin", "undo_self_edit", "restart_byte"}
    assert "look_at_screen" not in reg.names()  # only for brains that can see


def test_memory_plugin_is_skipped_without_a_store():
    reg = ToolRegistry()
    load_plugins(reg, Deps())
    assert "remember" not in reg.names() and "calculate" in reg.names()


def test_include_limits_families():
    reg = ToolRegistry()
    load_plugins(reg, Deps(), include={"files"})
    assert reg.names() == ["list_directory", "read_text_file", "open_folder", "open_file", "write_text_file"]


# --- dry run --------------------------------------------------------------------------------------
def test_dry_run_changes_nothing_but_reads_for_real(tmp_path):
    reg = ToolRegistry(dry_run=True)
    load_plugins(reg, Deps())
    target = tmp_path / "x.txt"
    out = reg.call("write_text_file", {"path": str(target), "content": "hi"})
    assert out.startswith("Wrote") and not target.exists()
    target.write_text("real text", encoding="utf-8")
    assert reg.call("read_text_file", {"path": str(target)}) == "real text"
    with pytest.raises(Exception, match="Not a folder"):  # dry versions still validate like the real ones
        reg.call("open_folder", {"path": str(target)})


def test_dry_run_without_a_dry_function_describes_the_call():
    reg = ToolRegistry(dry_run=True)
    ran = []

    def poke(x: int) -> str:
        """poke"""
        ran.append(x)
        return "poked"
    reg.register(poke, tier=OPEN)
    assert reg.call("poke", {"x": 3}) == "(dry run) would call poke(x=3)" and ran == []


# --- retrieval ------------------------------------------------------------------------------------
class BagEmbedder:
    """Tiny fake: vectors are word-count bags over a fixed vocabulary."""
    VOCAB = ["time", "clock", "music", "song", "file", "read", "weather", "web"]

    def embed(self, texts, query=False):
        out = np.zeros((len(texts), len(self.VOCAB)), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, w in enumerate(self.VOCAB):
                out[i, j] = t.lower().count(w)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def small_registry():
    reg = ToolRegistry()
    for name, doc, ex in (("clock", "tells the time", ("what time",)), ("player", "plays music", ("a song",)),
                          ("reader", "read a file", ()), ("searcher", "search the web", ("weather",))):
        def fn() -> str:
            return name
        fn.__name__, fn.__doc__ = name, doc
        reg.register(fn, examples=ex)
    return reg


def test_picker_ranks_by_best_match_including_examples():
    picker = ToolPicker(BagEmbedder(), small_registry(), k=1)
    assert picker.pick("what's the weather") == ["searcher"]   # only its example mentions weather
    assert picker.pick("play a song") == ["player"]
    assert len(picker.pick("anything", k=10)) == 4


def test_picker_refreshes_when_tools_change():
    reg = small_registry()
    picker = ToolPicker(BagEmbedder(), reg, k=1)

    assert picker.pick("song") == ["player"]
    reg.unregister("player")    # same number of tools afterwards, so a count check would miss the swap

    def radio() -> str:
        """streams a song"""
        return "x"
    reg.register(radio)
    assert picker.pick("song") == ["radio"]


class ScriptLLM:
    def __init__(self, replies):
        self.replies, self.systems = list(replies), []

    def chat(self, messages, **kw):
        self.systems.append(messages[0]["content"])
        return self.replies.pop(0)


def test_agent_shows_only_picked_tools_and_keeps_last_used():
    reg = small_registry()
    call = '<tool_call>{"name": "clock", "arguments": {}}</tool_call>'
    llm = ScriptLLM([call, "it's noon", "again: noon"])
    agent = Agent(llm, reg, "S", tool_picker=ToolPicker(BagEmbedder(), reg, k=1),
                  prompt_for_tools=lambda schemas: "TOOLS " + json.dumps([s["function"]["name"] for s in schemas]))
    agent.run("what time is it")
    assert llm.systems[0] == 'TOOLS ["clock"]'
    agent.run("play a song")                      # clock was used last turn, so it stays visible
    assert llm.systems[-1] == 'TOOLS ["clock", "player"]'


def test_agent_picker_respects_max_tier():
    reg = ToolRegistry()

    def peek() -> str:
        """time peek"""
        return "x"

    def change_time() -> str:
        """time change"""
        return "x"
    reg.register(peek, tier=LOOK)
    reg.register(change_time, tier=CHANGE)
    llm = ScriptLLM(["ok"])
    agent = Agent(llm, reg, "S", max_tier=LOOK, tool_picker=ToolPicker(BagEmbedder(), reg, k=5),
                  prompt_for_tools=lambda schemas: ",".join(s["function"]["name"] for s in schemas))
    agent.run("time")
    assert llm.systems[0] == "peek"


@pytest.mark.skipif(not config.EMBEDDER_DIR.exists(), reason="bge-small not downloaded")
def test_real_retrieval_meets_the_phase_1_bar():
    from evals.toolpick_eval import build, evaluate, load_queries
    queries = load_queries()
    assert len(queries) >= 60
    hit, _, misses = evaluate(build(), queries, k=6)
    assert hit >= 0.95, misses

