"""Byte editing itself: every rule that keeps that safe, tested against a throwaway copy of the repo."""
import pytest

from companion.plugins import Deps, load_plugins
from companion.plugins import selfcode as sc
from companion.tools import CHANGE, LOOK, ToolError, ToolRegistry


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "companion" / "plugins").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "companion" / "greet.py").write_text('def hello():\n    return "hi"\n', encoding="utf-8")
    (root / "companion" / "agent.py").write_text("# safety core\n", encoding="utf-8")
    (root / "companion" / "plugins" / "web.py").write_text("registry.register(web_search, tier=LOOK)\n", encoding="utf-8")
    (root / "tests" / "test_x.py").write_text("def test(): pass\n", encoding="utf-8")
    monkeypatch.setattr(sc, "REPO", root)
    monkeypatch.setattr(sc, "BACKUPS", tmp_path / "backups")
    monkeypatch.setattr(sc, "USER_PLUGINS", tmp_path / "plugins")
    monkeypatch.setattr(sc, "run_tests", lambda: (True, "all passed"))
    return root


def test_edit_is_kept_when_tests_pass_and_can_be_undone(repo):
    out = sc.edit_own_code("companion/greet.py", 'return "hi"', 'return "hey there"', "friendlier")
    assert "all tests pass" in out
    assert "hey there" in (repo / "companion" / "greet.py").read_text(encoding="utf-8")
    assert "Undid" in sc.undo_self_edit()
    assert 'return "hi"' in (repo / "companion" / "greet.py").read_text(encoding="utf-8")
    assert "no change of mine" in sc.undo_self_edit()


def test_edit_is_rolled_back_when_tests_fail(repo, monkeypatch):
    monkeypatch.setattr(sc, "run_tests", lambda: (False, "1 failed: test_greeting"))
    out = sc.edit_own_code("companion/greet.py", 'return "hi"', "return 42", "oops")
    assert "undone automatically" in out and "test_greeting" in out
    assert 'return "hi"' in (repo / "companion" / "greet.py").read_text(encoding="utf-8")
    assert "no change of mine" in sc.undo_self_edit()   # a rolled-back edit leaves nothing to undo


@pytest.mark.parametrize("path", ["companion/agent.py", "tests/test_x.py", "companion/plugins/selfcode.py",
                                  "companion/config.py", "evals/tasks.py"])
def test_safety_core_and_tests_are_never_editable(repo, path):
    with pytest.raises(ToolError, match="protected"):
        sc.edit_own_code(path, "a", "b", "try")


def test_cannot_escape_the_repo_or_change_tiers(repo):
    with pytest.raises(ToolError, match="inside"):
        sc.edit_own_code("../../Windows/win.ini", "a", "b", "x")
    # touching the line is fine as long as the tier stays the same...
    assert "all tests pass" in sc.edit_own_code("companion/plugins/web.py", "tier=LOOK",
                                                "tier=LOOK, examples=()", "harmless")
    with pytest.raises(ToolError, match="tier"):   # ...changing it is not
        sc.edit_own_code("companion/plugins/web.py", "tier=LOOK", "tier=OPEN", "sneaky")


def test_old_text_must_be_unique(repo):
    (repo / "companion" / "greet.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    with pytest.raises(ToolError, match="exactly once"):
        sc.edit_own_code("companion/greet.py", "x = 1", "x = 2", "r")


GOOD_PLUGIN = '''
from companion.tools import LOOK
def register(registry, deps):
    def coin_flip() -> str:
        """Flip a coin."""
        return "heads"
    registry.register(coin_flip, tier=LOOK)   # claims to be harmless
'''


def test_written_plugin_loads_but_always_asks(repo, tmp_path):
    out = sc.write_plugin("coins", GOOD_PLUGIN)
    assert "coin_flip" in out
    reg = ToolRegistry()
    load_plugins(reg, Deps(), folder=tmp_path / "plugins", min_tier=CHANGE)
    assert reg.tier("coin_flip") == CHANGE   # a self-written tool can't skip Allow/Deny


@pytest.mark.parametrize("code,why", [
    ("def register(registry, deps):\n    raise RuntimeError('boom')\n", "boom"),
    ("x = 1\n", "register"),
    ("def register(registry, deps):\n    def calculate() -> str:\n        'dupe'\n        return 'x'\n"
     "    registry.register(calculate)\n", "already registered"),
])
def test_broken_or_clashing_plugins_are_removed(repo, tmp_path, code, why):
    out = sc.write_plugin("bad", code)
    assert "removed" in out and why in out
    assert not (tmp_path / "plugins" / "bad.py").exists()


def test_plugin_names_are_checked(repo):
    with pytest.raises(ToolError):
        sc.write_plugin("../evil", GOOD_PLUGIN)


def test_restart_needs_the_app():
    reg = ToolRegistry()
    load_plugins(reg, Deps(), include={"selfcode"})
    assert reg.tier("edit_own_code") == CHANGE and reg.tier("read_own_code") == LOOK
    with pytest.raises(ToolError, match="close and reopen"):
        reg.call("restart_byte", {})
    fired = []
    reg2 = ToolRegistry()
    load_plugins(reg2, Deps(restart=lambda: fired.append(1)), include={"selfcode"})
    reg2.call("restart_byte", {})
    assert fired == [1]
