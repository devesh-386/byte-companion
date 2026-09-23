"""Byte working on its own code, with hard limits.

    read_own_code / search_own_code         look (tier 0)
    edit_own_code  (Allow/Deny)  ─► backup ─► apply ─► run ALL tests ─► pass: keep · fail: undo automatically
    write_plugin   (Allow/Deny)  ─► new file in ~/.companion/plugins ─► must load cleanly, else removed
    undo_self_edit (Allow/Deny)  ─► puts the last change back
    restart_byte   (Allow/Deny)  ─► relaunch so changes take effect

Never editable, whatever the model says: the safety core (tiers, Allow/Deny, stop hotkey, action log),
this file, the plugin loader, config, the tests and the task suite. Otherwise a model could switch off
its own brakes, or "fix" a failing test instead of the code.
"""
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Annotated

from companion.tools import CHANGE, LOOK, ToolError

ORDER = 90
REPO = Path(__file__).resolve().parents[2]
USER_PLUGINS = Path(os.environ.get("COMPANION_HOME", Path.home() / ".companion")) / "plugins"
BACKUPS = Path(os.environ.get("COMPANION_HOME", Path.home() / ".companion")) / "self-edits"
EDITABLE_ROOTS = ("companion", "training", "docs")
PROTECTED = {
    "companion/agent.py", "companion/tools.py", "companion/cancel.py", "companion/hotkey.py",
    "companion/actionlog.py", "companion/config.py", "companion/plugins/__init__.py",
    "companion/plugins/selfcode.py",
}
PROTECTED_DIRS = ("tests/", "evals/")
TEST_TIMEOUT_S = 600
_lock = threading.Lock()


def _rel(path: str) -> tuple[Path, str]:
    p = (REPO / path.strip().strip('"').replace("\\", "/")).resolve()
    try:
        rel = p.relative_to(REPO).as_posix()
    except ValueError:
        raise ToolError("Only files inside Byte's own code folder can be read or changed.")
    return p, rel


def _check_editable(rel: str) -> None:
    if rel in PROTECTED or rel.startswith(PROTECTED_DIRS):
        raise ToolError(f"{rel} is protected: it holds Byte's safety rules or its tests, so Byte can never "
                        "change it. Tell the user; they can edit it themselves.")
    if not rel.startswith(tuple(r + "/" for r in EDITABLE_ROOTS)) or not rel.endswith((".py", ".md", ".txt", ".yaml")):
        raise ToolError(f"{rel} isn't an editable code or text file of Byte's.")


def run_tests() -> tuple[bool, str]:
    """The whole test suite, the same one the user runs. Returns (passed, last lines of output)."""
    try:
        out = subprocess.run([sys.executable.replace("pythonw.exe", "python.exe"), "-m", "pytest", "-q", "-x",
                              "-p", "no:cacheprovider", "tests"], cwd=REPO, capture_output=True, text=True,
                             timeout=TEST_TIMEOUT_S, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return False, f"tests took longer than {TEST_TIMEOUT_S} s"
    tail = "\n".join((out.stdout + out.stderr).strip().splitlines()[-15:])
    return out.returncode == 0, tail


def _tiers(text: str) -> list[str]:
    return re.findall(r"\btier\s*=\s*[\w.]+|\b(?:LOOK|OPEN|CHANGE|NEVER)\b", text)


def _backup(p: Path, rel: str, reason: str) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder = BACKUPS / stamp
    folder.mkdir(parents=True, exist_ok=True)
    if p.exists():
        shutil.copy2(p, folder / "before")
    (folder / "meta.json").write_text(json.dumps({"file": rel, "existed": p.exists(), "reason": reason,
                                                  "when": stamp}, indent=1), encoding="utf-8")
    return folder


# --- looking --------------------------------------------------------------------------------------
def search_own_code(text: Annotated[str, "Text or a regular expression to look for in Byte's code"]) -> str:
    """Search Byte's own source code (the companion folder) and show matching lines with file:line."""
    try:
        rx = re.compile(text, re.I)
    except re.error:
        rx = re.compile(re.escape(text), re.I)
    hits = []
    for f in sorted((REPO / "companion").rglob("*.py")):
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if rx.search(line):
                hits.append(f"{f.relative_to(REPO).as_posix()}:{i}: {line.strip()[:120]}")
                if len(hits) >= 40:
                    return "\n".join(hits) + "\n...(more)"
    return "\n".join(hits) or "No matches."


def read_own_code(path: Annotated[str, "File inside Byte's code folder, e.g. 'companion/plugins/web.py'"],
                  start_line: Annotated[int, "First line to show"] = 1,
                  end_line: Annotated[int, "Last line to show"] = 200) -> str:
    """Read part of one of Byte's own source files, with line numbers."""
    p, rel = _rel(path)
    if not p.is_file():
        raise ToolError(f"No file {rel}. Use search_own_code to find the right one.")
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    start, end = max(1, start_line), min(len(lines), end_line)
    body = "\n".join(f"{i:4} {lines[i - 1]}" for i in range(start, end + 1))
    return f"{rel} (lines {start}-{end} of {len(lines)}){' [PROTECTED: read-only]' if rel in PROTECTED else ''}\n{body}"


# --- changing (all tier 2: the user sees and approves each one) -----------------------------------
def edit_own_code(path: Annotated[str, "File to change, e.g. 'companion/plugins/web.py'"],
                  old_text: Annotated[str, "The exact existing text to replace (must appear exactly once)"],
                  new_text: Annotated[str, "The text to put in its place"],
                  reason: Annotated[str, "One line: why this change"]) -> str:
    """Change Byte's own code. The change is kept only if every test still passes; otherwise it is undone
    automatically. It takes effect after restart_byte."""
    p, rel = _rel(path)
    _check_editable(rel)
    if not p.is_file():
        raise ToolError(f"No file {rel}.")
    with _lock:
        before = p.read_text(encoding="utf-8")
        if _tiers(old_text) != _tiers(new_text):
            raise ToolError("That edit changes a tool's tier (how much it's allowed to do without asking). "
                            "Tiers are safety settings: only the user may change them.")
        count = before.count(old_text)
        if count != 1:
            raise ToolError(f"old_text appears {count} times in {rel}; it must appear exactly once. "
                            "Read the file and copy a longer, unique piece.")
        backup = _backup(p, rel, reason)
        p.write_text(before.replace(old_text, new_text, 1), encoding="utf-8")
        ok, tail = run_tests()
        if not ok:
            p.write_text(before, encoding="utf-8")
            shutil.rmtree(backup, ignore_errors=True)
            return f"Tests FAILED, so the change to {rel} was undone automatically. Test output:\n{tail}"
    return f"Changed {rel}; all tests pass. Backup: {backup.name}. Restart Byte (restart_byte) to use it."


PLUGIN_TEMPLATE = '''import random                      # any imports you need; never "import registry"


def register(registry, deps):      # Byte calls this; don't call it yourself
    def flip_coin() -> str:        # every argument needs a type: str, int, float or bool
        """Flip a coin and say heads or tails."""   # the docstring tells you when to use the tool
        return random.choice(["heads", "tails"])   # return a string

    registry.register(flip_coin)'''


def write_plugin(name: Annotated[str, "Plugin name, letters/digits/underscores, e.g. 'weather'"],
                 code: Annotated[str, "Full Python source, shaped exactly like the example in this tool's description"]) -> str:
    """Give Byte a new ability as a plugin file (in ~/.companion/plugins). It must load cleanly, otherwise
    it's removed. Its tools always ask the user before running. Active after restart_byte.
    The code must follow this shape:
    import random
    def register(registry, deps):
        def flip_coin() -> str:
            \"\"\"Flip a coin and say heads or tails.\"\"\"
            return random.choice(["heads", "tails"])
        registry.register(flip_coin)"""
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,30}", name):
        raise ToolError("Plugin names are lowercase letters, digits and underscores.")
    from companion.plugins import Deps, load_plugins
    from companion.tools import ToolRegistry
    with _lock:
        USER_PLUGINS.mkdir(parents=True, exist_ok=True)
        target = USER_PLUGINS / f"{name}.py"
        backup = _backup(target, f"~plugins/{name}.py", "write_plugin")
        target.write_text(code, encoding="utf-8")
        probe = ToolRegistry(dry_run=True)
        base = ToolRegistry(dry_run=True)
        load_plugins(base, Deps())
        for n in base.names():  # the new plugin must not clash with a built-in tool
            probe._tools[n] = base.get(n)
        report = load_plugins(probe, Deps(), folder=USER_PLUGINS, include={name}, min_tier=CHANGE)
        added = report.loaded.get(name, [])
        if name in report.failed or not added:
            _restore(backup, target)
            why = report.failed.get(name, "it registered no tools")
            return (f"The plugin didn't load ({why}), so it was removed. Rewrite it in exactly this shape:\n"
                    f"{PLUGIN_TEMPLATE}")
        if re.search(r"^register\s*\(", code, re.M):
            _restore(backup, target)
            return ("Removed: the code calls register(...) itself at the bottom. Delete that line; Byte calls "
                    f"register for you. Shape:\n{PLUGIN_TEMPLATE}")
    return f"Plugin '{name}' saved with tools {added}. Restart Byte (restart_byte) to use it."


def _restore(folder: Path, target: Path) -> None:
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    if meta["existed"]:
        shutil.copy2(folder / "before", target)
    elif target.exists():
        target.unlink()
    shutil.rmtree(folder, ignore_errors=True)


def undo_self_edit() -> str:
    """Undo Byte's most recent change to its own code or plugins."""
    with _lock:
        edits = sorted(BACKUPS.glob("*/meta.json")) if BACKUPS.exists() else []
        if not edits:
            return "There's no change of mine to undo."
        folder = edits[-1].parent
        meta = json.loads(edits[-1].read_text(encoding="utf-8"))
        rel = meta["file"]
        target = USER_PLUGINS / rel.split("/", 1)[1] if rel.startswith("~plugins/") else REPO / rel
        _restore(folder, target)
    return f"Undid my change to {rel}. Restart Byte (restart_byte) for it to take effect."


def register(registry, deps) -> None:
    def restart_byte() -> str:
        """Restart Byte so changes to its code or plugins take effect (takes about 30 seconds)."""
        if deps.restart is None:
            raise ToolError("I can't restart myself from here; close and reopen me.")
        deps.restart()
        return "Restarting now. I'll be back in about 30 seconds."

    registry.register(search_own_code, tier=LOOK, examples=(
        "where in your code do you handle web search", "find the function that sets the volume"))
    registry.register(read_own_code, tier=LOOK, examples=(
        "show me your own code for the memory tools", "read your file companion/voice.py"))
    registry.register(edit_own_code, tier=CHANGE, examples=(
        "change your code so that", "fix your own bug in", "improve yourself: make the reminder message friendlier"))
    registry.register(write_plugin, tier=CHANGE, examples=(
        "teach yourself a new tool", "write a plugin that gives you a new ability", "add a new skill to yourself"))
    registry.register(undo_self_edit, tier=CHANGE, examples=(
        "undo your last change to yourself", "revert the change you made to your code"))
    registry.register(restart_byte, tier=CHANGE, examples=(
        "restart yourself", "reload your code"))
