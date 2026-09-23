"""The 11 tasks from docs/tasks.md, as code. Each check looks at the real computer, not at Byte's words,
except where the task *is* a question (T02, T04, T05).

    setup ─► pre-check (already true? -> INVALID) ─► Byte runs ─► check ─► teardown (always)
"""
import json
import os
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from companion import winutil

CREATE_NEW_CONSOLE = 0x10
CREATE_NO_WINDOW = 0x08000000
SPOTIFY_APP_ID = "SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify"


@dataclass
class Ctx:
    scratch: Path
    windows_before: set[int] = field(default_factory=set)
    procs: list[subprocess.Popen] = field(default_factory=list)
    state: dict = field(default_factory=dict)

    def new_windows(self, *needles: str) -> list[winutil.Window]:
        return [w for w in winutil.windows_matching(*needles) if w.hwnd not in self.windows_before]


Check = Callable[[Ctx, str], tuple[bool, str]]


@dataclass
class Task:
    id: str
    slug: str
    ask: str
    check: Check
    setup: Callable[[Ctx], None] | None = None
    teardown: Callable[[Ctx], None] | None = None
    allow: frozenset[str] = frozenset()    # tier-2 tools the runner approves for this task
    timeout_s: float = 120.0
    precheck: bool = True                  # False for questions, where "already true" is meaningless

    def prompt(self, ctx: Ctx) -> str:
        return self.ask.format(scratch=ctx.scratch)


# --- helpers ----------------------------------------------------------------------------------------
def wait_for(pred: Callable[[], bool], timeout: float = 15.0, step: float = 0.3) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(step)
    return pred()


def kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True,
                       creationflags=CREATE_NO_WINDOW)


def close_new(ctx: Ctx, *needles: str) -> None:
    for w in ctx.new_windows(*needles):
        winutil.close_window(w.hwnd)


def spotify_windows() -> list[winutil.Window]:
    import psutil
    out = []
    for w in winutil.list_windows():
        try:
            if psutil.Process(w.pid).name().lower() == "spotify.exe":
                out.append(w)
        except psutil.Error:
            pass
    return out


def spotify_playing() -> bool:
    # While a track plays, Spotify's window title is "Artist - Song"; paused it's just "Spotify ...".
    return any(" - " in w.title for w in spotify_windows())


# --- T01 open-project -------------------------------------------------------------------------------
def t01_check(ctx, answer):
    # Any CogniHire VS Code window counts (VS Code reuses an open one); the pre-check makes the task
    # INVALID if it was already open. VS Code is slow to show its window, so wait for it.
    def found():
        return winutil.windows_matching("cognihire", "visual studio code")
    if answer:  # not during the pre-check, when nothing has been asked yet
        wait_for(lambda: bool(found()), 15)
    return bool(found()), f"CogniHire VS Code windows: {[w.title for w in found()]}"


def t01_teardown(ctx):
    close_new(ctx, "cognihire", "visual studio code")


# --- T02 explain-screen-error -----------------------------------------------------------------------
CRASH = '''import json

def load_settings(raw):
    settings = json.loads(raw)
    if not str(settings["port"]).isdigit():
        raise ValueError(f"bad config value 'port': {settings['port']!r}")
    return settings

load_settings('{"host": "localhost", "port": "eighty"}')
'''


def t02_setup(ctx):
    (ctx.scratch / "server_config.py").write_text(CRASH, encoding="utf-8")
    ctx.procs.append(subprocess.Popen(["cmd", "/k", "title Byte Suite Error && python server_config.py"],
                                      cwd=ctx.scratch, creationflags=CREATE_NEW_CONSOLE))
    wait_for(lambda: bool(winutil.windows_matching("byte suite error")), 10)
    time.sleep(1.5)


def t02_check(ctx, answer):
    low = answer.lower()  # naming the exception, or explaining its cause, both show Byte read the screen
    return "valueerror" in low or ("port" in low and "eighty" in low), "answer must name ValueError or its cause"


def t02_teardown(ctx):
    for p in ctx.procs:
        kill_tree(p)


# --- T03 play-music ---------------------------------------------------------------------------------
def t03_setup(ctx):
    ctx.state["spotify_was_running"] = bool(spotify_windows())
    if not spotify_windows():
        subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{SPOTIFY_APP_ID}"])
        wait_for(lambda: bool(spotify_windows()), 20)
        time.sleep(3)
    if spotify_playing():
        winutil.press_key(winutil.VK_MEDIA_PLAY_PAUSE)
        wait_for(lambda: not spotify_playing(), 5)


def t03_check(ctx, answer):
    return spotify_playing(), f"spotify titles: {[w.title for w in spotify_windows()]}"


def t03_teardown(ctx):
    if spotify_playing():
        winutil.press_key(winutil.VK_MEDIA_PLAY_PAUSE)
    if not ctx.state.get("spotify_was_running"):
        for w in spotify_windows():
            winutil.close_window(w.hwnd)


# --- T04 newest-download ----------------------------------------------------------------------------
def newest_download() -> Path | None:
    files = [p for p in (Path.home() / "Downloads").iterdir() if p.is_file()]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def t04_check(ctx, answer):
    newest = newest_download()
    if newest is None:
        return False, "Downloads is empty"
    low = answer.lower()
    return (newest.name.lower() in low or newest.stem.lower() in low), f"newest is {newest.name!r}"


# --- T05 latest-pytorch -----------------------------------------------------------------------------
def t05_check(ctx, answer):
    with urllib.request.urlopen("https://pypi.org/pypi/torch/json", timeout=15) as r:
        version = json.load(r)["info"]["version"]
    major_minor = ".".join(version.split(".")[:2])
    ok = re.search(rf"(?<![\d.]){re.escape(major_minor)}(?![\d])", answer) is not None
    return ok, f"PyPI says {version}; looking for {major_minor}"


# --- T06 add-todo -----------------------------------------------------------------------------------
TODO_BEFORE = "- buy milk\n- finish the CogniHire report\n"


def t06_setup(ctx):
    (ctx.scratch / "todo.txt").write_text(TODO_BEFORE, encoding="utf-8")


def t06_check(ctx, answer):
    text = (ctx.scratch / "todo.txt").read_text(encoding="utf-8").lower()
    ok = "study cn unit 3" in text and "buy milk" in text and "cognihire report" in text
    return ok, f"todo.txt now: {text[:200]!r}"


# --- T07 close-window -------------------------------------------------------------------------------
def t07_setup(ctx):
    folder = ctx.scratch / "close-me"
    folder.mkdir(exist_ok=True)
    os.startfile(folder)  # noqa: S606  an Explorer window we own
    wait_for(lambda: bool(ctx.new_windows("close-me")), 10)


def t07_check(ctx, answer):
    left = ctx.new_windows("close-me")
    return not left, f"still open: {[w.title for w in left]}"


def t07_teardown(ctx):
    close_new(ctx, "close-me")


# --- T08 set-volume ---------------------------------------------------------------------------------
def t08_setup(ctx):
    ctx.state["volume"] = winutil.get_volume()
    winutil.set_volume(0.5)


def t08_check(ctx, answer):
    v = winutil.get_volume()
    return 0.28 <= v <= 0.32, f"volume is {v:.2f}"


def t08_teardown(ctx):
    if "volume" in ctx.state:
        winutil.set_volume(ctx.state["volume"])


# --- T09 dismiss-dialog -----------------------------------------------------------------------------
def dialog_open() -> bool:
    return any(w.title == "Byte Suite" for w in winutil.list_windows())


def t09_setup(ctx):
    # The dialog writes which button closed it; "the window vanished" alone gave a false pass in the baseline.
    result = ctx.scratch / "dialog_result.txt"
    script = ("Add-Type -AssemblyName PresentationFramework; "
              "$r = [System.Windows.MessageBox]::Show('This is a Byte test dialog.', 'Byte Suite'); "
              f"Set-Content -Path '{result}' -Value $r")
    ctx.procs.append(subprocess.Popen(["powershell", "-NoProfile", "-Command", script],
                                      creationflags=CREATE_NO_WINDOW))
    wait_for(dialog_open, 15)


def t09_check(ctx, answer):
    result = ctx.scratch / "dialog_result.txt"
    clicked = result.read_text(encoding="utf-8", errors="replace").strip() if result.exists() else ""
    return clicked == "OK" and not dialog_open(), f"dialog closed with: {clicked or 'nothing'}"


def t09_teardown(ctx):
    for p in ctx.procs:
        kill_tree(p)


# --- T10 notepad-save -------------------------------------------------------------------------------
def t10_check(ctx, answer):
    f = ctx.scratch / "hello.txt"
    if not f.exists():
        return False, "hello.txt does not exist"
    text = f.read_text(encoding="utf-8", errors="replace").strip()
    return text == "hello", f"hello.txt contains {text[:60]!r}"


def t10_teardown(ctx):
    close_new(ctx, "hello", "notepad")


# --- T11 telegram-video -----------------------------------------------------------------------------
def telegram_windows() -> list[winutil.Window]:
    import psutil
    out = []
    for w in winutil.list_windows():
        try:
            if psutil.Process(w.pid).name().lower() == "telegram.exe":
                out.append(w)
        except psutil.Error:
            pass
    return out


def t11_setup(ctx):
    ctx.state["telegram_windows"] = {w.hwnd for w in telegram_windows()}


def t11_check(ctx, answer):
    # A playing video makes sound, so Telegram gets an active audio session; the chat list alone never does.
    playing = wait_for(lambda: winutil.playing_audio("Telegram.exe"), 5)
    return playing, f"telegram audio active: {playing}; windows: {[w.title for w in telegram_windows()]}"


def t11_teardown(ctx):
    for w in telegram_windows():  # the media viewer, and the main window if Byte opened Telegram
        if w.hwnd not in ctx.state.get("telegram_windows", set()):
            winutil.close_window(w.hwnd)


TASKS: list[Task] = [
    Task("T01", "open-project", "Open my CogniHire project in VS Code.", t01_check,
         teardown=t01_teardown, timeout_s=90),
    Task("T02", "explain-screen-error", "There's an error on my screen. What is it?", t02_check,
         setup=t02_setup, teardown=t02_teardown, precheck=False),
    Task("T03", "play-music", "Play some music on Spotify.", t03_check,
         setup=t03_setup, teardown=t03_teardown, timeout_s=90),
    Task("T04", "newest-download", "What's the newest file in my Downloads folder?", t04_check,
         precheck=False),
    Task("T05", "latest-pytorch", "What's the latest version of PyTorch?", t05_check, precheck=False),
    Task("T06", "add-todo", "Add 'study CN unit 3' to my todo list at {scratch}\\todo.txt.", t06_check,
         setup=t06_setup, allow=frozenset({"write_text_file"})),
    Task("T07", "close-window", "Close the 'close-me' folder window.", t07_check,
         setup=t07_setup, teardown=t07_teardown, allow=frozenset({"close_window"})),
    Task("T08", "set-volume", "Set my volume to 30%.", t08_check,
         setup=t08_setup, teardown=t08_teardown),
    Task("T09", "dismiss-dialog", "A dialog popped up. Click OK on it.", t09_check,
         setup=t09_setup, teardown=t09_teardown, allow=frozenset({"click"})),
    Task("T10", "notepad-save", "Open Notepad, type hello, and save it as {scratch}\\hello.txt.", t10_check,
         teardown=t10_teardown, allow=frozenset({"write_text_file", "type_text", "click", "press_keys"})),
    Task("T11", "telegram-video", "Play the latest video in my Telegram Saved Messages.", t11_check,
         setup=t11_setup, teardown=t11_teardown, allow=frozenset({"click", "press_keys"}), timeout_s=150),
]
