"""Acting on the desktop (phase 4). Direct commands first (media keys, volume, VS Code, Spotify), then
UI Automation for everything else: observe_window lists what's clickable, click / type_text / press_keys
act on it and report what changed."""
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Annotated

from companion import uia, winutil
from companion.paths import resolve
from companion.tools import CHANGE, LOOK, OPEN, ToolError

ORDER = 45
WM_APPCOMMAND, APPCOMMAND_MEDIA_PLAY_PAUSE = 0x0319, 14
MEDIA_KEYS = {"play": 0xB3, "pause": 0xB3, "play_pause": 0xB3, "next": 0xB0, "previous": 0xB1,
              "mute": 0xAD, "stop": 0xB2}
SEARCH_ROOTS = [Path.home(), Path("C:/claude"), Path("D:/")]
SKIP_DIRS = {"node_modules", ".git", "AppData", "__pycache__", ".venv", "venv", "$RECYCLE.BIN",
             "System Volume Information", "Windows", "Program Files", "Program Files (x86)"}

_observer = uia.Observer()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _find_window(title: str) -> winutil.Window:
    if not title.strip():
        w = winutil.front_window(exclude_pids={os.getpid()})
        if w is None:
            raise ToolError("No window is open.")
        return w
    wanted = _norm(title)
    mine = [w for w in winutil.list_windows() if w.pid != os.getpid() and w.title.strip()]
    wins = [w for w in mine if wanted in _norm(w.title)]
    if not wins:
        # Models guess titles ("Saved Messages - Devesh" for "Saved Messages – (502)"): take the window
        # sharing the most words with the guess.
        words = set(re.findall(r"[a-z0-9]{3,}", title.lower()))
        scored = [(len(words & set(re.findall(r"[a-z0-9]{3,}", w.title.lower()))), w) for w in mine]
        best = max((s for s, _ in scored), default=0)
        wins = [w for s, w in scored if s == best and best > 0]
    if not wins:
        titles = [w.title for w in mine][:15]
        raise ToolError(f"No window titled like '{title}'. Open windows: {titles}. "
                        "Call observe_window with one of these titles, or with no title for the front window.")
    return wins[0]  # front-most match


# --- looking --------------------------------------------------------------------------------------
def list_open_windows() -> str:
    """List the titles of the windows that are open right now, front-most first."""
    wins = [w for w in winutil.list_windows() if w.pid != os.getpid()
            and not winutil._cloaked(w.hwnd) and winutil._class_name(w.hwnd) not in ("Progman", "Shell_TrayWnd")]
    return "\n".join(f"- {w.title}" for w in wins[:30]) or "No windows are open."


def find_folder(name: Annotated[str, "Folder (or project) name to look for, e.g. 'cognihire'"]) -> str:
    """Find folders on this computer by name (home folder, C:\\claude and D:\\, a few levels deep)."""
    wanted = _norm(name)
    if not wanted:
        raise ToolError("Give a folder name to look for.")
    hits: list[Path] = []
    deadline = time.monotonic() + 6
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        stack = [(root, 0)]
        while stack and time.monotonic() < deadline:
            folder, depth = stack.pop()
            try:
                children = [c for c in folder.iterdir() if c.is_dir()]
            except OSError:
                continue
            for c in children:
                if c.name in SKIP_DIRS or c.name.startswith("."):
                    continue
                if wanted in _norm(c.name):
                    hits.append(c)
                if depth < 3:
                    stack.append((c, depth + 1))
    if not hits:
        return f"No folder named like '{name}' found."
    hits = sorted(set(hits), key=lambda p: (len(_norm(p.name)) - len(wanted), len(p.parts)))
    return "\n".join(str(h) for h in hits[:10])


def observe_window(title: Annotated[str, "Part of the window title; empty = the window in front"] = "") -> str:
    """See what's inside a window: its buttons, boxes, menus and text, each with an [id] you can
    click or type into. Call it before click/type_text, and again after the window changes."""
    w = _find_window(title)
    try:
        elements, skeleton = _observer.observe(w.hwnd, w.title)
    except uia.UIAError as e:
        raise ToolError(str(e))
    lines = [f"Window: {w.title}"]
    lines += [e.line() for e in elements]
    if skeleton:
        lines.append("(This app shows almost nothing to accessibility tools. Use look_at_screen to see it, "
                     "or a keyboard shortcut via press_keys.)")
    elif not elements:
        lines.append("(No clickable elements found.)")
    return "\n".join(lines)


# --- direct commands (tier 1) ---------------------------------------------------------------------
def media_control(action: Annotated[str, "play_pause, next, previous, mute or stop"]) -> str:
    """Control whatever music/video is playing (Spotify, YouTube, players) with the media keys."""
    key = MEDIA_KEYS.get(action.strip().lower().replace(" ", "_").replace("/", "_"))
    if key is None:
        raise ToolError(f"Unknown action '{action}'. Use: play_pause, next, previous, mute, stop.")
    winutil.press_key(key)
    return f"Pressed the {action} media key."


def set_volume(percent: Annotated[int, "Volume from 0 to 100"]) -> str:
    """Set the computer's master volume."""
    if not 0 <= percent <= 100:
        raise ToolError("Volume must be between 0 and 100.")
    winutil.set_volume(percent / 100)
    return f"Volume is now {round(winutil.get_volume() * 100)}%."


def focus_window(title: Annotated[str, "Part of the window title"]) -> str:
    """Bring a window to the front."""
    w = _find_window(title)
    user32 = winutil.user32
    if user32.IsIconic(w.hwnd):
        user32.ShowWindow(w.hwnd, 9)  # SW_RESTORE
    user32.keybd_event(0x12, 0, 0, 0)  # a tap of Alt lets us take the foreground from another app
    user32.SetForegroundWindow(w.hwnd)
    user32.keybd_event(0x12, 0, 2, 0)
    return f"'{w.title}' is in front."


def open_in_vscode(path: Annotated[str, "Folder or file to open; use find_folder first if you only know a name"]) -> str:
    """Open a folder (project) or file in Visual Studio Code."""
    p = resolve(path)
    if not p.exists():
        raise ToolError(f"{p} does not exist. Use find_folder to locate it.")
    code = shutil.which("code") or shutil.which("code.cmd")
    if code is None:
        raise ToolError("VS Code's 'code' command isn't installed.")
    subprocess.Popen([code, str(p)], creationflags=subprocess.CREATE_NO_WINDOW)
    return f"Opening {p} in VS Code."


def play_spotify(query: Annotated[str, "What to play (song, artist, playlist); empty = resume"] = "") -> str:
    """Play music on Spotify: resume, or search for something and show it."""
    from companion.plugins.apps import SPOTIFY_URI
    running = any(w for w in winutil.list_windows() if "spotify" in w.title.lower())
    if query.strip():
        os.startfile(SPOTIFY_URI + "search:" + query.strip().replace(" ", "%20"))  # noqa: S606
        return (f"Opened a Spotify search for '{query}'. To start a track, observe_window('Spotify') and "
                "click it, or press play.")
    if not running:
        os.startfile(SPOTIFY_URI)  # noqa: S606
        for _ in range(30):
            time.sleep(0.5)
            if any(" - " in w.title or w.title.lower().startswith("spotify") for w in winutil.list_windows()):
                break
        time.sleep(2)
    def playing() -> list[str]:
        return [w.title for w in winutil.list_windows() if " - " in w.title and _is_spotify(w)]

    if playing():
        return f"Spotify is already playing: {playing()[0]}"
    # Aim play/pause at Spotify itself: a plain media key goes to whichever app owns "now playing",
    # which with Netflix or YouTube open is not Spotify.
    for w in (w for w in winutil.list_windows() if _is_spotify(w)):
        winutil.user32.SendMessageW(w.hwnd, WM_APPCOMMAND, w.hwnd, APPCOMMAND_MEDIA_PLAY_PAUSE << 16)
    time.sleep(1.5)
    if not playing():
        winutil.press_key(MEDIA_KEYS["play_pause"])  # second try: the global media key
        time.sleep(1.5)
    now = playing()
    return (f"Spotify is playing: {now[0]}" if now else
            "Spotify did NOT start playing (checked its window title). Tell the user it isn't playing; "
            "they may need to press play in Spotify once.")


def _is_spotify(w) -> bool:
    try:
        import psutil
        return psutil.Process(w.pid).name().lower() == "spotify.exe"
    except Exception:  # noqa: BLE001
        return False


# --- acting (tier 2: asks Allow/Deny) -------------------------------------------------------------
def _verified(action) -> str:
    if not _observer.window_handle or not winutil.user32.IsWindow(_observer.window_handle):
        # Nothing observed yet: act on the window in front, and read it first so the result can be checked.
        w = _find_window("")
        try:
            _observer.observe(w.hwnd, w.title)
        except uia.UIAError as e:
            raise ToolError(str(e))
    hwnd, title = _observer.window_handle, _observer.window_title
    before = _observer.snapshot()
    try:
        did = action()
    except uia.UIAError as e:
        raise ToolError(str(e))
    time.sleep(0.6)
    if not winutil.user32.IsWindow(hwnd):
        return f"{did}. The window '{title}' closed."
    try:
        _observer.observe(hwnd, title)
    except uia.UIAError:
        return f"{did}. (Couldn't re-read the window to check the result.)"
    return f"{did}. Now: {uia.diff(before, _observer.snapshot())}"


def click(element_id: Annotated[int, "The [id] from observe_window"]) -> str:
    """Click a button, link, tab, list item or checkbox seen with observe_window."""
    return _verified(lambda: _observer.click(element_id))


def type_text(text: Annotated[str, "The text to type"],
              element_id: Annotated[int, "The [id] of the box to type into; 0 = wherever the cursor is"] = 0) -> str:
    """Type text into a text box seen with observe_window (or wherever the cursor is)."""
    return _verified(lambda: _observer.type_text(text, element_id or None))


def press_keys(keys: Annotated[str, "A key or shortcut, e.g. 'enter', 'ctrl+s', 'alt+f4', 'esc'"]) -> str:
    """Press a key or keyboard shortcut in the window you last observed (it is brought to the front)."""
    try:
        seq = uia.combo_to_sendkeys(keys)
    except uia.UIAError as e:
        raise ToolError(str(e))

    def do():
        if _observer.window_handle and winutil.user32.IsWindow(_observer.window_handle):
            focus_window(_observer.window_title)
            time.sleep(0.2)

        def send():
            import uiautomation as auto
            auto.SendKeys(seq, waitTime=0.1)
        uia._run_with_timeout(send, 3)
        return f"Pressed {keys}"
    return _verified(do) if _observer.window_handle else do()


def close_window(title: Annotated[str, "Part of the window title"]) -> str:
    """Close a window, like clicking its X (the app may still ask to save)."""
    w = _find_window(title)
    winutil.close_window(w.hwnd)
    time.sleep(1.0)
    return f"Closed '{w.title}'." if not winutil.user32.IsWindow(w.hwnd) else \
        f"Asked '{w.title}' to close, but it's still open (maybe a 'save?' prompt)."


def register(registry, deps) -> None:
    registry.register(list_open_windows, tier=LOOK, examples=(
        "what windows do I have open", "which apps are open right now"))
    registry.register(find_folder, tier=LOOK, examples=(
        "where is my cognihire project", "find the folder called paras", "locate my college notes folder"))
    registry.register(observe_window, tier=LOOK, examples=(
        "a dialog popped up, click OK on it", "what buttons are in this window", "click the save button",
        "play the video in telegram", "open the chat in this app"))
    registry.register(media_control, tier=OPEN, examples=(
        "pause the music", "next song", "skip this track", "mute the video"))
    registry.register(set_volume, tier=OPEN, examples=(
        "set my volume to 30%", "turn the volume down", "make it louder", "volume 80"))
    registry.register(focus_window, tier=OPEN, examples=(
        "switch to chrome", "bring VS Code to the front", "go to the telegram window"))
    registry.register(open_in_vscode, tier=OPEN, examples=(
        "open my cognihire project in VS Code", "open this folder in code", "edit the companion repo in vscode"))
    registry.register(play_spotify, tier=OPEN, examples=(
        "play some music on spotify", "play lofi on spotify", "put on some music", "resume spotify"))
    registry.register(click, tier=CHANGE, examples=(
        "click OK", "press the play button", "click on saved messages"))
    registry.register(type_text, tier=CHANGE, examples=(
        "type hello in notepad", "write my name in the search box", "fill in the box"))
    registry.register(press_keys, tier=CHANGE, examples=(
        "press ctrl+s", "hit enter", "save it with the keyboard", "press escape"))
    registry.register(close_window, tier=CHANGE, examples=(
        "close the close-me folder window", "close that window", "close notepad"))
