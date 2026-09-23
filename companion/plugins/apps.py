"""Installed apps: find them in the Start menu and start them."""
import re
import subprocess
import sys
from typing import Annotated

from companion.tools import OPEN, ToolError

ORDER = 40
SPOTIFY_URI = "spotify:"

APP_ALIASES = {"vscode": "visualstudiocode", "vs": "visualstudiocode", "code": "visualstudiocode",
               "chrome": "googlechrome", "edge": "microsoftedge", "browser": "microsoftedge",
               "explorer": "fileexplorer", "files": "fileexplorer", "cmd": "commandprompt",
               "settings": "settings", "calc": "calculator", "whatsapp": "whatsapp",
               "telegram": "telegramdesktop"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def best_app_match(name: str, apps: list[tuple[str, str]]) -> tuple[str, str] | None:
    """apps: (display name, AppID). Exact name beats prefix beats substring; shorter names win ties."""
    wanted = _norm(name)
    wanted = APP_ALIASES.get(wanted, wanted)
    if not wanted:
        return None
    best: tuple[int, tuple[str, str]] | None = None
    for app in apps:
        label = _norm(app[0])
        if not label or "uninstall" in label or app[1].startswith("http"):
            continue
        if app[0].strip().lower() == name.strip().lower():
            score = -1  # "Notepad" must beat "Notepad++", which normalises to the same letters
        elif label == wanted:
            score = 0
        elif label.startswith(wanted):
            score = 1 + len(label) - len(wanted)
        elif wanted in label:
            score = 100 + len(label)
        else:
            continue
        if best is None or score < best[0]:
            best = (score, app)
    return best[1] if best else None


_start_apps_cache: list[tuple[str, str]] | None = None


def start_apps() -> list[tuple[str, str]]:
    """Everything in the Start menu, including Store apps that have no .lnk file."""
    global _start_apps_cache
    if _start_apps_cache is None:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-StartApps | ForEach-Object { $_.Name + \"`t\" + $_.AppID }"],
                             capture_output=True, text=True, timeout=20,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        _start_apps_cache = [tuple(l.split("\t", 1)) for l in out.stdout.splitlines() if "\t" in l]
    return _start_apps_cache


def _find(name: str) -> tuple[str, str]:
    match = best_app_match(name, start_apps()) if sys.platform == "win32" else None
    if match is None:
        raise ToolError(f"Couldn't find an app called '{name}' in the Start menu.")
    return match


def launch_app(name: Annotated[str, "App name as it appears in the Start menu, e.g. 'Spotify', 'Visual Studio Code'"]) -> str:
    """Start an installed application from the Start menu."""
    match = _find(name)
    subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{match[1]}"])
    return f"Started {match[0]}"


def _dry_launch_app(name: str) -> str:
    return f"Started {_find(name)[0]}"


def register(registry, deps) -> None:
    registry.register(launch_app, tier=OPEN, dry=_dry_launch_app, examples=(
        "open Spotify", "start VS Code", "launch Telegram", "open notepad", "open the calculator app", "fire up a game", "run an app"))
