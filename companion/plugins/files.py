"""Files and folders: look inside, read, write, open."""
import os
import subprocess
import sys
from typing import Annotated

from companion.paths import resolve
from companion.tools import CHANGE, LOOK, OPEN, ToolError

ORDER = 20


def _open(target: str) -> None:
    if sys.platform == "win32":
        os.startfile(target)  # noqa: S606  (opens with the default app, like double-clicking)
    else:
        subprocess.Popen(["xdg-open", target])


def list_directory(path: Annotated[str, "Folder path. Relative paths start from the user's home folder, e.g. 'Downloads'"],
                   sort: Annotated[str, "'name', or 'newest' to put the most recently changed first"] = "name") -> str:
    """List the files and folders inside a folder on this computer, with when each was last changed.
    Use sort='newest' to find the newest/latest/most recent file."""
    import datetime as dt
    folder = resolve(path)
    if not folder.is_dir():
        raise ToolError(f"Not a folder: {folder}")

    def mtime(p):
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0
    entries = list(folder.iterdir())
    if sort.strip().lower() in ("newest", "latest", "recent", "date", "modified", "time"):
        entries.sort(key=mtime, reverse=True)
        order = "newest first"
    else:
        entries.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
        order = "folders first, then by name"
    n_dirs = sum(1 for e in entries if e.is_dir())
    # Small models misread vague labels like "228 items" as "228 folders", so spell it out.
    lines = [f"{folder} contains {n_dirs} folders and {len(entries) - n_dirs} files ({order})"]
    if order == "newest first":
        # Spell out the answer to "what's the newest file": small models otherwise name the newest folder.
        newest_file = next((e for e in entries if e.is_file()), None)
        newest_dir = next((e for e in entries if e.is_dir()), None)
        if newest_file:
            lines.append(f"NEWEST FILE: {newest_file.name}")
        if newest_dir:
            lines.append(f"NEWEST FOLDER: {newest_dir.name}")
    for entry in entries[:200]:
        when = dt.datetime.fromtimestamp(mtime(entry)).strftime("%Y-%m-%d %H:%M")
        lines.append(f"  [dir]  {entry.name}  ({when})" if entry.is_dir() else f"  [file] {entry.name}  ({when})")
    if len(entries) > 200:
        lines.append(f"  ...and {len(entries) - 200} more")
    return "\n".join(lines)


def read_text_file(path: Annotated[str, "File path. Relative paths start from the user's home folder"],
                   max_chars: Annotated[int, "Maximum characters to return"] = 3000) -> str:
    """Read the contents of a text file on this computer."""
    file = resolve(path)
    if not file.is_file():
        raise ToolError(f"Not a file: {file}")
    text = file.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + f"\n...[showing {max_chars} of {len(text)} chars]"
    return text


def open_folder(path: Annotated[str, "Folder path; relative paths start from the home folder, e.g. 'Downloads'"]) -> str:
    """Open a folder in File Explorer so the user can see it."""
    folder = _dry_open_folder(path).removeprefix("Opened ")
    _open(folder)
    return f"Opened {folder}"


def _dry_open_folder(path: str) -> str:
    folder = resolve(path)
    if not folder.is_dir():
        raise ToolError(f"Not a folder: {folder}")
    return f"Opened {folder}"


def open_file(path: Annotated[str, "File path; relative paths start from the home folder"]) -> str:
    """Open a file with its default program (like double-clicking it), e.g. play a video or show a PDF."""
    file = _dry_open_file(path).removeprefix("Opened ")
    _open(file)
    return f"Opened {file}"


def _dry_open_file(path: str) -> str:
    file = resolve(path)
    if not file.is_file():
        raise ToolError(f"Not a file: {file}")
    return f"Opened {file}"


def _check_write(file, content: str, append: bool, overwrite: bool) -> str:
    """Refuse to silently wipe an existing file: adding an item to a list must not lose the list."""
    if append and file.exists() and file.stat().st_size and not content.startswith("\n"):
        existing = file.read_text(encoding="utf-8", errors="replace")
        if not existing.endswith("\n"):
            content = "\n" + content  # the new line goes on its own line
    if not append and not overwrite and file.exists() and file.stat().st_size:
        raise ToolError(f"Nothing was written: {file} already has content, and replacing it would delete "
                        "what's there. To ADD your text, call write_text_file again with the same path and "
                        "content and append=true. Never use overwrite=true unless the user said to replace or "
                        "erase the file.")
    return content


def write_text_file(path: Annotated[str, "File path; relative paths start from the home folder"],
                    content: Annotated[str, "The text to write (or to add, when appending)"],
                    append: Annotated[bool, "Add to the end of the file (use this to add items to a list)"] = False,
                    overwrite: Annotated[bool, "Replace an existing file's whole content"] = False) -> str:
    """Create a new text file, or add to the end of one. To add an item to a list, todo or notes file,
    ALWAYS use append=true. overwrite=true erases the old content: only when the user asks for that."""
    file = resolve(path)
    content = _check_write(file, content, append, overwrite)
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "a" if append else "w", encoding="utf-8") as f:
        f.write(content)
    return f"{'Appended to' if append else 'Wrote'} {file} ({len(content)} chars)"


def _dry_write_text_file(path: str, content: str, append: bool = False, overwrite: bool = False) -> str:
    file = resolve(path)
    content = _check_write(file, content, append, overwrite)
    return f"{'Appended to' if append else 'Wrote'} {file} ({len(content)} chars)"


def register(registry, deps) -> None:
    registry.register(list_directory, tier=LOOK, examples=(
        "what's in my Downloads folder", "show the files on my desktop", "newest file in downloads", "which folders are inside this directory"))
    registry.register(read_text_file, tier=LOOK, examples=(
        "read my notes.txt", "what does this file say", "show me the contents of todo.txt"))
    registry.register(open_folder, tier=OPEN, dry=_dry_open_folder, examples=(
        "open my Documents folder", "show me the downloads folder in explorer"))
    # Opening an arbitrary file can run it (.exe, .bat), so it's a change, not just "open".
    registry.register(open_file, tier=CHANGE, dry=_dry_open_file, examples=(
        "open this pdf", "play the video file movie.mp4", "open report.docx"))
    registry.register(write_text_file, tier=CHANGE, dry=_dry_write_text_file, examples=(
        "add an item to my todo list", "save this to a text file", "write a note to notes.txt"))
