"""Seeing the screen (phase 3). Read-only: a screenshot goes to the vision model, nothing is clicked.
Loads only when the brain can see (a profile with an mmproj file)."""
import io
import os
from typing import Annotated

from companion.tools import LOOK, ToolError, ToolImage

ORDER = 15
MAX_SIDE = 1280   # P1: long side at most this many pixels; enough to read text, cheap in tokens


def capture(whole_screen: bool = False):
    """Returns (PIL image, what it shows)."""
    from PIL import ImageGrab

    from companion import winutil
    winutil.make_dpi_aware()
    if whole_screen:
        return ImageGrab.grab(all_screens=False), "the whole screen"
    w = winutil.front_window(exclude_pids={os.getpid()})
    if w is None:
        return ImageGrab.grab(all_screens=False), "the whole screen (no window was open)"
    box = winutil.window_rect(w.hwnd)
    return ImageGrab.grab(bbox=box, all_screens=True), f"the window '{w.title[:80]}'"


def shrink(img, max_side: int = MAX_SIDE):
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    return img


def look_at_screen(whole_screen: Annotated[bool, "True for the whole screen instead of the front window"] = False) -> "ToolImage | str":
    """Take a screenshot of the window the user is working in (or the whole screen) so you can SEE it:
    read errors, text, what's open. Use it whenever the user mentions their screen or 'this'."""
    try:
        img, what = capture(whole_screen)
    except OSError as e:
        raise ToolError(f"Couldn't take a screenshot: {e}")
    img = shrink(img.convert("RGB"))
    if is_blank(img):
        # Netflix, Prime and other DRM-protected windows capture as solid black. The whole screen may still
        # show the other windows; if that is black too, say why instead of sending a black picture.
        if whole_screen:
            return _blank_note(what)
        whole = shrink(capture(whole_screen=True)[0].convert("RGB"))
        if is_blank(whole):
            return _blank_note(what)
        img, what = whole, f"the whole screen ({what} itself showed up black: protected content)"
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return ToolImage(buf.getvalue(), f"Screenshot of {what} ({img.width}x{img.height}).")


def is_blank(img) -> bool:
    from PIL import ImageStat
    stat = ImageStat.Stat(img.convert("L"))
    return stat.mean[0] < 4 and stat.stddev[0] < 3


def _blank_note(what: str) -> str:
    return (f"The screenshot of {what} came out completely black. That happens with protected video apps "
            "(Netflix, Prime Video) or a minimised window: you can't see its contents. Tell the user that, and "
            "ask them to bring the right window to the front or describe what they see.")


def register(registry, deps) -> None:
    if not getattr(deps, "vision", False):
        return
    registry.register(look_at_screen, tier=LOOK, examples=(
        "there's an error on my screen, what is it", "what am I looking at", "read what's on my screen",
        "what does this window say", "look at my screen"))
