"""observe() / act(): see and use any app's buttons and boxes through Windows UI Automation (phase 4).

    observe(window) ─► walk the accessibility tree (depth ≤ 12, ≤ 150 kept, 2 s per call)
                   ─► "[7] Button 'OK'" lines + an id -> element map kept until the next observe
    act(id)        ─► Invoke/Toggle/Select pattern if the element has one, else a real click at its centre
                   ─► observe again and report what changed (A7: every action is verified)

Rule of thumb from the design review: a direct command beats UI Automation beats vision.
Electron/Chromium apps may show only a skeleton tree until accessibility is switched on; observe()
says so explicitly, and the agent can fall back to look_at_screen.
"""
import threading
from dataclasses import dataclass

MAX_ELEMENTS = 150
MAX_DEPTH = 12
CALL_TIMEOUT_S = 2.0
SKELETON_BELOW = 12   # fewer named elements than this in a big window = the app hides its tree

# Roles worth showing: things you can act on, plus text that explains what you see.
INTERACTIVE = {"ButtonControl", "EditControl", "CheckBoxControl", "RadioButtonControl", "ComboBoxControl",
               "ListItemControl", "MenuItemControl", "TabItemControl", "HyperlinkControl", "TreeItemControl",
               "SliderControl", "SplitButtonControl", "DataItemControl", "DocumentControl"}
TEXTUAL = {"TextControl", "TitleBarControl", "HeaderItemControl", "StatusBarControl", "ListControl",
           "WindowControl", "PaneControl", "GroupControl", "ToolBarControl", "ImageControl"}


class UIAError(RuntimeError):
    pass


@dataclass
class Element:
    id: int
    role: str
    name: str
    rect: tuple[int, int, int, int]
    control: object = None
    value: str = ""

    def line(self) -> str:
        v = f" = {self.value[:60]!r}" if self.value else ""
        return f"[{self.id}] {self.role} '{self.name[:80]}'{v}"


class _Worker:
    """All UI Automation work happens on ONE thread: the COM objects it hands out belong to that thread.
    If an app hangs a call, the worker is abandoned and a fresh one starts on the next request."""

    def __init__(self):
        import queue
        self.jobs: "queue.Queue" = queue.Queue()
        self.generation = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        import uiautomation as auto
        with auto.UIAutomationInitializerInThread():
            while True:
                fn, box, done = self.jobs.get()
                if fn is None:
                    return
                try:
                    box["value"] = fn()
                except Exception as e:  # noqa: BLE001  reported to the caller
                    box["error"] = e
                done.set()


_worker: _Worker | None = None
_worker_lock = threading.Lock()


def _run_with_timeout(fn, timeout: float):
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = _Worker()
        w = _worker
    box: dict = {}
    done = threading.Event()
    w.jobs.put((fn, box, done))
    if not done.wait(timeout):
        with _worker_lock:
            if _worker is w:
                _worker = None  # stuck on a frozen app: leave it, start fresh next time
        raise UIAError("The app is not responding to accessibility requests (timed out). "
                       "Elements from before are no longer valid; observe again.")
    if "error" in box:
        e = box["error"]
        raise e if isinstance(e, UIAError) else UIAError(f"{type(e).__name__}: {e}")
    return box.get("value")


class Observer:
    def __init__(self):
        self.elements: dict[int, Element] = {}
        self.window_title = ""
        self.window_handle = 0

    # --- observe ------------------------------------------------------------------------------------
    def observe(self, hwnd: int, title: str) -> tuple[list[Element], bool]:
        """Returns (elements, is_skeleton)."""
        def walk():
            import uiautomation as auto
            root = auto.ControlFromHandle(hwnd)
            if root is None:
                raise UIAError("That window has no accessibility tree.")
            found: list[Element] = []
            total_named = 0
            r = root.BoundingRectangle
            big = (r.right - r.left) * (r.bottom - r.top) > 500 * 400  # a dialog is allowed to be sparse
            for ctrl, depth in auto.WalkControl(root, includeTop=False, maxDepth=MAX_DEPTH):
                role = ctrl.ControlTypeName
                name = (ctrl.Name or "").strip()
                if name:
                    total_named += 1
                if role not in INTERACTIVE and not (role in TEXTUAL and name):
                    continue
                try:
                    if ctrl.IsOffscreen:
                        continue
                    r = ctrl.BoundingRectangle
                    rect = (r.left, r.top, r.right, r.bottom)
                except Exception:  # noqa: BLE001  elements can vanish mid-walk
                    continue
                if rect[2] - rect[0] <= 1 or rect[3] - rect[1] <= 1:
                    continue
                value = ""
                if role in ("EditControl", "ComboBoxControl"):
                    try:
                        value = ctrl.GetValuePattern().Value or ""
                    except Exception:  # noqa: BLE001
                        value = ""
                if role not in INTERACTIVE and not name:
                    continue
                found.append(Element(len(found) + 1, role.removesuffix("Control"), name, rect, ctrl, value))
                if len(found) >= MAX_ELEMENTS:
                    break
            return found, big and total_named < SKELETON_BELOW
        elements, skeleton = _run_with_timeout(walk, CALL_TIMEOUT_S * 3)
        self.elements = {e.id: e for e in elements}
        self.window_title, self.window_handle = title, hwnd
        return elements, skeleton

    def snapshot(self) -> set[str]:
        return {f"{e.role} '{e.name}'{'=' + e.value if e.value else ''}" for e in self.elements.values()}

    # --- act ----------------------------------------------------------------------------------------
    def element(self, element_id: int) -> Element:
        el = self.elements.get(element_id)
        if el is None:
            raise UIAError(f"No element [{element_id}] in the last observation. Call observe_window again.")
        return el

    def click(self, element_id: int) -> str:
        el = self.element(element_id)

        def do():
            c = el.control
            for getter, action, how in (("GetInvokePattern", "Invoke", "pressed"),
                                        ("GetTogglePattern", "Toggle", "toggled"),
                                        ("GetSelectionItemPattern", "Select", "selected")):
                try:
                    pattern = getattr(c, getter)()
                    if pattern:
                        getattr(pattern, action)()
                        return how
                except Exception:  # noqa: BLE001  try the next way
                    continue
            c.Click(simulateMove=False)
            return "clicked"
        return f"{_run_with_timeout(do, CALL_TIMEOUT_S * 2)} {el.role} '{el.name}'"

    def type_text(self, text: str, element_id: int | None) -> str:
        def do():
            import uiautomation as auto
            if element_id:
                el = self.element(element_id)
                try:
                    el.control.GetValuePattern().SetValue(text)
                    return f"set {el.role} '{el.name}' to the text"
                except Exception:  # noqa: BLE001  no value pattern: focus it and type
                    el.control.SetFocus()
            auto.SendKeys(escape_keys(text), interval=0.01, waitTime=0.1)
            return "typed the text"
        return _run_with_timeout(do, CALL_TIMEOUT_S + len(text) * 0.03)


def escape_keys(text: str) -> str:
    """uiautomation.SendKeys treats {} as key names; make plain text type literally."""
    return "".join("{" + ch + "}" if ch in "{}" else ch for ch in text)


KEY_NAMES = {"ctrl": "Ctrl", "control": "Ctrl", "alt": "Alt", "shift": "Shift", "win": "Win",
             "enter": "Enter", "return": "Enter", "esc": "Esc", "escape": "Esc", "tab": "Tab",
             "space": "Space", "backspace": "Back", "delete": "Delete", "del": "Delete", "up": "Up",
             "down": "Down", "left": "Left", "right": "Right", "home": "Home", "end": "End",
             "pageup": "PageUp", "pagedown": "PageDown"}


def combo_to_sendkeys(combo: str) -> str:
    """'ctrl+s' -> '{Ctrl}s', 'alt+f4' -> '{Alt}{F4}', 'enter' -> '{Enter}'."""
    out = []
    for part in combo.lower().replace(" ", "").split("+"):
        if not part:
            continue
        if part in KEY_NAMES:
            out.append("{" + KEY_NAMES[part] + "}")
        elif part.startswith("f") and part[1:].isdigit():
            out.append("{" + part.upper() + "}")
        elif len(part) == 1:
            out.append(part)
        else:
            raise UIAError(f"Unknown key '{part}'")
    return "".join(out)


def diff(before: set[str], after: set[str], limit: int = 8) -> str:
    gone, new = sorted(before - after), sorted(after - before)
    if not gone and not new:
        return "Nothing visible changed in that window."
    parts = []
    if new:
        parts.append("appeared: " + "; ".join(new[:limit]) + (" ..." if len(new) > limit else ""))
    if gone:
        parts.append("gone: " + "; ".join(gone[:limit]) + (" ..." if len(gone) > limit else ""))
    return " | ".join(parts)
