"""Small Win32 helpers (windows, keys, volume) used by tools and by the task-suite checks."""
import ctypes
import ctypes.wintypes as wt
from dataclasses import dataclass

user32 = ctypes.windll.user32
WM_CLOSE = 0x0010
KEYEVENTF_KEYUP = 0x0002
VK_MEDIA_PLAY_PAUSE = 0xB3


@dataclass(frozen=True)
class Window:
    hwnd: int
    title: str
    pid: int


def list_windows() -> list[Window]:
    """Visible top-level windows that have a title."""
    found: list[Window] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                pid = wt.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                found.append(Window(int(hwnd), buf.value, pid.value))
        return True

    user32.EnumWindows(cb, 0)
    return found


def make_dpi_aware() -> None:
    """Without this, a 125%-scaled screen reports shrunken coordinates and screenshots get cropped.
    Qt already does it for the desktop app; calling it twice is harmless."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor
    except (OSError, AttributeError):
        pass


def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    r = wt.RECT()
    # The "extended frame" is the visible window without its invisible resize border.
    if ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(r), ctypes.sizeof(r)) != 0:
        user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right, r.bottom


def _cloaked(hwnd: int) -> bool:
    val = ctypes.c_int(0)
    ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(val), ctypes.sizeof(val))
    return bool(val.value)


def _class_name(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


GWL_EXSTYLE = -20
WS_EX_TOPMOST, WS_EX_TOOLWINDOW = 0x8, 0x80


def _usable(w: Window, exclude_pids) -> bool:
    if w.pid in exclude_pids or not user32.IsWindowVisible(w.hwnd) or user32.IsIconic(w.hwnd) or _cloaked(w.hwnd):
        return False
    if _class_name(w.hwnd) in ("Progman", "Shell_TrayWnd", "WorkerW"):
        return False
    l, t, r, b = window_rect(w.hwnd)
    return r - l > 120 and b - t > 80


def _overlay(hwnd: int) -> bool:
    """Always-on-top widgets and tool windows (meeting overlays, status pop-outs) float above the window
    you're working in without being it."""
    ex = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    return bool(ex & (WS_EX_TOPMOST | WS_EX_TOOLWINDOW))


def _window(hwnd: int) -> Window:
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    pid = wt.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return Window(int(hwnd), buf.value, pid.value)


def front_window(exclude_pids: set[int] = frozenset()) -> Window | None:
    """The window the user is working in. The foreground window if it isn't ours; otherwise the
    front-most real window (EnumWindows goes front to back), preferring normal windows over overlays."""
    fg = user32.GetForegroundWindow()
    if fg:
        w = _window(fg)
        if w.title and _usable(w, exclude_pids):
            return w
    candidates = [w for w in list_windows() if _usable(w, exclude_pids)]
    normal = [w for w in candidates if not _overlay(w.hwnd)]
    return (normal or candidates or [None])[0]


def windows_matching(*needles: str) -> list[Window]:
    """Windows whose title contains every needle (case-insensitive)."""
    return [w for w in list_windows() if all(n.lower() in w.title.lower() for n in needles)]


def close_window(hwnd: int) -> None:
    """Asks the window to close, the same as clicking its X. The app can still show a 'save?' prompt."""
    user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)


def press_key(vk: int) -> None:
    user32.keybd_event(vk, 0, 0, 0)
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)


def get_volume() -> float:
    from pycaw.pycaw import AudioUtilities
    return float(AudioUtilities.GetSpeakers().EndpointVolume.GetMasterVolumeLevelScalar())


def playing_audio(exe: str) -> bool:
    """True while a process with this exe name has an *active* audio session (sound is playing)."""
    from pycaw.pycaw import AudioUtilities
    for s in AudioUtilities.GetAllSessions():
        if s.Process and s.Process.name().lower() == exe.lower() and s.State == 1:
            return True
    return False


def set_volume(level: float) -> None:
    from pycaw.pycaw import AudioUtilities
    AudioUtilities.GetSpeakers().EndpointVolume.SetMasterVolumeLevelScalar(max(0.0, min(1.0, level)), None)
