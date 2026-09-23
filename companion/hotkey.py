"""A system-wide hotkey (default Ctrl+Alt+Esc) that calls a function from any app.

Windows delivers WM_HOTKEY to the thread that registered the key, so registration and the message loop
live on one dedicated thread. stop() posts WM_QUIT to that thread to end it.
"""
import ctypes
import ctypes.wintypes as wt
import threading
from typing import Callable

MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x4000
VK_ESCAPE, VK_F12 = 0x1B, 0x7B
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012


class HotkeyError(RuntimeError):
    pass


class GlobalHotkey:
    def __init__(self, callback: Callable[[], None], modifiers: int = MOD_CONTROL | MOD_ALT,
                 vk: int = VK_ESCAPE, hotkey_id: int = 0xB17E):
        self.callback = callback
        self.modifiers = modifiers | MOD_NOREPEAT
        self.vk = vk
        self.id = hotkey_id
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._ready = threading.Event()
        self._error: str | None = None

    def start(self, timeout: float = 3.0) -> None:
        """Raises HotkeyError if another program already owns this key combination."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="stop-hotkey")
        self._thread.start()
        self._ready.wait(timeout)
        if self._error:
            raise HotkeyError(self._error)

    def _run(self) -> None:
        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        self._thread_id = kernel32.GetCurrentThreadId()
        if not user32.RegisterHotKey(None, self.id, self.modifiers, self.vk):
            self._error = f"RegisterHotKey failed (error {kernel32.GetLastError()}); the key is probably taken"
            self._ready.set()
            return
        self._ready.set()
        msg = wt.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY and msg.wParam == self.id:
                    try:
                        self.callback()
                    except Exception:
                        pass  # a failing callback must not kill the hotkey
        finally:
            user32.UnregisterHotKey(None, self.id)

    def stop(self) -> None:
        if self._thread and self._thread.is_alive() and self._thread_id:
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            self._thread.join(2)
