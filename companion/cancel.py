"""Stopping Byte mid-action.

One CancelToken per agent run. Anything slow (the model stream, a tool, speech) either checks
`token.cancelled` between steps or registers a callback that interrupts it (for example, closing the HTTP socket
so a blocked read returns at once). The stop hotkey cancels every active token through STOP.
"""
import threading
from typing import Callable


class Cancelled(Exception):
    pass


class CancelToken:
    def __init__(self):
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[], None]] = []
        self.reason = ""

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self, reason: str = "stopped") -> None:
        with self._lock:
            if self._event.is_set():
                return
            self.reason = reason
            self._event.set()
            callbacks, self._callbacks = self._callbacks, []
        for cb in callbacks:
            try:
                cb()
            except Exception:
                pass  # an interrupt that fails must not stop the others from running

    def check(self) -> None:
        if self._event.is_set():
            raise Cancelled(self.reason)

    def on_cancel(self, cb: Callable[[], None]) -> Callable[[], None]:
        """Run cb when cancelled (immediately if already). Returns a function that unregisters it."""
        with self._lock:
            if not self._event.is_set():
                self._callbacks.append(cb)
                return lambda: self._remove(cb)
        cb()
        return lambda: None

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)

    def _remove(self, cb) -> None:
        with self._lock:
            if cb in self._callbacks:
                self._callbacks.remove(cb)


class StopController:
    """Tracks the tokens of everything running right now so one key press can stop them all."""

    def __init__(self):
        self._lock = threading.Lock()
        self._active: set[CancelToken] = set()
        self._listeners: list[Callable[[], None]] = []

    def new_token(self) -> CancelToken:
        token = CancelToken()
        with self._lock:
            self._active.add(token)
        return token

    def release(self, token: CancelToken) -> None:
        with self._lock:
            self._active.discard(token)

    def add_listener(self, cb: Callable[[], None]) -> None:
        """Extra things to stop that aren't agent runs (speech, microphone)."""
        self._listeners.append(cb)

    def stop_all(self, reason: str = "stopped by hotkey") -> int:
        with self._lock:
            tokens = list(self._active)
        for t in tokens:
            t.cancel(reason)
        for cb in list(self._listeners):
            try:
                cb()
            except Exception:
                pass
        return len(tokens)


STOP = StopController()
