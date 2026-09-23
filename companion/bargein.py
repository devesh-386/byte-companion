"""Barge-in: let the user cut Byte off mid-sentence ("wait", "no, I meant...").

    while Byte speaks or thinks:
    mic ─► loud bit? ─► collect until a short pause (max 3 s) ─► whisper ─► is_interruption()?
                                                                              │ yes
                                                              stop speaking ◄─┘ + hand over the words

The mic also hears Byte's own voice through the speakers (echo). So a heard phrase only counts when it
has words Byte is NOT saying right now: an interrupt word ("wait", "stop", "no"...) or 3+ new words.
Headphones make this perfect; on laptop speakers, a quiet room and a moderate volume help.
"""
import queue
import re
import threading
import time
from typing import Callable

import numpy as np

from .voice import SAMPLE_RATE

# Only unmistakable words. Whisper hallucinates short phrases from near-silence ("Fair enough." in a live
# test with echo cancellation on), so soft words like "enough", "hey" or "actually" caused false cut-ins.
INTERRUPT_WORDS = {"wait", "stop", "no", "nope", "hold", "pause", "shush", "quiet", "hang", "listen", "cancel"}
# Saying only one of these means "be quiet and listen", not a message to answer.
PURE_WAIT = {"wait", "stop", "hold on", "hang on", "pause", "shush", "quiet", "enough", "no", "nope",
             "hey", "hey byte", "byte", "wait wait", "stop stop", "listen", "cancel", "okay wait", "ok wait"}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def is_interruption(heard: str, spoken: str) -> bool:
    heard_words = words(heard)
    if not heard_words:
        return False
    echo = set(words(spoken))
    novel = [w for w in heard_words if w not in echo]
    if any(w in INTERRUPT_WORDS for w in novel):
        return True
    # Whisper mishears some of Byte's own echo ("on Friday" -> "on my eye"), so a few odd words inside
    # a mostly-echo phrase don't count; the new words must be most of what was heard.
    return len(novel) >= 3 and len(novel) / len(heard_words) >= 0.6


def leftover_message(heard: str) -> str:
    """What to keep as the start of the user's next message ("no, open Spotify" -> keep it all;
    "wait" alone -> nothing)."""
    clean = " ".join(words(heard))
    return "" if not clean or clean in PURE_WAIT else heard.strip()


class BargeIn:
    """Runs a mic listener on its own threads. start() while Byte talks/thinks, stop() when it's done."""

    def __init__(self, transcribe: Callable[[np.ndarray], str], threshold: float = 0.03,
                 pause_s: float = 0.45, max_s: float = 3.0):
        self.transcribe = transcribe
        self.threshold = threshold      # higher than normal listening: only clear speech counts
        self.pause_s = pause_s
        self.max_s = max_s
        self._stream = None
        self._segments: queue.Queue = queue.Queue()
        self._alive = threading.Event()
        self._spoken = ""
        self._on_interrupt: Callable[[str], None] | None = None

    def set_spoken(self, text: str) -> None:
        self._spoken = text

    def start(self, spoken: str, on_interrupt: Callable[[str], None]) -> None:
        from .voice import MicStream
        self.stop()
        self._spoken, self._on_interrupt = spoken, on_interrupt
        self._alive.set()
        self._buf: list[np.ndarray] = []
        self._loud = False
        self._quiet_since: float | None = None
        self._seg_start = 0.0
        # Windows' echo cancellation removes most of Byte's own voice (measured: loud blocks 65 -> 5 of
        # ~120 during a 12 s reply), so what's left above the threshold is mostly the user.
        self._stream = MicStream(self._callback)
        self._stream.start()
        threading.Thread(target=self._worker, daemon=True).start()

    @property
    def using_aec(self) -> bool:
        return bool(self._stream and self._stream.using_aec)

    def _callback(self, chunk: np.ndarray) -> None:
        level = float(np.sqrt(np.mean(chunk ** 2)))
        now = time.monotonic()
        if level > self.threshold:
            if not self._loud:
                self._loud, self._seg_start, self._buf = True, now, []
            self._quiet_since = None
        if self._loud:
            self._buf.append(chunk)
            if level <= self.threshold and self._quiet_since is None:
                self._quiet_since = now
            if (self._quiet_since and now - self._quiet_since >= self.pause_s) or now - self._seg_start >= self.max_s:
                self._segments.put(np.concatenate(self._buf))
                self._loud, self._buf, self._quiet_since = False, [], None

    def _worker(self) -> None:
        while self._alive.is_set():
            try:
                audio = self._segments.get(timeout=0.2)
            except queue.Empty:
                continue
            if len(audio) < SAMPLE_RATE * 0.25:  # a click or a bump, not a word
                continue
            heard = self.transcribe(audio)
            if self._alive.is_set() and is_interruption(heard, self._spoken):
                cb, self._on_interrupt = self._on_interrupt, None
                self.stop()
                if cb:
                    cb(heard)
                return

    def stop(self) -> None:
        self._alive.clear()
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream = None
        while not self._segments.empty():
            self._segments.get_nowait()
