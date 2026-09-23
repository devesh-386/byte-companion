"""Offline ears (faster-whisper) and voice (Piper). Both run on the CPU so the GPU stays with the LLM."""
import re
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

SAMPLE_RATE = 16000


def speakable(text: str, max_chars: int = 420) -> str:
    """What's worth saying out loud: no code, links or markdown, and not an essay."""
    text = re.sub(r"```.*?```", " I've put the code in the chat. ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"https?://\S+", "the link", text)
    text = re.sub(r"[*_#>|]+", "", text)
    text = re.sub(r"^\s*[-\d.]+\s+", "", text, flags=re.M)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return (cut[: end + 1] if end > 80 else cut) + " The rest is in the chat."


class MicStream:
    """The microphone as a stream of 0.1 s float32 blocks. Prefers Windows' echo-cancelled mic (aec.py),
    so Byte doesn't hear itself through the speakers; falls back to the plain mic if that can't start."""

    def __init__(self, on_block: Callable[[np.ndarray], None], echo_cancel: bool = True):
        self.on_block = on_block
        self.echo_cancel = echo_cancel
        self.using_aec = False
        self._aec = None
        self._stream = None

    def start(self) -> None:
        if self.echo_cancel:
            try:
                from .aec import EchoCancelledMic
                self._aec = EchoCancelledMic(self.on_block)
                self._aec.start()
                self.using_aec = True
                return
            except Exception:  # noqa: BLE001  no AEC on this machine: plain mic below
                self._aec = None
        import sounddevice as sd
        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=1600,
                                      callback=lambda indata, frames, t, status: self.on_block(indata[:, 0].copy()))
        self._stream.start()

    def stop(self) -> None:
        if self._aec is not None:
            self._aec.stop()
            self._aec = None
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class Recorder:
    """Records the microphone until stop() is called or the speaker goes quiet."""

    def __init__(self, on_auto_stop: Callable[[], None] | None = None, silence_s: float = 1.3,
                 threshold: float = 0.012, max_s: float = 30.0, echo_cancel: bool = True):
        self.on_auto_stop = on_auto_stop
        self.echo_cancel = echo_cancel
        self.silence_s = silence_s
        self.threshold = threshold
        self.max_s = max_s
        self._chunks: list[np.ndarray] = []
        self._stream = None
        self._heard_speech = False
        self._quiet_since: float | None = None
        self._started = 0.0
        self.level = 0.0

    def start(self) -> None:
        self._chunks, self._heard_speech, self._quiet_since = [], False, None
        self._started = time.monotonic()
        self._stream = MicStream(self._callback, echo_cancel=self.echo_cancel)
        self._stream.start()

    def _callback(self, chunk: np.ndarray) -> None:
        self._chunks.append(chunk)
        self.level = float(np.sqrt(np.mean(chunk ** 2)))
        now = time.monotonic()
        if self.level > self.threshold:
            self._heard_speech, self._quiet_since = True, None
        elif self._heard_speech and self._quiet_since is None:
            self._quiet_since = now
        done = (self._quiet_since is not None and now - self._quiet_since >= self.silence_s) or \
               now - self._started >= self.max_s
        if done and self.on_auto_stop:
            cb, self.on_auto_stop = self.on_auto_stop, None  # fire once
            threading.Thread(target=cb, daemon=True).start()

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream = None
        return np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)


class Ears:
    def __init__(self, model: str, download_root: Path, hint: str = ""):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device="cpu", compute_type="int8", download_root=str(download_root))
        self.hint = hint
        self._lock = threading.Lock()  # the barge-in listener and normal listening share one model

    def transcribe(self, audio: np.ndarray, strict: bool = False) -> str:
        """strict: drop segments Whisper itself doubts (likely no speech, or a low-confidence guess).
        Used by the cut-in listener, where a hallucinated word would interrupt Byte for nothing."""
        if len(audio) < SAMPLE_RATE * 0.3:
            return ""
        with self._lock:
            return self._transcribe(audio, strict)

    def transcribe_strict(self, audio: np.ndarray) -> str:
        return self.transcribe(audio, strict=True)

    def _transcribe(self, audio: np.ndarray, strict: bool = False) -> str:
        # The hint nudges Whisper toward names it would otherwise mishear ("Byte", not "Bite").
        # No hint in strict mode: a prompt makes Whisper more willing to invent words from noise.
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1, vad_filter=True,
                                            initial_prompt=None if strict else (self.hint or None))
        kept = [s for s in segments if not strict or (s.no_speech_prob < 0.5 and s.avg_logprob > -0.9)]
        return " ".join(s.text.strip() for s in kept).strip()


class Voice:
    def __init__(self, voice_model: Path, speed: float = 1.0):
        from piper import PiperVoice, SynthesisConfig
        self.voice = PiperVoice.load(str(voice_model))
        self.syn_config = SynthesisConfig(length_scale=1.0 / speed)
        self._stop = threading.Event()
        self._lock = threading.Lock()  # one utterance at a time; a new one interrupts the old

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        chunks, rate = [], 22050
        for chunk in self.voice.synthesize(text, syn_config=self.syn_config):
            chunks.append(chunk.audio_int16_array)
            rate = chunk.sample_rate
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
        return audio, rate

    def speak(self, text: str, on_start: Callable[[], None] = lambda: None,
              on_end: Callable[[], None] = lambda: None) -> None:
        """Blocking; run it on a worker thread. stop() cuts it off."""
        import sounddevice as sd
        text = speakable(text)
        if not text:
            return
        self._stop.set()
        with self._lock:
            self._stop.clear()
            audio, rate = self.synthesize(text)
            if self._stop.is_set() or not len(audio):
                return
            on_start()
            try:
                sd.play(audio, rate)
                end = time.monotonic() + len(audio) / rate + 0.2
                while time.monotonic() < end and not self._stop.is_set():
                    time.sleep(0.05)
                sd.stop()
            finally:
                on_end()

    def stop(self) -> None:
        self._stop.set()
