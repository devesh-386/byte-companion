from pathlib import Path

import numpy as np
import pytest

from companion.ui.state import GameState, level_for, level_progress
from companion.voice import Recorder, speakable

ROOT = Path(__file__).resolve().parents[1]


def test_speakable_strips_code_links_and_markdown():
    text = "Try this:\n```python\nprint(1)\n```\nDocs at https://python.org are **great**."
    out = speakable(text)
    assert "print" not in out and "https" not in out and "*" not in out
    assert "code in the chat" in out and "the link" in out


def test_speakable_truncates_at_a_sentence():
    text = "This is a sentence that goes on. " * 40
    out = speakable(text, max_chars=200)
    assert len(out) < 260 and out.endswith("The rest is in the chat.")


def test_recorder_auto_stops_after_speech_then_silence():
    fired = []
    rec = Recorder(on_auto_stop=lambda: fired.append(True), silence_s=0.0, threshold=0.1)
    rec._started = 1e18  # never hit max length
    loud = np.full(1600, 0.5, dtype=np.float32)
    quiet = np.zeros(1600, dtype=np.float32)
    rec._callback(quiet)   # silence before speaking must not stop it
    rec._callback(loud)
    rec._callback(quiet)
    rec._callback(quiet)
    import time
    time.sleep(0.1)
    assert fired == [True]
    assert rec.stop().shape == (6400,)


def test_levels():
    assert level_for(0) == 1 and level_for(49) == 1 and level_for(50) == 2 and level_for(200) == 3
    assert level_progress(0) == 0.0 and 0 < level_progress(100) < 1


def test_game_state_persists(tmp_path):
    s = GameState(tmp_path / "state.json")
    assert s.add_xp(60) is True  # crossed into level 2
    s.speak = False
    s.save()
    again = GameState(tmp_path / "state.json")
    assert again.xp == 60 and again.level == 2 and again.speak is False


@pytest.mark.skipif(not (ROOT / "models" / "piper" / "en_US-lessac-medium.onnx").exists()
                    or not (ROOT / "models" / "whisper").exists(), reason="voice models not downloaded")
def test_voice_round_trip():
    from companion.voice import Ears, Voice
    voice = Voice(ROOT / "models" / "piper" / "en_US-lessac-medium.onnx")
    audio, rate = voice.synthesize("Open my downloads folder please.")
    f = audio.astype(np.float32) / 32768
    x16 = np.interp(np.arange(0, len(f), rate / 16000), np.arange(len(f)), f).astype(np.float32)
    heard = Ears("base.en", ROOT / "models" / "whisper").transcribe(x16).lower()
    assert "downloads" in heard and "folder" in heard
