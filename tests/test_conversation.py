"""Conversation mode: one mic click starts it, Byte keeps listening after each reply, the next click ends it.
Runs the real window offscreen with a fake mic, fake speech and a fake agent."""
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

from companion import config, voice  # noqa: E402


def pump(pred, timeout=5.0):
    app = QApplication.instance()
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if pred():
            return True
        time.sleep(0.01)
    return pred()


class FakeRecorder:
    started = 0

    def __init__(self, on_auto_stop=None, **kw):
        self.on_auto_stop = on_auto_stop

    def start(self):
        FakeRecorder.started += 1

    def stop(self):
        return np.ones(16000, dtype=np.float32)


class FakeEars:
    def __init__(self):
        self.said = ["what time is it", "", "thanks"]  # "" = the user stayed quiet

    def transcribe(self, audio):
        return self.said.pop(0) if self.said else ""


class FakeVoice:
    def __init__(self):
        self.spoken, self.stops = [], 0

    def speak(self, text, on_start=lambda: None, on_end=lambda: None):
        on_start()
        self.spoken.append(text)

    def stop(self):
        self.stops += 1


class FakeAgent:
    def __init__(self):
        self.heard = []

    def run(self, text, on_event=None, cancel=None):
        self.heard.append(text)
        on_event(type("E", (), {"kind": "done", "data": {"answer": f"reply to {text}"}})())
        return f"reply to {text}"


class FakeBarge:
    def __init__(self, transcribe):
        import threading
        self._alive = threading.Event()
        self.spoken = None

    def start(self, spoken, on_interrupt):
        self._alive.set()
        self.spoken = spoken

    def set_spoken(self, text):
        self.spoken = text

    def stop(self):
        self._alive.clear()


@pytest.fixture
def win(tmp_path, monkeypatch):
    from companion import bargein
    monkeypatch.setattr(config, "DATA_HOME", tmp_path)   # never touch the real XP / state file
    monkeypatch.setattr(voice, "Recorder", FakeRecorder)
    monkeypatch.setattr(bargein, "BargeIn", FakeBarge)   # never open the real mic in tests
    FakeRecorder.started = 0
    QApplication.instance() or QApplication([])
    from companion.ui.window import Bridge, CompanionWindow
    bridge = Bridge()
    w = CompanionWindow(bridge)
    w.on_booted(type("C", (), {"agent": FakeAgent(), "notes": []})())
    w.on_voice_ready(FakeVoice(), FakeEars())
    w.state.speak = True
    yield w
    w.end_conversation(quiet=True)
    w.close()


def test_one_click_keeps_the_conversation_going_until_the_next_click(win):
    win.mic_btn.click()
    assert win.convo and win.recorder is not None and FakeRecorder.started == 1

    win.stop_recording()                          # the user paused: "what time is it"
    assert pump(lambda: win.voice.spoken == ["reply to what time is it"])
    assert pump(lambda: win.recorder is not None and FakeRecorder.started == 2)  # listening again, no click

    win.stop_recording()                          # silence: no reply, just keeps listening
    assert pump(lambda: FakeRecorder.started == 3)
    assert win.companion.agent.heard == ["what time is it"]

    win.stop_recording()                          # "thanks"
    assert pump(lambda: len(win.voice.spoken) == 2 and FakeRecorder.started == 4)

    win.mic_btn.click()                           # second click ends it
    assert not win.convo and win.recorder is None
    pump(lambda: False, timeout=0.6)              # nothing restarts on its own afterwards
    assert FakeRecorder.started == 4


def test_ending_mid_transcription_does_not_send(win):
    win.mic_btn.click()
    win.recorder.stop()
    audio_owner = win.recorder
    win.recorder = None
    win.end_conversation()
    win.on_transcript("open spotify")             # the transcript arrives after the click
    pump(lambda: False, timeout=0.5)
    assert win.companion.agent.heard == [] and audio_owner is not None


def test_mic_button_stays_clickable_while_byte_thinks(win):
    win.mic_btn.click()
    win.busy = True
    win.input.setText("hi")
    win.busy = False
    win.companion.agent.run = lambda *a, **k: None  # an agent that never finishes
    win.send()
    assert win.busy and win.mic_btn.isEnabled()
    win.mic_btn.click()
    assert not win.convo


# --- cutting in ---------------------------------------------------------------------------------------
from companion.bargein import is_interruption, leftover_message  # noqa: E402

SPOKEN = ("Sure, I opened Spotify and started your liked songs playlist for you. "
          "Here is a longer answer so we can test the echo. Your exam is on Friday, so maybe study.")


@pytest.mark.parametrize("heard,cut", [
    ("wait", True),
    ("no no, not that one", True),
    ("hold on", True),
    ("open YouTube instead please", True),                 # 3+ words Byte isn't saying
    ("opened Spotify and started your liked songs", False),  # its own voice through the speakers
    ("playlist for you", False),
    ("Your exam is on my eye. So fast.", False),             # real mis-heard echo from the live test
    ("Here is a longer answer so we can test the echo. I open-", False),
    ("", False),
    ("um", False),
])
def test_is_interruption_tells_you_from_the_echo(heard, cut):
    assert is_interruption(heard, SPOKEN) is cut


def test_leftover_message_keeps_real_requests_only():
    assert leftover_message("Wait.") == ""
    assert leftover_message("hold on") == ""
    assert leftover_message("No, open YouTube instead") == "No, open YouTube instead"


def test_cutting_in_while_byte_speaks_stops_it_and_listens(win):
    win.mic_btn.click()
    win.voice.speak = lambda text, on_start=lambda: None, on_end=lambda: None: None  # "still talking"
    win.stop_recording()                                   # user: "what time is it"
    assert pump(lambda: win.barge.spoken == "reply to what time is it")
    stops = win.voice.stops
    win.bridge.interrupted.emit("wait")
    assert pump(lambda: win.voice.stops > stops)           # Byte went quiet
    assert pump(lambda: win.recorder is not None)          # and is listening to you
    assert win._carry == ""                                # "wait" alone isn't a message


def test_what_you_say_when_cutting_in_is_kept(win):
    win.mic_btn.click()
    win.recorder = None
    win.bridge.interrupted.emit("no, open YouTube")
    assert pump(lambda: win.recorder is not None)
    win.ears.said = ["instead"]
    win.stop_recording()
    assert pump(lambda: win.companion.agent.heard == ["no, open YouTube instead"])


def test_cutting_in_while_thinking_cancels_and_stays_quiet(win):
    from companion.agent import STOPPED_ANSWER
    from companion.cancel import STOP
    win.mic_btn.click()
    stopped = []
    orig = STOP.stop_all
    STOP.stop_all = lambda reason="": stopped.append(reason) or 1
    try:
        win.companion.agent.run = lambda *a, **k: None     # still thinking
        win.input.setText("tell me a long story")
        win.recorder = None
        win.send()
        win.bridge.interrupted.emit("stop")
        assert pump(lambda: stopped == ["interrupted by the user"])
        spoken_before = list(win.voice.spoken)
        win.on_event("done", {"answer": STOPPED_ANSWER})   # what the agent says after a cancel
        assert pump(lambda: win.recorder is not None)
        assert win.voice.spoken == spoken_before           # it did not read "(Stopped.)" out loud
    finally:
        STOP.stop_all = orig


def test_confirm_card_keys_and_chips(win):
    import threading
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent
    call = type("Call", (), {"name": "write_text_file", "arguments": {"path": "todo.txt", "content": "x"}})()
    win.set_bubble("")
    win.on_event("tool_call", {"name": "write_text_file", "arguments": {}})
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("ok", win.ask_confirmation(call, "summary")), daemon=True)
    t.start()
    assert pump(lambda: not win.card.isHidden())
    assert "write_text_file" in win.card.body.text() and "todo.txt" in win.card.body.text()
    win.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_N, Qt.NoModifier))
    t.join(2)
    assert result["ok"] is False and win.card.isHidden()
    assert "(denied)" in win._tools[-1][1]
    win.on_event("tool_error", {"name": "write_text_file", "error": "The user declined"})
    assert win._tools[-1][2] == "error"
