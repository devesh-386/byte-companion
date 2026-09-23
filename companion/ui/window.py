"""The desktop companion: a frameless, always-on-top, transparent window in the corner of the screen."""
import html
import random
import threading
import time

from PySide6.QtCore import QObject, QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QCursor, QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel, QLayout,
                               QLineEdit, QMenu, QPushButton, QSystemTrayIcon, QVBoxLayout, QWidget)

from .. import config
from .chat import PANEL_CSS, ChatView, ConfirmCard, Message, Pill, fade_in, tool_chip
from .sprite import MOOD_COLORS, Sprite
from .state import XP_PER_MEMORY, XP_PER_MESSAGE, XP_PER_TOOL, GameState, level_progress

TOOL_PHRASES = {
    "web_search": "searching the web", "fetch_webpage": "reading a web page",
    "list_directory": "looking in a folder", "read_text_file": "reading a file",
    "calculate": "crunching numbers", "get_current_time": "checking the clock",
    "open_folder": "opening a folder", "open_file": "opening a file", "launch_app": "launching an app",
    "system_info": "checking your laptop", "set_reminder": "setting a reminder",
    "write_text_file": "writing a file", "remember": "saving to memory", "recall": "searching memory",
    "forget": "forgetting something", "memory_overview": "going through my memories",
    "look_at_screen": "looking at your screen", "observe_window": "reading the window",
    "click": "clicking", "type_text": "typing", "press_keys": "pressing keys", "close_window": "closing a window",
    "focus_window": "switching windows", "set_volume": "adjusting the volume", "media_control": "pressing play/pause",
    "open_in_vscode": "opening VS Code", "play_spotify": "starting Spotify", "find_folder": "looking for a folder",
    "list_open_windows": "checking open windows",
    "read_own_code": "reading my own code", "search_own_code": "searching my own code",
    "edit_own_code": "editing myself and running my tests", "write_plugin": "writing myself a new plugin",
    "undo_self_edit": "undoing my last change", "restart_byte": "restarting myself",
}
GREETINGS = ["Byte online. Ready when you are, {u}.", "Hey {u}! Systems green, brain loaded.",
             "Back in the corner and ready for quests, {u}.", "Boot complete. What are we doing today, {u}?"]

INPUT_CSS = """
QLineEdit { background: rgba(255, 255, 255, 14); border: 1px solid rgba(58, 214, 232, 90); border-radius: 12px;
            color: #e6f7ff; padding: 7px 12px; font-family: 'Segoe UI'; font-size: 10.5pt;
            selection-background-color: #1e98b0; }
QLineEdit:focus { border-color: #3ad6e8; background: rgba(255, 255, 255, 20); }
QLineEdit:disabled { color: #6f8494; }
QPushButton#mic, QPushButton#send { background: rgba(255, 255, 255, 14); border: 1px solid rgba(58, 214, 232, 90);
                  border-radius: 12px; color: #3ad6e8; font-size: 12pt; min-width: 36px; max-width: 36px;
                  min-height: 34px; max-height: 34px; }
QPushButton#mic:hover, QPushButton#send:hover { background: #1e98b0; color: white; }
QPushButton#mic[recording="true"] { background: #ff4d4d; border-color: #ff8f8f; color: white; }
QPushButton#mic:disabled, QPushButton#send:disabled { color: #3a4552; border-color: rgba(58, 214, 232, 40); }
"""
PANEL_W, CHAT_H = 404, 360


class Bridge(QObject):
    """Worker threads talk to the UI only through these signals (Qt queues them onto the UI thread)."""
    event = Signal(str, object)
    booted = Signal(object)
    boot_failed = Signal(str)
    progress = Signal(str)
    confirm_request = Signal(object)   # (tool name, arguments, one-line summary)
    notify = Signal(str)
    memory_saved = Signal(object)
    voice_ready = Signal(object, object)   # (Voice | None, Ears | None)
    mic_auto_stop = Signal()
    transcript = Signal(str)
    speaking = Signal(bool)
    stop_pressed = Signal()
    interrupted = Signal(str)   # the user cut in while Byte was talking or thinking
    restart_requested = Signal()  # Byte changed its own code and asked to be restarted


class SpriteView(QWidget):
    clicked = Signal()

    def __init__(self, sprite: Sprite, window: "CompanionWindow"):
        super().__init__()
        self.sprite = sprite
        self.win = window
        w, h = sprite.size
        self.setFixedSize(w, h)
        self.setCursor(Qt.PointingHandCursor)
        self._press: QPoint | None = None
        self._dragged = False

    def paintEvent(self, _):
        p = QPainter(self)
        self.sprite.draw(p, 0, 0)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press = e.globalPosition().toPoint()
            self._win_start = self.win.pos()
            self._dragged = False

    def mouseMoveEvent(self, e):
        if self._press is None:
            return
        delta = e.globalPosition().toPoint() - self._press
        if delta.manhattanLength() > 4:
            self._dragged = True
        if self._dragged:
            self.win.move(self._win_start + delta)

    def mouseReleaseEvent(self, e):
        if self._press is not None and not self._dragged:
            self.clicked.emit()
        elif self._dragged:
            self.win.remember_anchor()
        self._press = None

    def contextMenuEvent(self, e):
        # Right-click the character: the same menu as the tray icon, including Quit.
        tray = getattr(self.win, "tray", None)
        if tray is not None and tray.contextMenu() is not None:
            tray.contextMenu().exec(e.globalPos())


class Hud(QWidget):
    """Name, level and XP bar, drawn like a game health bar."""

    def __init__(self, state: GameState):
        super().__init__()
        self.state = state
        self.mood = "neutral"
        self.flash_text = ""
        self.flash_until = 0.0
        self.setFixedSize(150, 46)

    def flash(self, text: str, seconds: float = 2.5) -> None:
        self.flash_text, self.flash_until = text, time.time() + seconds
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(QRectF(0, 4, 150, 38), QColor(14, 22, 34, 220))
        p.setPen(QPen(QColor(58, 214, 232, 150), 2))
        p.drawRect(QRectF(1, 5, 148, 36))
        font = QFont("Consolas", 10)
        font.setBold(True)
        p.setFont(font)
        p.setPen(QColor("#e6f7ff"))
        p.drawText(8, 20, config.NAME.upper())
        p.setPen(MOOD_COLORS.get(self.mood, MOOD_COLORS["neutral"]))
        p.drawText(92, 20, f"LV {self.state.level}")
        # XP bar in pixel blocks.
        blocks, filled = 16, int(level_progress(self.state.xp) * 16)
        for i in range(blocks):
            color = QColor("#3ad6e8") if i < filled else QColor(58, 214, 232, 45)
            p.fillRect(QRectF(8 + i * 8.4, 27, 7, 7), color)
        if self.flash_text and time.time() < self.flash_until:
            f2 = QFont("Consolas", 8)
            p.setFont(f2)
            p.setPen(QColor("#ffd84a"))
            p.drawText(QRectF(0, 0, 150, 12), Qt.AlignRight, self.flash_text)


class CompanionWindow(QWidget):
    def __init__(self, bridge: Bridge):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet(PANEL_CSS + INPUT_CSS)
        self.bridge = bridge
        self.companion = None
        self.state = GameState(config.DATA_HOME / "state.json")
        self.sprite = Sprite(scale=5)
        self.busy = False
        self.reply_md = ""
        self.lines: list[str] = []
        self.mood_until = 0.0
        self._confirm_event: threading.Event | None = None
        self._confirm_ok = False
        self._last_tick = time.monotonic()
        self.voice = None
        self.ears = None
        self.recorder = None
        self.speaking = False
        # Conversation mode: one click on the mic starts it, the next click ends it. In between Byte
        # listens -> answers -> speaks -> listens again. While it speaks, only a cut-in listener runs, so it
        # doesn't mistake its own voice for yours (see bargein.py for how cutting in works).
        self.convo = False
        self.barge = None             # BargeIn listener, active while Byte talks or thinks in a conversation
        self._carry = ""              # words said while cutting in ("no, open Spotify"), sent with the next message
        self._cut_in = False          # the current answer was interrupted: don't read it out

        self._tools: list[list] = []     # [name, phrase, state, detail] for this reply's tool chips
        self._current: Message | None = None
        self._status_msg: Message | None = None

        # --- header -------------------------------------------------------------------------------
        header = QHBoxLayout()
        header.setContentsMargins(4, 0, 0, 0)
        header.setSpacing(6)
        title = QLabel(config.NAME.upper(), objectName="title")
        self.level_label = QLabel(objectName="level")
        self.pill = Pill()
        header.addWidget(title)
        header.addWidget(self.level_label)
        header.addStretch(1)
        header.addWidget(self.pill)

        def icon_btn(text: str, tip: str, checkable: bool = False) -> QPushButton:
            b = QPushButton(text, objectName="iconBtn")
            b.setToolTip(tip)
            b.setCheckable(checkable)
            b.setCursor(Qt.PointingHandCursor)
            header.addWidget(b)
            return b
        self.mem_btn = icon_btn("🧠", f"What {config.NAME} remembers")
        self.speak_btn = icon_btn("🔊", "Speak replies out loud", checkable=True)
        self.speak_btn.setChecked(self.state.speak)
        self.new_btn = icon_btn("＋", "New chat (long-term memory stays)")
        self.hide_btn = icon_btn("–", "Hide (Esc)")

        # --- chat + Allow/Deny card ---------------------------------------------------------------
        self.chat = ChatView(bubble_width=PANEL_W - 70)
        self.chat.setFixedHeight(CHAT_H)
        self.card = ConfirmCard()

        # --- input --------------------------------------------------------------------------------
        self.input = QLineEdit()
        self.input.setPlaceholderText(f"Message {config.NAME}…")
        self.input.setToolTip("Enter to send · Esc to hide")
        self.input.setEnabled(False)
        self.send_btn = QPushButton("➤", objectName="send")
        self.send_btn.setToolTip("Send")
        self.send_btn.setEnabled(False)

        # --- sprite + hud -------------------------------------------------------------------------
        self.view = SpriteView(self.sprite, self)
        self.hud = Hud(self.state)
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.addStretch(1)
        bottom.addWidget(self.hud, 0, Qt.AlignBottom)
        bottom.addWidget(self.view, 0, Qt.AlignBottom)

        self.mic_btn = QPushButton("●", objectName="mic")
        self.mic_btn.setToolTip("Start a conversation (click again to end it)")
        self.mic_btn.setEnabled(False)
        input_row = QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(6)
        input_row.addWidget(self.input, 1)
        input_row.addWidget(self.send_btn)
        input_row.addWidget(self.mic_btn)

        self.panel = QFrame(objectName="panel")
        pl = QVBoxLayout(self.panel)
        pl.setContentsMargins(12, 10, 12, 12)
        pl.setSpacing(8)
        pl.addLayout(header)
        pl.addWidget(self.chat)
        pl.addWidget(self.card)
        pl.addLayout(input_row)
        shadow = QGraphicsDropShadowEffect(self.panel, blurRadius=28, offset=QPointF(0, 6))
        shadow.setColor(QColor(0, 0, 0, 150))
        self.panel.setGraphicsEffect(shadow)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)
        root.addWidget(self.panel)
        root.addLayout(bottom)
        # The window is always exactly as big as its contents, so hiding the chat shrinks it at once.
        root.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        root.setContentsMargins(14, 10, 14, 6)  # room for the panel's shadow
        self.panel.setFixedWidth(PANEL_W)

        # --- wiring -------------------------------------------------------------------------------
        self.view.clicked.connect(self.toggle_panel)
        self.input.returnPressed.connect(self.send)
        self.input.textChanged.connect(lambda t: self.send_btn.setEnabled(bool(t.strip()) and not self.busy))
        self.send_btn.clicked.connect(self.send)
        self.card.answered.connect(self._answer_confirm)
        self.mem_btn.clicked.connect(self.show_memories)
        self.speak_btn.toggled.connect(self.set_speak)
        self.new_btn.clicked.connect(self.new_chat)
        self.hide_btn.clicked.connect(self.hide_panel)
        bridge.event.connect(self.on_event)
        bridge.booted.connect(self.on_booted)
        bridge.boot_failed.connect(self.on_boot_failed)
        bridge.progress.connect(self.on_progress)
        bridge.confirm_request.connect(self.on_confirm_request)
        bridge.notify.connect(self.on_notify)
        bridge.memory_saved.connect(self.on_memory_saved)
        bridge.voice_ready.connect(self.on_voice_ready)
        bridge.mic_auto_stop.connect(self.stop_recording)
        bridge.transcript.connect(self.on_transcript)
        bridge.speaking.connect(self.on_speaking)
        bridge.interrupted.connect(self.on_interrupted)
        bridge.stop_pressed.connect(self.on_stop_pressed)
        self.mic_btn.clicked.connect(self.toggle_recording)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

        self.sprite.state.thinking = True
        self._place_initially()
        self.on_progress("Booting up")

    # --- placement --------------------------------------------------------------------------------
    def _place_initially(self) -> None:
        self.adjustSize()
        geo = QApplication.primaryScreen().availableGeometry()
        if self.state.pos:
            self.anchor = QPoint(*self.state.pos)
            if not geo.adjusted(-50, -50, 50, 50).contains(self.anchor):
                self.anchor = None
        else:
            self.anchor = None
        if self.anchor is None:
            self.anchor = QPoint(geo.right() - 12, geo.bottom() - 8)
        self._reanchor()

    def _reanchor(self) -> None:
        """Keep the bottom-right corner (where Byte stands) fixed while the bubble grows and shrinks."""
        self.layout().activate()  # layouts update lazily; measure the new size, not the old one
        self.adjustSize()
        self.move(self.anchor.x() - self.width(), self.anchor.y() - self.height())

    def resizeEvent(self, e):
        # Window resizes can land after the layout change that caused them; re-pin every time.
        super().resizeEvent(e)
        if getattr(self, "anchor", None) is not None:
            self.move(self.anchor.x() - self.width(), self.anchor.y() - self.height())

    def remember_anchor(self) -> None:
        self.anchor = QPoint(self.x() + self.width(), self.y() + self.height())
        self.state.pos = [self.anchor.x(), self.anchor.y()]
        self.state.save()

    # --- bubble content ---------------------------------------------------------------------------
    def set_bubble(self, markdown: str, lines: list[str] | None = None) -> None:
        """A new message from Byte (greetings, notices, the memory view)."""
        self._current = self.chat.add("byte", markdown)
        self.reply_md = markdown
        self.lines = lines or []
        self._tools = []
        self._render()

    def system(self, markdown: str) -> None:
        """A quiet line in the chat (not something Byte said)."""
        self.chat.add("system", markdown)

    def _render(self) -> None:
        """Push the current reply (streaming text + tool chips) into Byte's newest bubble."""
        if self._current is None:
            self._current = self.chat.add("byte")
        chips = [tool_chip(*t) for t in self._tools] + list(self.lines)
        self._current.set_chips(chips)
        self._current.set_text(self.reply_md)

    def on_progress(self, msg: str) -> None:
        if self._status_msg is None:
            self._status_msg = self.chat.add("system")
        self._status_msg.set_text(f"*{msg}…*")

    def toggle_panel(self) -> None:
        if self.speaking and self.voice:
            self.voice.stop()  # a click while Byte talks means "shush", not "hide"
            return
        if self.panel.isVisible():
            self.hide_panel()
        else:
            self.show_panel()
            if self.input.isEnabled():
                self.input.setFocus()
                self.activateWindow()

    def show_panel(self) -> None:
        if not self.panel.isVisible():
            self.panel.show()
            self._reanchor()
            fade_in(self.chat, 160)

    def hide_panel(self) -> None:
        self.panel.hide()
        self._reanchor()

    def new_chat(self) -> None:
        if self.busy:
            return
        if self.companion:
            self.companion.agent.reset()
        self.chat.clear()
        self._current = None
        self.system("*Fresh chat. My long-term memory is still here.*")

    def set_speak(self, on: bool) -> None:
        self.state.speak = on
        self.state.save()
        self.speak_btn.setText("🔊" if on else "🔇")
        if not on and self.voice:
            self.voice.stop()
        if hasattr(self, "tray_speak"):
            self.tray_speak.setChecked(on)

    def keyPressEvent(self, e):
        if not self.card.isHidden() and e.key() in (Qt.Key_Y, Qt.Key_N):
            self._answer_confirm(e.key() == Qt.Key_Y)
        elif e.key() == Qt.Key_Escape:
            self.hide_panel()
        else:
            super().keyPressEvent(e)

    # --- animation --------------------------------------------------------------------------------
    def tick(self) -> None:
        now = time.monotonic()
        self.sprite.tick(now - self._last_tick)
        self._last_tick = now
        st = self.sprite.state
        if not self.busy and st.mood != "neutral" and time.time() > self.mood_until:
            self.set_mood("neutral")
        st.listening = self.recorder is not None or (self.input.hasFocus() and bool(self.input.text())
                                                     and not self.busy)
        if self.speaking:
            st.talking = True
        # Eyes follow the cursor.
        c = self.view.mapToGlobal(self.view.rect().center())
        d = QCursor.pos() - c
        st.look = (max(-1, min(1, round(d.x() / 250))), max(-1, min(1, round(d.y() / 250))))
        self.view.update()
        self.hud.update()
        self._update_status()

    def _update_status(self) -> None:
        if self.companion is None:
            key = "offline" if self.sprite.state.mood == "fear" else "booting"
        elif not self.card.isHidden():
            key = "waiting"
        elif self.recorder is not None:
            key = "listening"
        elif self.speaking:
            key = "speaking"
        elif self.busy:
            key = "thinking"
        else:
            key = "ready"
        if key != self.pill.key:
            self.pill.set(key)
        lv = f"LV {self.state.level}"
        if self.level_label.text() != lv:
            self.level_label.setText(lv)
            self.level_label.setToolTip(f"{self.state.xp} XP")

    def set_mood(self, mood: str, hold: float = 0.0) -> None:
        self.sprite.state.mood = mood
        self.hud.mood = mood
        if hold:
            self.mood_until = time.time() + hold
        if mood in ("joy", "surprise"):
            self.sprite.hop(3 if mood == "joy" else 2)

    # --- boot -------------------------------------------------------------------------------------
    def on_booted(self, companion) -> None:
        self.companion = companion
        self.sprite.state.thinking = False
        self.input.setEnabled(True)
        if self._status_msg is not None:
            self._status_msg.set_text("*" + " · ".join(html.escape(n) for n in companion.notes) + "*"
                                      if companion.notes else "*Ready.*")
            self._status_msg = None
        self.set_bubble(random.choice(GREETINGS).format(u=config.USER_NAME))
        self.set_mood("joy", hold=3)
        self.input.setFocus()

    def on_boot_failed(self, message: str) -> None:
        self.sprite.state.thinking = False
        self.set_mood("fear", hold=3600)
        self.set_bubble(f"**I couldn't start.**\n\n```\n{message[-900:]}\n```")

    # --- chat -------------------------------------------------------------------------------------
    def send(self) -> None:
        text = self.input.text().strip()
        if not text or self.busy or self.companion is None:
            return
        if text in ("/reset", "/new"):
            self.input.clear()
            self.new_chat()
            return
        self.input.clear()
        if self.voice:
            self.voice.stop()
        self.busy = True
        self.input.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.mic_btn.setEnabled(self.convo)  # during a conversation the button stays live, to end it
        self.chat.add("user", html.escape(text))
        self.set_bubble("")  # Byte's reply bubble, with typing dots until the first word
        self.sprite.state.thinking = True
        self._cut_in = False
        self._arm_barge("")  # "no wait" while Byte is still thinking works too
        threading.Thread(target=self._run_agent, args=(text,), daemon=True).start()

    def _run_agent(self, text: str) -> None:
        try:
            self.companion.agent.run(text, on_event=lambda e: self.bridge.event.emit(e.kind, e.data))
        except Exception as e:  # the UI must survive anything the model or a tool throws
            self.bridge.event.emit("tool_error", {"name": "?", "error": f"{type(e).__name__}: {e}"})
            self.bridge.event.emit("done", {"answer": ""})

    def on_event(self, kind: str, data: dict) -> None:
        st = self.sprite.state
        if kind == "emotion":
            if data["label"] != "neutral":
                self.set_mood(data["label"], hold=6)
        elif kind == "memory":
            if data["recalled"]:
                self.hud.flash(f"recalled {len(data['recalled'])}")
        elif kind == "thinking":
            st.thinking, st.talking = True, False
        elif kind == "text":
            st.thinking, st.talking = False, True
            self.reply_md += data["text"]
            self._render()
        elif kind == "tool_call":
            st.thinking, st.talking = True, False
            self._tools.append([data["name"], TOOL_PHRASES.get(data["name"], data["name"]), "running", ""])
            self._render()
        elif kind == "tool_result":
            self._finish_chip(data["name"], "ok")
            if self.state.add_xp(XP_PER_TOOL):
                self.level_up()
        elif kind == "tool_error":
            if not self._finish_chip(data["name"], "error", data["error"]):
                self.lines.append(f"<span style='color:#ff8f8f'>{html.escape(data['error'][:120])}</span>")
            self._render()
        elif kind == "step_limit":
            self.lines.append("⚠️ I got stuck going in circles and stopped.")
            self._render()
        elif kind == "stopped":
            self.lines.append("<span style='color:#ff8f8f'>■ stopped</span>")
            self._render()
        elif kind == "done":
            st.thinking = st.talking = False
            self.busy = False
            self.input.setEnabled(True)
            if self._current is not None:
                self._current.done()
            self.mic_btn.setEnabled(self.ears is not None)
            self.input.setFocus()
            if not self.reply_md.strip() and data.get("answer"):
                self.reply_md = data["answer"]
                self._render()
            if self._cut_in:  # you interrupted: don't read out the half-finished answer
                self._cut_in = False
                self._disarm_barge()
                self._listen_again()
            elif not self.say(data.get("answer") or self.reply_md):
                self._disarm_barge()
                self._listen_again()  # nothing to say out loud, so go straight back to listening
            self.mood_until = max(self.mood_until, time.time() + 4)
            if self.state.add_xp(XP_PER_MESSAGE):
                self.level_up()

    def _finish_chip(self, name: str, state: str, detail: str = "") -> bool:
        for t in reversed(self._tools):
            if t[0] == name and t[2] == "running":
                t[2], t[3] = state, detail
                self._render()
                return True
        return False

    def level_up(self) -> None:
        self.set_mood("joy", hold=4)
        self.hud.flash(f"LEVEL UP! LV {self.state.level}", 4)
        self.system(f"⭐ **LEVEL UP!** {config.NAME} reached LV {self.state.level}.")

    def on_memory_saved(self, facts) -> None:
        self.hud.flash("+ memory")
        if self.state.add_xp(XP_PER_MEMORY * len(facts)):
            self.level_up()

    def show_memories(self) -> None:
        """The memory dashboard: every fact, grouped, with the date a dated one runs out."""
        store = getattr(self.companion, "store", None)
        self.show_panel()
        if store is None:
            self.set_bubble("*Memory is off, so I don't remember anything between chats.*")
            return
        from ..memory.structure import LABELS
        groups = store.overview()
        if not groups:
            self.set_bubble("*I don't remember anything about you yet. Tell me things and I'll keep them.*")
            return
        md = [f"**What I remember about you** ({sum(len(v) for v in groups.values())} things)\n"]
        for cat, mems in groups.items():
            md.append(f"\n**{LABELS.get(cat, cat)}**\n")
            for m in mems:
                until = time.strftime(" *(until %a %d %b)*", time.localtime(m.expires_at)) if m.expires_at else ""
                md.append(f"- {m.text.removeprefix('The user').strip().capitalize()}{until}")
        md.append("\n*Say \"forget ...\" to remove one.*")
        self.set_bubble("\n".join(md))

    # --- confirmation (called from the agent's thread) --------------------------------------------
    def ask_confirmation(self, call, summary: str) -> bool:
        ev = threading.Event()
        self._confirm_event = ev
        self.bridge.confirm_request.emit((call.name, call.arguments, summary))
        ev.wait()
        return self._confirm_ok

    def on_confirm_request(self, request) -> None:
        name, arguments, _summary = request
        self.show_panel()
        self.set_mood("surprise", hold=2)
        self.card.ask(name, arguments, TOOL_PHRASES.get(name, name))
        self.setFocus()  # so Y / N work straight away
        self.activateWindow()

    def _answer_confirm(self, ok: bool) -> None:
        if self.card.isHidden() and self._confirm_event is None:
            return
        self._confirm_ok = ok
        self.card.hide()
        if self._tools:
            self._tools[-1][1] += " (allowed)" if ok else " (denied)"
        self._render()
        if self._confirm_event:
            self._confirm_event.set()
            self._confirm_event = None

    # --- emergency stop (Ctrl+Alt+Esc) -------------------------------------------------------------
    def on_stop_pressed(self) -> None:
        from ..cancel import STOP
        stopped = STOP.stop_all("stopped by hotkey")
        self.end_conversation(quiet=True)
        if self.voice:
            self.voice.stop()
        if self._confirm_event:  # a pending Allow/Deny counts as Deny
            self._answer_confirm(False)
        self.show_panel()
        self.set_mood("surprise", hold=2)
        self.hud.flash("STOPPED", 2)
        if not stopped and not self.busy:
            self.system("*Stopped. (Nothing was running.)*")

    # --- voice ------------------------------------------------------------------------------------
    def on_voice_ready(self, voice, ears) -> None:
        self.voice, self.ears = voice, ears
        if ears is not None:
            from .. import bargein
            self.barge = bargein.BargeIn(getattr(ears, "transcribe_strict", ears.transcribe))
        self.mic_btn.setEnabled(ears is not None and not self.busy)
        if ears is None:
            self.mic_btn.setToolTip("Voice input unavailable")

    def say(self, text: str) -> bool:
        """Speak on a worker thread. Returns False when there is nothing to speak."""
        if not (self.voice and self.state.speak and text and text.strip()):
            return False
        self._arm_barge(text)  # knows what Byte is saying, so it can tell its echo from you

        def run() -> None:
            try:
                self.voice.speak(text, on_start=lambda: self.bridge.speaking.emit(True))
            finally:
                # Always, even if speech was cut short or empty, so a conversation never gets stuck.
                self.bridge.speaking.emit(False)
        threading.Thread(target=run, daemon=True).start()
        return True

    def on_speaking(self, on: bool) -> None:
        self.speaking = on
        if not on:
            self.sprite.state.talking = False
            if not self.busy:
                self._disarm_barge()
            self._listen_again()

    # --- cutting in -------------------------------------------------------------------------------
    def _arm_barge(self, spoken: str) -> None:
        if not (self.convo and self.barge):
            return
        if self.barge._alive.is_set():
            self.barge.set_spoken(spoken)
            return
        try:
            self.barge.start(spoken, self.bridge.interrupted.emit)
        except Exception:  # no mic: cutting in just doesn't work, everything else does
            pass

    def _disarm_barge(self) -> None:
        if self.barge:
            self.barge.stop()

    def on_interrupted(self, heard: str) -> None:
        if not self.convo:
            return
        from ..bargein import leftover_message
        self._carry = leftover_message(heard)
        if self.busy and self._confirm_event is None:  # still thinking: stop that too
            from ..cancel import STOP
            self._cut_in = True
            STOP.stop_all("interrupted by the user")
        if self.voice:
            self.voice.stop()
        self.sprite.state.talking = False
        self.system(f"✋ *you cut in: \"{html.escape(heard.strip()[:80])}\"*")
        self._listen_again()

    def _listen_again(self) -> None:
        # A short pause lets the speaker's last echo die out before the mic opens.
        if self.convo:
            QTimer.singleShot(350, self._resume_listening)

    def _resume_listening(self) -> None:
        if self.convo and not self.busy and not self.speaking and self.recorder is None:
            self.start_recording()

    def toggle_recording(self) -> None:
        if self.convo:
            self.end_conversation()
        else:
            self.convo = True
            self._set_mic_look(True)
            self.start_recording()
            if self.recorder is None:  # no microphone
                self.end_conversation(quiet=True)

    def end_conversation(self, quiet: bool = False) -> None:
        was_on = self.convo
        self.convo = False
        self._carry = ""
        self._disarm_barge()
        if self.recorder is not None:
            self.recorder.stop()  # what was said since the last pause is dropped
            self.recorder = None
        if self.voice:
            self.voice.stop()
        self._set_mic_look(False)
        self.mic_btn.setEnabled(self.ears is not None and not self.busy)
        if was_on and not quiet and not self.busy:
            self.sprite.state.thinking = False
            self.system("*Conversation ended. Click the mic to talk again.*")

    def _set_mic_look(self, on: bool) -> None:
        self.mic_btn.setProperty("recording", on)
        self.mic_btn.style().polish(self.mic_btn)
        self.mic_btn.setToolTip("End the conversation" if on else "Start a conversation (click again to end it)")

    def start_recording(self) -> None:
        if self.ears is None or self.busy:
            return
        from ..voice import Recorder
        if self.voice:
            self.voice.stop()
        self._disarm_barge()  # one listener at a time
        self.recorder = Recorder(on_auto_stop=self.bridge.mic_auto_stop.emit)
        try:
            self.recorder.start()
        except Exception as e:
            self.recorder = None
            self.system(f"*I can't reach a microphone:* `{type(e).__name__}: {e}`")
            return
        self.show_panel()  # the status pill says "listening"; the red mic ends the conversation

    def stop_recording(self) -> None:
        if self.recorder is None:
            return
        audio = self.recorder.stop()
        self.recorder = None
        self.sprite.state.thinking = True
        threading.Thread(target=lambda: self.bridge.transcript.emit(self.ears.transcribe(audio)),
                         daemon=True).start()

    def on_transcript(self, text: str) -> None:
        self.sprite.state.thinking = False
        if not self.convo:
            return  # the conversation was ended while this was being transcribed
        text = f"{self._carry} {text}".strip()  # "no, open" + "Spotify instead"
        self._carry = ""
        if not text:
            self._listen_again()  # silence or a cough: keep listening, don't nag
            return
        self.input.setText(text)
        self.send()

    # --- reminders --------------------------------------------------------------------------------
    def on_notify(self, message: str) -> None:
        self.show_panel()
        self.set_mood("surprise", hold=8)
        self.set_bubble(f"**{message}**")
        QApplication.beep()
        self.say(message)
        if hasattr(self, "tray"):
            self.tray.showMessage(config.NAME, message, self.tray.icon(), 8000)


def sprite_icon(size: int = 64) -> QIcon:
    sprite = Sprite(scale=3)
    w, h = sprite.size
    pm = QPixmap(w, h)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    sprite.draw(p, 0, 0)
    p.end()
    return QIcon(pm.scaled(size, size, Qt.KeepAspectRatio, Qt.FastTransformation))


def build_tray(app: QApplication, win: CompanionWindow, on_quit) -> QSystemTrayIcon:
    tray = QSystemTrayIcon(sprite_icon(), app)
    tray.setToolTip(f"{config.NAME} - local AI companion")
    menu = QMenu()
    show = QAction("Show / hide chat", menu)
    show.triggered.connect(win.toggle_panel)
    new = QAction("New chat", menu)
    new.triggered.connect(win.new_chat)
    speak = QAction("Speak replies out loud", menu, checkable=True)
    speak.setChecked(win.state.speak)
    speak.toggled.connect(lambda on: win.speak_btn.setChecked(on))  # one switch, shown in two places
    win.tray_speak = speak
    memories = QAction(f"What {config.NAME} remembers...", menu)
    memories.triggered.connect(win.show_memories)
    quit_ = QAction(f"Quit {config.NAME}", menu)
    quit_.triggered.connect(on_quit)
    for a in (show, new, memories, speak, quit_):
        menu.addAction(a)
    tray.setContextMenu(menu)
    tray.activated.connect(lambda reason: win.toggle_panel() if reason == QSystemTrayIcon.Trigger else None)
    tray.show()
    win.tray = tray
    return tray
