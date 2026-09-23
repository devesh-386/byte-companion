"""The chat panel's parts: message bubbles, tool chips, the typing indicator, the Allow/Deny card and the
header. Plain Qt widgets styled with one palette (THEME), so the whole panel can be re-skinned in one place.

    ┌ Header ─────────────────────────────── [status pill] [🧠][🔊][＋][–] ┐
    │ ChatView (scrolls)                                                  │
    │   Byte ▸ bubble (markdown) + tool chips              you ◂ bubble   │
    │ ConfirmCard (only while Byte asks)                                  │
    └ input row ──────────────────────────────────────────── [mic] ──────┘
"""
import html
import json
import time

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QSizePolicy, QVBoxLayout, QWidget)

THEME = {
    "panel": "rgba(12, 18, 28, 242)", "panel_border": "rgba(58, 214, 232, 110)",
    "byte_bg": "#16222f", "user_bg": "#1e6f86", "text": "#e6f7ff", "muted": "#8aa0b3",
    "accent": "#3ad6e8", "accent_dark": "#1e98b0", "warn": "#ffd84a", "danger": "#ff6b6b", "ok": "#5ee08a",
    "chip_bg": "rgba(58, 214, 232, 28)",
}
MAX_MESSAGES = 60  # older bubbles are dropped so a long session stays fast

PANEL_CSS = f"""
QFrame#panel {{ background: {THEME['panel']}; border: 1px solid {THEME['panel_border']}; border-radius: 16px; }}
QLabel {{ color: {THEME['text']}; font-family: 'Segoe UI'; font-size: 10.5pt; }}
QLabel#title {{ font-family: Consolas; font-weight: 700; font-size: 11pt; color: {THEME['text']}; }}
QLabel#level {{ font-family: Consolas; font-size: 8.5pt; color: {THEME['accent']}; }}
QLabel#pill {{ border-radius: 9px; padding: 1px 9px; font-size: 8.5pt; font-weight: 600; }}
QLabel#who {{ color: {THEME['muted']}; font-size: 8pt; }}
QLabel#chip {{ background: {THEME['chip_bg']}; color: {THEME['muted']}; border-radius: 8px; padding: 2px 8px;
              font-size: 8.5pt; }}
QFrame#byteBubble {{ background: {THEME['byte_bg']}; border-radius: 12px; border-top-left-radius: 4px; }}
QFrame#userBubble {{ background: {THEME['user_bg']}; border-radius: 12px; border-top-right-radius: 4px; }}
QFrame#systemBubble {{ background: transparent; border: 1px dashed rgba(138, 160, 179, 90); border-radius: 10px; }}
QFrame#card {{ background: #2a2413; border: 1px solid {THEME['warn']}; border-radius: 12px; }}
QLabel#cardTitle {{ color: {THEME['warn']}; font-weight: 700; }}
QLabel#code {{ font-family: Consolas; font-size: 9pt; color: #f3e9c6; background: rgba(0,0,0,70);
              border-radius: 6px; padding: 6px; }}
QPushButton#iconBtn {{ background: transparent; border: none; color: {THEME['muted']}; font-size: 11pt;
                      padding: 2px 5px; border-radius: 6px; min-width: 22px; }}
QPushButton#iconBtn:hover {{ background: rgba(58, 214, 232, 40); color: {THEME['text']}; }}
QPushButton#iconBtn:checked {{ color: {THEME['accent']}; }}
QPushButton#allow {{ background: {THEME['ok']}; color: #0b1b12; border: none; border-radius: 8px; padding: 5px 14px;
                    font-weight: 700; }}
QPushButton#allow:hover {{ background: #86f0ab; }}
QPushButton#deny {{ background: #3a4552; color: white; border: none; border-radius: 8px; padding: 5px 14px;
                   font-weight: 700; }}
QPushButton#deny:hover {{ background: {THEME['danger']}; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 6px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: rgba(138, 160, 179, 80); border-radius: 3px; min-height: 24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""

STATUS = {  # pill text, background, foreground
    "booting": ("booting", "#3a4552", "#e6f7ff"), "ready": ("online", "#12402b", "#5ee08a"),
    "thinking": ("thinking…", "#16384a", "#3ad6e8"), "listening": ("listening", "#4a1616", "#ff8f8f"),
    "speaking": ("speaking", "#2b1f4a", "#c9a7ff"), "waiting": ("needs you", "#4a3b12", "#ffd84a"),
    "offline": ("offline", "#4a1616", "#ff8f8f"),
}
TOOL_ICONS = {
    "web_search": "🔍", "fetch_webpage": "🌐", "list_directory": "📁", "read_text_file": "📄",
    "write_text_file": "✏️", "calculate": "🧮", "get_current_time": "🕒", "open_folder": "📂", "open_file": "📂",
    "launch_app": "🚀", "system_info": "💻", "set_reminder": "⏰", "remember": "🧠", "recall": "🧠",
    "forget": "🧹", "memory_overview": "🧠", "look_at_screen": "👀", "observe_window": "🪟", "click": "🖱️",
    "type_text": "⌨️", "press_keys": "⌨️", "close_window": "✖️", "focus_window": "🪟", "set_volume": "🔊",
    "media_control": "⏯️", "open_in_vscode": "🧑‍💻", "play_spotify": "🎵", "find_folder": "🔎",
    "list_open_windows": "🪟", "read_own_code": "🧬", "search_own_code": "🧬", "edit_own_code": "🧬",
    "write_plugin": "🧩", "undo_self_edit": "↩️", "restart_byte": "🔄",
}


class Pill(QLabel):
    def __init__(self):
        super().__init__(objectName="pill")
        self.set("booting")

    def set(self, key: str) -> None:
        text, bg, fg = STATUS.get(key, STATUS["ready"])
        self.key = key
        self.setText(f"● {text}")
        self.setStyleSheet(f"background: {bg}; color: {fg};")


class TypingDots(QWidget):
    """Three dots bouncing in turn, shown in Byte's bubble until the first word arrives."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(38, 16)
        self._t0 = time.monotonic()
        self._timer = QTimer(self, interval=60, timeout=self.update)
        self._timer.start()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        t = time.monotonic() - self._t0
        for i in range(3):
            phase = (t * 4 - i * 0.6) % 3.0
            lift = 4 * max(0.0, 1 - abs(phase - 0.5) * 2)
            p.setBrush(QColor(THEME["accent"]))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QRectF(4 + i * 11, 6 - lift, 7, 7))


class Message(QWidget):
    """One chat bubble. Byte's can grow while streaming and carries tool chips above the text."""

    def __init__(self, who: str, markdown: str = "", width: int = 300):
        super().__init__()
        self.who = who
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.bubble = QFrame(objectName={"byte": "byteBubble", "user": "userBubble"}.get(who, "systemBubble"))
        self.bubble.setMaximumWidth(width)
        inner = QVBoxLayout(self.bubble)
        inner.setContentsMargins(11, 8, 11, 8)
        inner.setSpacing(5)
        self.chips = QVBoxLayout()
        self.chips.setSpacing(3)
        inner.addLayout(self.chips)
        self.dots = TypingDots() if who == "byte" and not markdown else None
        if self.dots:
            inner.addWidget(self.dots)
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.MarkdownText)
        self.label.setOpenExternalLinks(True)
        self.label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        if who == "system":
            self.label.setStyleSheet(f"color: {THEME['muted']};")
        inner.addWidget(self.label)
        self.label.setVisible(bool(markdown))
        self.set_text(markdown)
        self._chip_labels: dict[int, QLabel] = {}
        if who == "user":
            outer.addStretch(1)
            outer.addWidget(self.bubble)
        else:
            outer.addWidget(self.bubble)
            outer.addStretch(1)

    def set_text(self, markdown: str) -> None:
        if markdown.strip() and self.dots:
            self.dots.hide()
        self.label.setVisible(bool(markdown.strip()))
        self.label.setText(markdown)

    def set_chips(self, lines: list[str]) -> None:
        """Tool activity lines -> small chips (already-shown chips are reused, so nothing flickers)."""
        for i, line in enumerate(lines):
            chip = self._chip_labels.get(i)
            if chip is None:
                chip = QLabel(objectName="chip")
                chip.setTextFormat(Qt.RichText)
                chip.setWordWrap(True)
                self.chips.addWidget(chip)
                self._chip_labels[i] = chip
            if chip.text() != line:
                chip.setText(line)
        for i in list(self._chip_labels):
            if i >= len(lines):
                self._chip_labels.pop(i).deleteLater()

    def done(self) -> None:
        if self.dots:
            self.dots.hide()


class ChatView(QScrollArea):
    def __init__(self, bubble_width: int = 300):
        super().__init__()
        self.bubble_width = bubble_width
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        host = QWidget(objectName="chatHost")
        # Selector-scoped: an unscoped "background: transparent" would also wipe every bubble inside.
        host.setStyleSheet("QWidget#chatHost { background: transparent; }")
        self.box = QVBoxLayout(host)
        self.box.setContentsMargins(2, 2, 6, 2)
        self.box.setSpacing(8)
        self.box.addStretch(1)
        self.setWidget(host)
        self.messages: list[Message] = []
        self.verticalScrollBar().rangeChanged.connect(lambda lo, hi: self.verticalScrollBar().setValue(hi))

    def add(self, who: str, markdown: str = "") -> Message:
        m = Message(who, markdown, self.bubble_width)
        self.box.addWidget(m)
        self.messages.append(m)
        while len(self.messages) > MAX_MESSAGES:
            self.messages.pop(0).deleteLater()
        fade_in(m)
        return m

    def clear(self) -> None:
        for m in self.messages:
            m.deleteLater()
        self.messages.clear()

    @property
    def last(self) -> Message | None:
        return self.messages[-1] if self.messages else None


class ConfirmCard(QFrame):
    """'Byte wants to …' with the tool, a readable view of its arguments, and Allow / Deny."""
    answered = Signal(bool)

    def __init__(self):
        super().__init__(objectName="card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        self.title = QLabel(objectName="cardTitle")
        self.body = QLabel(objectName="code")
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.PlainText)
        self.body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row = QHBoxLayout()
        hint = QLabel("Y = allow · N = deny", objectName="who")
        self.deny = QPushButton("Deny", objectName="deny")
        self.allow = QPushButton("Allow", objectName="allow")
        row.addWidget(hint)
        row.addStretch(1)
        row.addWidget(self.deny)
        row.addWidget(self.allow)
        lay.addWidget(self.title)
        lay.addWidget(self.body)
        lay.addLayout(row)
        self.deny.clicked.connect(lambda: self.answered.emit(False))
        self.allow.clicked.connect(lambda: self.answered.emit(True))
        self.hide()

    def ask(self, name: str, arguments: dict, phrase: str) -> None:
        icon = TOOL_ICONS.get(name, "⚙️")
        self.title.setText(f"{icon}  Byte needs your OK: {phrase}")
        self.body.setText(format_arguments(name, arguments))
        self.show()
        fade_in(self)
        self.allow.setFocus()


def format_arguments(name: str, arguments: dict, limit: int = 900) -> str:
    """Show what will happen in a form a person can check: one line per argument, long text kept whole."""
    if not arguments:
        return f"{name}()"
    lines = [f"{name}:"]
    for k, v in arguments.items():
        text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        if "\n" in text or len(text) > 60:
            lines.append(f"  {k}:\n" + "\n".join("    " + l for l in text.splitlines()[:25]))
        else:
            lines.append(f"  {k}: {text}")
    out = "\n".join(lines)
    return out if len(out) <= limit else out[:limit] + "\n  …"


def tool_chip(name: str, phrase: str, state: str = "running", detail: str = "") -> str:
    icon = TOOL_ICONS.get(name, "⚙️")
    mark = {"running": "…", "ok": f"<span style='color:{THEME['ok']}'>✓</span>",
            "error": f"<span style='color:{THEME['danger']}'>✗</span>"}[state]
    extra = f" <span style='color:{THEME['danger']}'>{html.escape(detail[:90])}</span>" if detail else ""
    return f"{icon} {html.escape(phrase)} {mark}{extra}"


def fade_in(widget: QWidget, ms: int = 180) -> None:
    effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)
    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setDuration(ms)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(QEasingCurve.OutCubic)
    # Drop the effect afterwards: a leftover opacity effect makes text render soft on Windows.
    anim.finished.connect(lambda: widget.setGraphicsEffect(None))
    anim.start()
