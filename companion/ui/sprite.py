"""Byte, drawn pixel by pixel. Everything is procedural on a 20x22 grid, so every expression and
animation frame is just a different set of cells; nothing is loaded from image files."""
import math
import random
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QRadialGradient

GRID_W, GRID_H = 20, 22

OUTLINE = QColor("#0b1b2b")
BODY = QColor("#3ad6e8")
LIGHT = QColor("#a8f4fb")
SHADE = QColor("#1e98b0")
PUPIL = QColor("#0b1b2b")
GLINT = QColor("#ffffff")
BLUSH = QColor("#ff8fb1")
TEAR = QColor("#7fc8ff")
MOUTH = QColor("#0b1b2b")
TONGUE = QColor("#ff6f91")

MOOD_COLORS = {
    "neutral": QColor("#6dff9c"),
    "joy": QColor("#ffd84a"),
    "sadness": QColor("#5aa8ff"),
    "anger": QColor("#ff4d4d"),
    "fear": QColor("#b98cff"),
    "surprise": QColor("#ff9a3c"),
}


def _in_body(x: int, y: int) -> bool:
    rows = {6: (5, 14), 7: (4, 15), 16: (4, 15), 17: (5, 14)}
    if y in rows:
        lo, hi = rows[y]
        return lo <= x <= hi
    return 8 <= y <= 15 and 3 <= x <= 16


BODY_CELLS = [(x, y) for y in range(GRID_H) for x in range(GRID_W) if _in_body(x, y)]
OUTLINE_CELLS = {(x, y) for x, y in BODY_CELLS
                 if not all(_in_body(x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))}


@dataclass
class SpriteState:
    mood: str = "neutral"          # which face
    talking: bool = False
    thinking: bool = False
    listening: bool = False
    look: tuple[int, int] = (0, 0)  # pupil offset toward the cursor, each -1..1


class Sprite:
    """Holds animation clocks; draw() paints one frame."""

    def __init__(self, scale: int = 5):
        self.scale = scale
        self.state = SpriteState()
        self.t = 0.0
        self._next_blink = 2.5
        self._blink_until = -1.0
        self._hop_until = -1.0
        self._hop_height = 0

    @property
    def size(self) -> tuple[int, int]:
        return GRID_W * self.scale, GRID_H * self.scale

    def hop(self, height: int = 2, duration: float = 0.35) -> None:
        self._hop_until = self.t + duration
        self._hop_height = height

    def tick(self, dt: float) -> None:
        self.t += dt
        if self.t >= self._next_blink:
            self._blink_until = self.t + 0.14
            self._next_blink = self.t + random.uniform(2.5, 5.5)

    # --- drawing -----------------------------------------------------------------------------------
    def draw(self, p: QPainter, ox: float, oy: float) -> None:
        s, st = self.scale, self.state
        bob = 1 if math.sin(self.t * 2.2) > 0.6 else 0
        if st.thinking:
            bob = 0
        hop = 0
        if self.t < self._hop_until:
            phase = 1 - (self._hop_until - self.t) / 0.35
            hop = round(self._hop_height * math.sin(math.pi * min(1.0, phase)))
        jitter = random.choice((-1, 0, 1)) if st.mood == "fear" and random.random() < 0.35 else 0
        dy = -bob - hop

        def cell(x: int, y: int, color: QColor) -> None:
            p.fillRect(QRectF(ox + (x + jitter) * s, oy + (y + dy) * s, s, s), color)

        mood_color = MOOD_COLORS.get(st.mood, MOOD_COLORS["neutral"])

        # Soft glow behind the antenna tip, pulsing.
        glow_r = s * (3.2 + 0.6 * math.sin(self.t * 4))
        cx, cy = ox + (10 + jitter) * s, oy + (1.5 + dy) * s
        g = QRadialGradient(QPointF(cx, cy), glow_r)
        c = QColor(mood_color)
        c.setAlpha(150)
        g.setColorAt(0, c)
        c.setAlpha(0)
        g.setColorAt(1, c)
        p.setPen(Qt.NoPen)
        p.setBrush(g)
        p.drawEllipse(QPointF(cx, cy), glow_r, glow_r)

        # Ground shadow shrinks while hopping.
        shadow = QColor(0, 0, 0, 60)
        w = 10 - hop
        p.fillRect(QRectF(ox + (10 - w / 2) * s, oy + 19.3 * s, w * s, s * 0.8), shadow)

        # Antenna.
        for y in (3, 4, 5):
            cell(10, y, OUTLINE)
        for x in (9, 10, 11):
            cell(x, 1, mood_color)
            cell(x, 2, mood_color)
        cell(9, 1, mood_color.lighter(150))

        # Body: outline ring, fill, light top-left, shade bottom-right.
        for x, y in BODY_CELLS:
            if (x, y) in OUTLINE_CELLS:
                cell(x, y, OUTLINE)
            elif y >= 15 or x >= 15:
                cell(x, y, SHADE)
            else:
                cell(x, y, BODY)
        for x, y in ((5, 8), (6, 8), (5, 9), (7, 7), (6, 7)):
            cell(x, y, LIGHT)
        # Feet.
        for x in (6, 7, 12, 13):
            cell(x, 18, OUTLINE)

        self._draw_face(cell, st)
        self._draw_extras(cell, st)

    def _draw_face(self, cell, st: SpriteState) -> None:
        lx, rx, ey = 6, 12, 10
        blinking = self.t < self._blink_until
        mood = st.mood
        lookx, looky = st.look

        if blinking and mood not in ("joy",):
            for ex in (lx, rx):
                cell(ex, ey + 2, PUPIL)
                cell(ex + 1, ey + 2, PUPIL)
        elif mood == "joy":
            for ex in (lx, rx):
                cell(ex, ey + 1, PUPIL)
                cell(ex + 1, ey, PUPIL)
                cell(ex + 2, ey + 1, PUPIL)
            cell(5, 13, BLUSH)
            cell(15, 13, BLUSH)
        elif mood == "sadness":
            for ex in (lx, rx):
                cell(ex, ey + 1, PUPIL)
                cell(ex + 1, ey + 1, PUPIL)
                cell(ex + 1, ey + 2, PUPIL)
            cell(lx, ey + 3, TEAR)
            if int(self.t * 2) % 2:
                cell(lx, ey + 4, TEAR)
        elif mood == "anger":
            for ex in (lx, rx):
                cell(ex, ey + 1, PUPIL)
                cell(ex + 1, ey + 1, PUPIL)
                cell(ex, ey + 2, PUPIL)
                cell(ex + 1, ey + 2, PUPIL)
            cell(lx - 1, ey - 1, OUTLINE)
            cell(lx, ey - 1, OUTLINE)
            cell(lx + 1, ey, OUTLINE)
            cell(rx + 2, ey - 1, OUTLINE)
            cell(rx + 1, ey - 1, OUTLINE)
            cell(rx, ey, OUTLINE)
        elif mood in ("fear", "surprise"):
            for ex in (lx, rx):
                for dx in (0, 1, 2):
                    for dyy in (0, 1, 2):
                        cell(ex - (1 if ex == lx else 0) + dx, ey + dyy, GLINT)
                cell(ex + (0 if ex == lx else 1), ey + 1, PUPIL)
        else:
            for ex in (lx, rx):
                x0, y0 = ex + lookx, ey + looky  # the whole eye slides a pixel toward the cursor
                for dyy in (0, 1, 2):
                    cell(x0, y0 + dyy, PUPIL)
                    cell(x0 + 1, y0 + dyy, PUPIL)
                cell(x0, y0, GLINT)

        # Mouth.
        if st.talking and int(self.t * 8) % 2 == 0:
            for x in (9, 10):
                cell(x, 14, MOUTH)
                cell(x, 15, MOUTH)
            cell(10, 15, TONGUE)
        elif mood == "joy":
            for x, y in ((8, 14), (9, 15), (10, 15), (11, 14)):
                cell(x, y, MOUTH)
        elif mood == "sadness":
            for x, y in ((8, 15), (9, 14), (10, 14), (11, 15)):
                cell(x, y, MOUTH)
        elif mood in ("surprise", "fear"):
            for x, y in ((9, 14), (10, 14), (9, 15), (10, 15)):
                cell(x, y, MOUTH)
        elif mood == "anger":
            for x in (8, 9, 10, 11):
                cell(x, 15, MOUTH)
        else:
            cell(9, 14, MOUTH)
            cell(10, 14, MOUTH)

    def _draw_extras(self, cell, st: SpriteState) -> None:
        if st.thinking:
            n = int(self.t * 3) % 4
            for i in range(n):
                cell(15 + i * 2, 3, LIGHT)
        if st.mood == "surprise":
            for y in (2, 3, 4):
                cell(17, y, MOOD_COLORS["surprise"])
            cell(17, 6, MOOD_COLORS["surprise"])
        if st.mood == "fear":
            cell(16, 7, TEAR)
            cell(16, 8, TEAR)
            cell(17, 8, TEAR)
        if st.mood == "joy":
            for i, (x, y) in enumerate(((2, 5), (17, 4), (18, 10))):
                if int(self.t * 3 + i) % 3 == 0:
                    cell(x, y, MOOD_COLORS["joy"])
        if st.listening and not st.thinking:
            cell(18, 12, LIGHT)
            cell(19, 11, LIGHT)
            cell(19, 13, LIGHT)
