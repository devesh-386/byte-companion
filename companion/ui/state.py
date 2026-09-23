"""Small persistent game state: XP, level, window position."""
import json
import math
from pathlib import Path

XP_PER_MESSAGE = 10
XP_PER_TOOL = 5
XP_PER_MEMORY = 15


def level_for(xp: int) -> int:
    # Level n starts at 50*(n-1)^2 XP: quick early levels, slower later ones.
    return int(math.sqrt(max(0, xp) / 50)) + 1


def level_progress(xp: int) -> float:
    lv = level_for(xp)
    lo, hi = 50 * (lv - 1) ** 2, 50 * lv ** 2
    return (xp - lo) / (hi - lo)


class GameState:
    def __init__(self, path: Path):
        self.path = path
        self.xp = 0
        self.pos: list[int] | None = None
        self.speak = True
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.xp = int(data.get("xp", 0))
            self.pos = data.get("pos")
            self.speak = bool(data.get("speak", True))
        except (OSError, ValueError):
            pass

    @property
    def level(self) -> int:
        return level_for(self.xp)

    def add_xp(self, amount: int) -> bool:
        """Returns True if this pushed Byte up a level."""
        before = self.level
        self.xp += amount
        self.save()
        return self.level > before

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"xp": self.xp, "pos": self.pos, "speak": self.speak}),
                                 encoding="utf-8")
        except OSError:
            pass
