"""Every tool call Byte makes, one JSON line each, so you can always see what it did and why."""
import json
import threading
import time
from pathlib import Path


class ActionLog:
    def __init__(self, path: Path, context: str):
        self.path = path
        self.context = context
        self._lock = threading.Lock()

    def write(self, *, tool: str, tier: int, arguments: dict, decision: str, outcome: str,
              detail: str = "", seconds: float = 0.0) -> None:
        """decision: auto | allowed | denied | blocked · outcome: ok | error | skipped | cancelled"""
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "context": self.context,
            "tool": tool,
            "tier": tier,
            "args": arguments,
            "decision": decision,
            "outcome": outcome,
            "detail": detail[:500],
            "seconds": round(seconds, 3),
        }
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass  # logging must never take the agent down


class MemoryActionLog(ActionLog):
    """Keeps entries in a list instead of a file (tests, and the suite's per-task view)."""

    def __init__(self, context: str = "test"):
        super().__init__(Path("unused"), context)
        self.entries: list[dict] = []

    def write(self, **kw) -> None:
        self.entries.append({"context": self.context, **kw})
