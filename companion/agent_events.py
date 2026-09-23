from dataclasses import dataclass
from typing import Literal

EventKind = Literal["text", "tool_call", "tool_result", "tool_error", "step_limit", "emotion", "memory",
                    "thinking", "stopped", "done", "tools_shown"]


@dataclass
class Event:
    kind: EventKind
    data: dict
