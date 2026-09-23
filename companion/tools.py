import inspect
import json
import typing
from dataclasses import dataclass
from typing import Annotated, Any, Callable

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


class ToolError(Exception):
    pass


@dataclass
class ToolImage:
    """A tool result the model should *see* (a screenshot), plus a line of text about it."""
    png: bytes
    text: str

    def data_url(self) -> str:
        import base64
        return "data:image/png;base64," + base64.b64encode(self.png).decode()


# How much a tool can change the world. Enforced by the agent in code, never left to the prompt.
LOOK = 0      # read-only: files, screen, web, system info            -> free
OPEN = 1      # open / navigate / play-pause / focus / reminders       -> free, logged, stoppable
CHANGE = 2    # click, type, write files, forget memories              -> asks Allow / Deny
NEVER = 3     # delete, send, pay                                      -> refused outright
TIER_NAMES = {LOOK: "look", OPEN: "open", CHANGE: "change", NEVER: "never"}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]
    types: dict[str, type]
    tier: int = LOOK
    dry: Callable[..., Any] | None = None   # what the tool "would do", used in dry-run mode
    examples: tuple[str, ...] = ()          # things a user might say; help tool retrieval


class ToolRegistry:
    def __init__(self, max_output_chars: int = 4000, dry_run: bool = False):
        self._tools: dict[str, Tool] = {}
        self.max_output_chars = max_output_chars
        # Dry run: tools that open or change something (tier >= OPEN) only describe what they would do.
        # Used to record training data without touching the laptop. Reading tools still run for real.
        self.dry_run = dry_run

    def register(self, func: Callable[..., Any] | None = None, *, tier: int = LOOK,
                 dry: Callable[..., Any] | None = None, examples: tuple[str, ...] = ()):
        """Decorator. The function's name, docstring and type hints become the schema the model sees.
        Use Annotated[type, "description"] to describe a parameter. `tier` says how much it can change.
        `dry` takes the same arguments and returns what would happen, without doing it."""
        if tier not in TIER_NAMES:
            raise ValueError(f"unknown tier {tier!r}")
        if func is None:
            return lambda f: self.register(f, tier=tier, dry=dry, examples=examples)
        if func.__name__ in self._tools:
            raise ValueError(f"a tool called '{func.__name__}' is already registered")
        hints = typing.get_type_hints(func, include_extras=True)
        properties: dict[str, dict] = {}
        required: list[str] = []
        types: dict[str, type] = {}

        for pname, param in inspect.signature(func).parameters.items():
            hint = hints.get(pname, str)
            desc = None
            if typing.get_origin(hint) is Annotated:
                hint, *meta = typing.get_args(hint)
                desc = next((m for m in meta if isinstance(m, str)), None)
            if hint not in _JSON_TYPES:
                raise TypeError(f"{func.__name__}.{pname}: unsupported type {hint!r}")
            types[pname] = hint
            prop = {"type": _JSON_TYPES[hint]}
            if desc:
                prop["description"] = desc
            if param.default is inspect.Parameter.empty:
                required.append(pname)
            else:
                prop["default"] = param.default
            properties[pname] = prop

        self._tools[func.__name__] = Tool(
            name=func.__name__,
            description=inspect.cleandoc(func.__doc__ or "").strip(),
            parameters={"type": "object", "properties": properties, "required": required},
            func=func,
            types=types,
            tier=tier,
            dry=dry,
            examples=tuple(examples),
        )
        return func

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def override(self, name: str, func: Callable[..., Any]) -> None:
        """Swap a tool's implementation but keep its schema (used to sandbox tools in training)."""
        self._tools[name].func = func

    def tier(self, name: str) -> int:
        """Unknown tools count as LOOK here; call() rejects them properly."""
        tool = self._tools.get(name)
        return tool.tier if tool else LOOK

    def schemas(self, max_tier: int = NEVER, only: set[str] | None = None) -> list[dict]:
        """Tools above max_tier are hidden from the model entirely, not just refused.
        `only` narrows it further to the tools picked for this turn (registry order is kept)."""
        return [
            {"type": "function",
             "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in self._tools.values() if t.tier <= max_tier and (only is None or t.name in only)
        ]

    def describe_call(self, name: str, arguments: dict) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
        return f"{name}({args})"

    def validate(self, name: str, arguments: dict) -> dict:
        """Checks and converts arguments without running anything. Returns the kwargs to call with."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"Unknown tool '{name}'. Available: {', '.join(self._tools)}")
        if not isinstance(arguments, dict):
            raise ToolError("arguments must be a JSON object")
        unknown = set(arguments) - set(tool.types)
        if unknown:
            raise ToolError(f"Unknown argument(s) for {name}: {', '.join(sorted(unknown))}")
        missing = [p for p in tool.parameters["required"] if p not in arguments]
        if missing:
            raise ToolError(f"Missing required argument(s) for {name}: {', '.join(missing)}")
        return {k: _coerce(v, tool.types[k], k) for k, v in arguments.items()}

    def call(self, name: str, arguments: dict) -> "str | ToolImage":
        kwargs = self.validate(name, arguments)
        tool = self._tools[name]
        func = tool.func
        if self.dry_run and tool.tier > LOOK:
            func = tool.dry or (lambda **kw: f"(dry run) would call {self.describe_call(name, kw)}")

        try:
            result = func(**kwargs)
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"{type(e).__name__}: {e}") from e

        if isinstance(result, ToolImage):
            return result
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        if len(text) > self.max_output_chars:
            text = text[: self.max_output_chars] + f"\n...[truncated, {len(text)} chars total]"
        return text


def _coerce(value: Any, target: type, pname: str) -> Any:
    """Small models often send "5" instead of 5 — accept the obvious conversions."""
    if isinstance(value, target) and not (target is int and isinstance(value, bool)):
        return value
    try:
        if target is bool and isinstance(value, str):
            if value.lower() in ("true", "yes", "1"):
                return True
            if value.lower() in ("false", "no", "0"):
                return False
            raise ValueError
        if target is int and isinstance(value, float) and value.is_integer():
            return int(value)
        if target in (int, float) and isinstance(value, str):
            return target(value.strip())
        if target is float and isinstance(value, int):
            return float(value)
        if target is str and isinstance(value, (int, float)):
            return str(value)
    except ValueError:
        pass
    raise ToolError(f"Argument '{pname}' should be {_JSON_TYPES[target]}, got {value!r}")
