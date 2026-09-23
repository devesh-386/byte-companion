"""The tool-calling protocol between us and the model.

Qwen2.5 was trained on a specific format: tool signatures inside <tools></tools> in the system
prompt, calls emitted as <tool_call>{json}</tool_call>, results fed back as
<tool_response>...</tool_response>. We speak that format by hand instead of letting a library
hide it, so every byte the model sees is ours.
"""
import json
import re
from dataclasses import dataclass

CALL_OPEN = "<tool_call>"
CALL_CLOSE = "</tool_call>"
RESPONSE_OPEN = "<tool_response>"

# A closing tag may be missing if generation stopped early, so also accept end-of-text.
_CALL_RE = re.compile(re.escape(CALL_OPEN) + r"(.*?)(?:" + re.escape(CALL_CLOSE) + r"|\Z)", re.S)


@dataclass
class ToolCall:
    name: str
    arguments: dict


def build_system_prompt(persona: str, tool_schemas: list[dict]) -> str:
    tools = "\n".join(json.dumps(s, ensure_ascii=False) for s in tool_schemas)
    return (
        f"{persona}\n\n"
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{tools}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )


THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.S)
# Qwen3.5 / Qwen3-Coder style: <function=name><parameter=p>value</parameter></function>
_XML_FN_RE = re.compile(r"<function=([\w.-]+)>(.*?)(?:</function>|\Z)", re.S)
_XML_PARAM_RE = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.S)


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _parse_xml_call(body: str) -> ToolCall | None:
    m = _XML_FN_RE.search(body)
    if not m:
        return None
    args = {}
    for p in _XML_PARAM_RE.finditer(m.group(2)):
        raw = p.group(2).strip()
        try:
            args[p.group(1)] = json.loads(raw) if raw[:1] in "[{0123456789-" or raw in ("true", "false") else raw
        except json.JSONDecodeError:
            args[p.group(1)] = raw
    return ToolCall(name=m.group(1), arguments=args)


def parse_tool_calls(text: str) -> tuple[list[ToolCall], list[str]]:
    """Returns (valid calls, error messages for calls we couldn't understand)."""
    calls: list[ToolCall] = []
    errors: list[str] = []
    text = strip_thinking(text)
    for match in _CALL_RE.finditer(text):
        body = match.group(1).strip()
        xml = _parse_xml_call(body) if body.startswith("<function=") else None
        if xml:
            calls.append(xml)
            continue
        try:
            obj = json.loads(body)
        except json.JSONDecodeError as e:
            errors.append(f"Could not parse tool call as JSON ({e.msg}): {body[:200]}")
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            errors.append(f'Tool call must be {{"name": ..., "arguments": {{...}}}}, got: {body[:200]}')
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):
            # Some models double-encode arguments as a JSON string.
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                errors.append(f"Arguments for {obj['name']} are not valid JSON: {args[:200]}")
                continue
        if args is None:
            args = {}
        calls.append(ToolCall(name=obj["name"], arguments=args))
    return calls, errors


def strip_tool_calls(text: str) -> str:
    return _CALL_RE.sub("", strip_thinking(text)).strip()


def format_tool_response(name: str, content: str) -> str:
    payload = json.dumps({"name": name, "content": content}, ensure_ascii=False)
    return f"{RESPONSE_OPEN}\n{payload}\n</tool_response>"


class VisibleTextFilter:
    """Streams tokens to the screen but hides everything from <tool_call> onwards, and any
    <think>...</think> block (a thinking model's private notes).

    Tokens arrive in arbitrary pieces ("<to", "ol_c", "all>"), so any tail that could still be
    the start of a tag is held back until we know which way it goes.
    """

    def __init__(self, tag: str = CALL_OPEN):
        self.tag = tag
        self._pending = ""
        self._hidden = False      # after a tool call: hidden for good
        self._thinking = False    # inside <think>: hidden until </think>

    def feed(self, chunk: str) -> str:
        if self._hidden:
            return ""
        self._pending += chunk
        out = []
        while True:
            if self._thinking:
                idx = self._pending.find(THINK_CLOSE)
                if idx == -1:
                    self._pending = self._pending[-(len(THINK_CLOSE) - 1):]  # only a partial close tag matters
                    return "".join(out)
                self._thinking = False
                self._pending = self._pending[idx + len(THINK_CLOSE):].lstrip("\n")
                continue
            hits = [(i, t) for t in (self.tag, THINK_OPEN) if (i := self._pending.find(t)) != -1]
            if not hits:
                break
            idx, tag = min(hits)
            out.append(self._pending[:idx])
            if tag == self.tag:
                self._hidden, self._pending = True, ""
                return "".join(out)
            self._thinking = True
            self._pending = self._pending[idx + len(THINK_OPEN):]
        keep = max(self._partial_suffix(self._pending, t) for t in (self.tag, THINK_OPEN))
        out.append(self._pending[: len(self._pending) - keep])
        self._pending = self._pending[len(self._pending) - keep:]
        return "".join(out)

    def flush(self) -> str:
        out, self._pending = ("" if self._hidden or self._thinking else self._pending), ""
        return out

    @staticmethod
    def _partial_suffix(text: str, tag: str) -> int:
        for n in range(min(len(tag) - 1, len(text)), 0, -1):
            if text.endswith(tag[:n]):
                return n
        return 0
