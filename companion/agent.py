"""The agent loop, and the one place where every safety rule is enforced.

    user text ─► hooks (mood, memories) ─► model ─┬─► plain answer ─► done
                                                   └─► <tool_call> ─► tier gate ─► tool ─► back to model
                                                                        │
                        LOOK/OPEN: run · CHANGE: ask the user · NEVER or above max_tier: refuse
                        every call ─► action log · the cancel token is checked between every step
"""
import json
import re
import time
from typing import Callable, Sequence

from .actionlog import ActionLog
from .agent_events import Event, EventKind  # noqa: F401  (re-exported for callers)
from .cancel import STOP, Cancelled, CancelToken
from .hooks import TurnHook
from .llm import LLM, Message
from .protocol import (RESPONSE_OPEN, ToolCall, VisibleTextFilter, format_tool_response,
                       parse_tool_calls, strip_tool_calls)
from .tools import CHANGE, LOOK, NEVER, TIER_NAMES, ToolError, ToolImage, ToolRegistry

ConfirmFn = Callable[[ToolCall, str], bool]
STOPPED_ANSWER = "(Stopped.)"


class Agent:
    def __init__(self, llm: LLM, registry: ToolRegistry, system_prompt: str,
                 max_steps: int = 6, history_budget_chars: int = 21000,
                 hooks: Sequence[TurnHook] = (), confirm: ConfirmFn | None = None,
                 max_tier: int = CHANGE, action_log: ActionLog | None = None,
                 tool_picker=None, prompt_for_tools: Callable[[list[dict]], str] | None = None,
                 context_in_user_turn: bool = False):
        self.context_in_user_turn = context_in_user_turn
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.history_budget_chars = history_budget_chars
        self.hooks = list(hooks)
        self.confirm = confirm
        self.max_tier = max_tier
        self.action_log = action_log
        # Tool retrieval: each message gets a system prompt listing only the tools picked for it.
        self.tool_picker = tool_picker if prompt_for_tools else None
        self.prompt_for_tools = prompt_for_tools
        self._last_used: set[str] = set()  # tools used last turn stay visible for follow-ups ("again")
        # One entry per user turn: the user message plus every assistant/tool message it caused.
        # Trimming whole turns keeps a tool call and its result from being split apart.
        self.turns: list[list[Message]] = []
        self._turn_context = ""

    def reset(self) -> None:
        self.turns.clear()

    def messages(self) -> list[Message]:
        history = [m for turn in self.turns for m in turn]
        system = self.system_prompt
        if self._turn_context and not self.context_in_user_turn:
            system += "\n\n# Context for this message\n" + self._turn_context
        elif self._turn_context and self.turns:
            # D3: mood and memories ride on the current user message, so the system prompt stays identical
            # between messages and llama-server can reuse its cached prefix.
            first = self.turns[-1][0]
            i = history.index(first)
            history[i] = {**first, "content": with_text_prefix(first["content"],
                                                               f"[Context]\n{self._turn_context}\n[/Context]\n\n")}
        return [{"role": "system", "content": system}] + history

    def run(self, user_text: str, on_event: Callable[[Event], None] | None = None,
            cancel: CancelToken | None = None) -> str:
        emit = on_event or (lambda e: None)
        # Every run is registered with STOP, so the hotkey reaches it even if the caller passed no token.
        token = cancel or STOP.new_token()
        own_token = cancel is None
        try:
            extras = [c for c in (h.before_turn(user_text, emit) for h in self.hooks) if c]
            self._turn_context = "\n".join(extras)
            if self.tool_picker:
                shown = set(self.tool_picker.pick(user_text)) | self._last_used
                self.system_prompt = self.prompt_for_tools(self.registry.schemas(self.max_tier, only=shown))
                emit(Event("tools_shown", {"names": sorted(shown)}))
                self._last_used = set()
            turn: list[Message] = [{"role": "user", "content": user_text}]
            self.turns.append(turn)
            self._trim_history()
            try:
                answer = self._loop(turn, emit, token)
            except Cancelled:
                # Keep the history well-formed: the assistant "answers" that it was stopped.
                turn.append({"role": "assistant", "content": STOPPED_ANSWER})
                emit(Event("stopped", {"reason": token.reason}))
                answer = STOPPED_ANSWER
            for h in self.hooks:
                h.after_turn(user_text, answer)
            emit(Event("done", {"answer": answer}))
            return answer
        finally:
            if own_token:
                STOP.release(token)

    def _loop(self, turn: list[Message], emit: Callable[[Event], None], token: CancelToken) -> str:
        used_tools = False
        nudged = False
        last_call = None
        for _ in range(self.max_steps):
            token.check()
            visible = VisibleTextFilter()

            def on_token(tok: str) -> None:
                shown = visible.feed(tok)
                if shown:
                    emit(Event("text", {"text": shown}))

            emit(Event("thinking", {}))
            # Stop before the model starts imagining a tool's result instead of waiting for it.
            raw = self.llm.chat(self.messages(), stop=[RESPONSE_OPEN], on_token=on_token, cancel=token)
            tail = visible.flush()
            if tail:
                emit(Event("text", {"text": tail}))
            turn.append({"role": "assistant", "content": raw})

            calls, parse_errors = parse_tool_calls(raw)
            if not calls and not parse_errors:
                answer = strip_tool_calls(raw)
                if not used_tools and not nudged and claims_action(answer) and self.registry.names():
                    # Small models sometimes *say* they clicked/opened something without calling a tool.
                    # One nudge: do it for real, or say it can't be done.
                    nudged = True
                    emit(Event("tool_error", {"name": "honesty check",
                                              "error": "claimed an action without doing it; asked to retry"}))
                    turn.append({"role": "user", "content": UNBACKED_CLAIM})
                    continue
                return answer
            used_tools = True

            responses, images = [], []
            for call in calls:
                token.check()
                if call.name in self.registry.names():
                    self._last_used.add(call.name)
                key = (call.name, json.dumps(call.arguments, sort_keys=True))
                if key == last_call and self.registry.tier(call.name) == LOOK:
                    # Small models loop on "look again" (T11: five observe_window calls in a row).
                    emit(Event("tool_error", {"name": call.name, "error": "repeated the same look; told to act"}))
                    responses.append(format_tool_response(call.name, REPEATED_LOOK))
                    continue
                last_call = key
                responses.append(self._run_tool(call, emit, token, images))
            for err in parse_errors:
                emit(Event("tool_error", {"name": "?", "error": err}))
                responses.append(format_tool_response("invalid_call", f"ERROR: {err}"))

            # Qwen expects tool results inside a user turn, wrapped in <tool_response> tags.
            text = "\n".join(responses)
            if images:
                turn.append({"role": "user", "content": [{"type": "text", "text": text}] + [
                    {"type": "image_url", "image_url": {"url": img.data_url()}} for img in images]})
                drop_old_images(self.turns)
            else:
                turn.append({"role": "user", "content": text})

        emit(Event("step_limit", {"max_steps": self.max_steps}))
        return f"(Stopped after {self.max_steps} tool steps without a final answer.)"

    def _run_tool(self, call: ToolCall, emit: Callable[[Event], None], token: CancelToken,
                  images: list | None = None) -> str:
        emit(Event("tool_call", {"name": call.name, "arguments": call.arguments}))
        tier = self.registry.tier(call.name)
        decision, outcome, detail, started = "auto", "ok", "", time.monotonic()
        try:
            if tier >= NEVER:
                decision = "blocked"
                raise ToolError("This action is never allowed. Tell the user you can't do it.")
            if tier > self.max_tier:
                decision = "blocked"
                raise ToolError(f"'{call.name}' ({TIER_NAMES[tier]}) is not allowed here. Do not retry it.")
            if tier == CHANGE:
                self.registry.validate(call.name, call.arguments)  # never ask the user to approve a broken call
                summary = self.registry.describe_call(call.name, call.arguments)
                if self.confirm is None or not self.confirm(call, summary):
                    decision = "denied"
                    raise ToolError("The user declined this action. Tell them it was not done.")
                decision = "allowed"
                token.check()  # the user may have hit stop while the Allow/Deny prompt was up
            result = self.registry.call(call.name, call.arguments)
            if isinstance(result, ToolImage):
                if images is not None:
                    images.append(result)
                result = result.text + " (the image is attached below)"
            detail = result
            emit(Event("tool_result", {"name": call.name, "result": result}))
        except ToolError as e:
            outcome, detail = ("skipped" if decision in ("blocked", "denied") else "error"), str(e)
            result = f"ERROR: {e}"
            emit(Event("tool_error", {"name": call.name, "error": str(e)}))
        except Cancelled:
            outcome = "cancelled"
            raise
        finally:
            if self.action_log:
                self.action_log.write(tool=call.name, tier=tier, arguments=call.arguments, decision=decision,
                                      outcome=outcome, detail=detail, seconds=time.monotonic() - started)
        return format_tool_response(call.name, result)

    def _trim_history(self) -> None:
        drop_old_images(self.turns)

        def size() -> int:
            return len(self.system_prompt) + sum(content_chars(m["content"]) for t in self.turns for m in t)

        while len(self.turns) > 1 and size() > self.history_budget_chars:
            self.turns.pop(0)


_CLAIM = re.compile(
    r"\b(?:i(?:'ve| have)?\s+(?:just\s+|now\s+|already\s+)?|i'm\s+)(?:clicked|opened|closed|launched|started|"
    r"set|added|saved|typed|pressed|played|paused|written|wrote|created|deleted|moved|turned|changed|muted|"
    r"switched|dismissed|opening|launching|clicking|playing)\b"
    # ...and the cousin: promising instead of doing ("I'll open Notepad... ready?")
    r"|\bi(?:'ll| will)\s+(?:now\s+|go\s+ahead\s+and\s+)?(?:open|click|close|launch|start|set|add|save|type|"
    r"press|play|pause|write|create|turn|change|mute|switch|dismiss)\b"
    # ...and the headline form: "Clicking OK on that dialog for you!"
    r"|(?:^|[.!?]\s+|,\s*|!\s*)(?:clicking|opening|closing|launching|starting|playing|saving|typing|pressing|"
    r"setting|adding|muting|switching|dismissing)\s+\w", re.I)
UNBACKED_CLAIM = ("[System check] You described an action, but you called no tool this turn, so nothing "
                  "actually happened. Don't promise or ask for permission (risky actions get an Allow/Deny "
                  "prompt automatically): call the right tool now. If no tool can do it, tell the user plainly "
                  "that you couldn't.")


REPEATED_LOOK = ("You just made this exact call and nothing has changed since; the result above still holds. "
                 "Act on it (for example click an element by its id), try a different approach, or tell "
                 "the user what's blocking you.")


def claims_action(answer: str) -> bool:
    return bool(_CLAIM.search(answer))


IMAGE_CHARS = 7000  # a screenshot costs ~2k tokens; counted as this many characters when trimming (A1)
IMAGE_REMOVED = "[an older screenshot was here; take a new one if you need to see the screen]"


def content_chars(content) -> int:
    if isinstance(content, str):
        return len(content)
    return sum(IMAGE_CHARS if p.get("type") == "image_url" else len(p.get("text", "")) for p in content)


def with_text_prefix(content, prefix: str):
    if isinstance(content, str):
        return prefix + content
    return [{"type": "text", "text": prefix}] + list(content)


def drop_old_images(turns: list[list[Message]], keep: int = 1) -> None:
    """At most `keep` images stay in the history (the newest); older ones become a short note."""
    seen = 0
    for turn in reversed(turns):
        for m in reversed(turn):
            if isinstance(m["content"], list):
                parts = []
                for p in reversed(m["content"]):
                    if p.get("type") == "image_url":
                        seen += 1
                        if seen > keep:
                            p = {"type": "text", "text": IMAGE_REMOVED}
                    parts.append(p)
                m["content"] = parts[::-1]
