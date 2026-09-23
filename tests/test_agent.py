from companion.agent import Agent
from companion.tools import CHANGE, ToolRegistry


class FakeLLM:
    """Replays scripted replies token-by-token and records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, *, stop=None, on_token=None, **kw):
        self.seen.append([dict(m) for m in messages])
        reply = self.replies.pop(0)
        for i in range(0, len(reply), 3):
            if on_token:
                on_token(reply[i:i + 3])
        return reply


def make_agent(replies, **kw):
    reg = ToolRegistry()

    @reg.register
    def get_current_time() -> str:
        """Time."""
        return "Tuesday 10:00"

    llm = FakeLLM(replies)
    return Agent(llm, reg, "SYSTEM", **kw), llm


def collect(agent, text):
    events = []
    answer = agent.run(text, on_event=events.append)
    return answer, events


def test_plain_answer():
    agent, llm = make_agent(["Hi there!"])
    answer, events = collect(agent, "hello")
    assert answer == "Hi there!"
    assert "".join(e.data["text"] for e in events if e.kind == "text") == "Hi there!"
    assert llm.seen[0] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "hello"}]


def test_tool_call_then_answer():
    agent, llm = make_agent([
        '<tool_call>\n{"name": "get_current_time", "arguments": {}}\n</tool_call>',
        "It's Tuesday, 10 AM.",
    ])
    answer, events = collect(agent, "what time is it")
    assert answer == "It's Tuesday, 10 AM."
    assert [e.kind for e in events if e.kind != "text"] == ["thinking", "tool_call", "tool_result", "thinking", "done"]
    # The second model call must include the tool result the first call asked for.
    last_msg = llm.seen[1][-1]
    assert last_msg["role"] == "user" and "<tool_response>" in last_msg["content"]
    assert "Tuesday 10:00" in last_msg["content"]
    # The raw tool-call markup never reaches the screen.
    assert "tool_call" not in "".join(e.data["text"] for e in events if e.kind == "text")


def test_unknown_tool_is_fed_back_and_model_recovers():
    agent, llm = make_agent([
        '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>',
        "Sorry, I can't check the weather yet.",
    ])
    answer, events = collect(agent, "weather?")
    assert answer.startswith("Sorry")
    assert any(e.kind == "tool_error" for e in events)
    assert "Unknown tool 'get_weather'" in llm.seen[1][-1]["content"]


def test_malformed_call_is_fed_back():
    agent, llm = make_agent(["<tool_call>{oops}</tool_call>", "Let me answer directly."])
    answer, _ = collect(agent, "x")
    assert answer == "Let me answer directly."
    assert "invalid_call" in llm.seen[1][-1]["content"]


def test_step_limit():
    call = '<tool_call>{"name": "get_current_time", "arguments": {}}</tool_call>'
    agent, _ = make_agent([call] * 3, max_steps=3)
    answer, events = collect(agent, "loop forever")
    assert "Stopped after 3" in answer
    assert [e.kind for e in events][-2:] == ["step_limit", "done"]


def test_history_carries_across_turns_and_trims_whole_turns():
    agent, llm = make_agent(["A" * 50, "B" * 50, "C"], history_budget_chars=120)
    collect(agent, "first")
    collect(agent, "second")
    assert [m["content"] for m in llm.seen[1]] == ["SYSTEM", "first", "A" * 50, "second"]
    collect(agent, "third")
    # Budget exceeded, so the oldest whole turn is dropped.
    assert [m["content"] for m in llm.seen[2]][:2] == ["SYSTEM", "second"]


class RecordingHook:
    def __init__(self, context):
        self.context = context
        self.after = []

    def before_turn(self, user_text, emit):
        from companion.agent_events import Event
        emit(Event("emotion", {"label": "joy"}))
        return self.context

    def after_turn(self, user_text, answer):
        self.after.append((user_text, answer))


def test_hook_context_goes_into_system_prompt_for_that_turn_only():
    hook = RecordingHook("User sounds happy.")
    reg = ToolRegistry()
    llm = FakeLLM(["yay", "ok"])
    agent = Agent(llm, reg, "SYSTEM", hooks=[hook])
    _, events = collect(agent, "I passed")
    assert "User sounds happy." in llm.seen[0][0]["content"]
    assert events[0].kind == "emotion"
    assert hook.after == [("I passed", "yay")]
    hook.context = None
    collect(agent, "next")
    assert llm.seen[1][0]["content"] == "SYSTEM"


def make_confirm_agent(replies, approve):
    reg = ToolRegistry()
    ran = []

    @reg.register(tier=CHANGE)
    def delete_everything(path: str) -> str:
        """Dangerous."""
        ran.append(path)
        return "deleted"

    asked = []

    def confirm(call, summary):
        asked.append(summary)
        return approve

    llm = FakeLLM(replies)
    return Agent(llm, reg, "S", confirm=confirm), llm, ran, asked


def test_confirmation_declined_does_not_run_tool():
    call = '<tool_call>{"name": "delete_everything", "arguments": {"path": "C:/"}}</tool_call>'
    agent, llm, ran, asked = make_confirm_agent([call, "Okay, I won't."], approve=False)
    collect(agent, "delete it")
    assert ran == [] and asked == ["delete_everything(path='C:/')"]
    assert "declined" in llm.seen[1][-1]["content"]


def test_confirmation_approved_runs_tool():
    call = '<tool_call>{"name": "delete_everything", "arguments": {"path": "x"}}</tool_call>'
    agent, _, ran, _ = make_confirm_agent([call, "Done."], approve=True)
    collect(agent, "delete x")
    assert ran == ["x"]


def test_reset_clears_history():
    agent, llm = make_agent(["one", "two"])
    collect(agent, "a")
    agent.reset()
    collect(agent, "b")
    assert len(llm.seen[1]) == 2
