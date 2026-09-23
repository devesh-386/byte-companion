import json

from companion.protocol import (VisibleTextFilter, build_system_prompt, format_tool_response,
                                parse_tool_calls, strip_tool_calls)


def test_parses_single_call():
    calls, errors = parse_tool_calls('Sure.\n<tool_call>\n{"name": "calculate", "arguments": {"expression": "2+2"}}\n</tool_call>')
    assert errors == []
    assert [(c.name, c.arguments) for c in calls] == [("calculate", {"expression": "2+2"})]


def test_parses_multiple_calls_and_missing_close_tag():
    text = ('<tool_call>{"name": "a", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "b", "arguments": {"x": 1}}')
    calls, errors = parse_tool_calls(text)
    assert errors == []
    assert [c.name for c in calls] == ["a", "b"]


def test_string_encoded_arguments_are_decoded():
    calls, _ = parse_tool_calls('<tool_call>{"name": "a", "arguments": "{\\"x\\": 1}"}</tool_call>')
    assert calls[0].arguments == {"x": 1}


def test_bad_json_becomes_error_not_exception():
    calls, errors = parse_tool_calls("<tool_call>{name: calculate}</tool_call>")
    assert calls == [] and len(errors) == 1


def test_missing_name_is_error():
    calls, errors = parse_tool_calls('<tool_call>{"arguments": {}}</tool_call>')
    assert calls == [] and "name" in errors[0]


def test_plain_answer_has_no_calls():
    assert parse_tool_calls("It is sunny.") == ([], [])


def test_strip_tool_calls():
    assert strip_tool_calls('Checking.<tool_call>{"name":"a"}</tool_call>') == "Checking."


def test_system_prompt_contains_schema():
    schema = {"type": "function", "function": {"name": "calculate"}}
    prompt = build_system_prompt("You are X.", [schema])
    assert prompt.startswith("You are X.")
    assert json.dumps(schema) in prompt
    assert "<tool_call>" in prompt


def test_tool_response_format():
    out = format_tool_response("calculate", "4")
    assert out.startswith("<tool_response>") and out.endswith("</tool_response>")
    assert json.loads(out.splitlines()[1]) == {"name": "calculate", "content": "4"}


def _run_filter(chunks):
    f = VisibleTextFilter()
    return "".join(f.feed(c) for c in chunks) + f.flush()


def test_filter_passes_plain_text():
    assert _run_filter(["Hel", "lo ", "there"]) == "Hello there"


def test_filter_hides_tag_split_across_chunks():
    assert _run_filter(["Let me check. <to", "ol_c", 'all>{"name":', '"x"}</tool_call>']) == "Let me check. "


def test_filter_releases_false_alarm():
    # "<to" looks like the start of the tag but turns out to be ordinary text.
    assert _run_filter(["a <to", "p> b"]) == "a <top> b"


def test_filter_flushes_held_back_suffix():
    assert _run_filter(["ends with <tool"]) == "ends with <tool"
