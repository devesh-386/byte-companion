from typing import Annotated

import pytest

from companion.plugins.files import list_directory, read_text_file
from companion.plugins.system import calculate
from companion.tools import ToolError, ToolRegistry


def make_registry():
    reg = ToolRegistry(max_output_chars=50)

    @reg.register
    def add(a: int, b: Annotated[int, "second number"] = 1) -> int:
        """Add two numbers."""
        return a + b

    @reg.register
    def boom() -> str:
        """Always fails."""
        raise RuntimeError("kaboom")

    @reg.register
    def big() -> str:
        """Returns a lot."""
        return "x" * 200

    return reg


def test_schema_from_signature():
    fn = next(s["function"] for s in make_registry().schemas() if s["function"]["name"] == "add")
    assert fn["description"] == "Add two numbers."
    assert fn["parameters"]["required"] == ["a"]
    assert fn["parameters"]["properties"]["b"] == {"type": "integer", "description": "second number", "default": 1}


def test_call_and_coercion():
    reg = make_registry()
    assert reg.call("add", {"a": 2, "b": 3}) == "5"
    assert reg.call("add", {"a": "2"}) == "3"
    assert reg.call("add", {"a": 4.0}) == "5"


@pytest.mark.parametrize("name,args,msg", [
    ("nope", {}, "Unknown tool"),
    ("add", {}, "Missing required"),
    ("add", {"a": 1, "c": 2}, "Unknown argument"),
    ("add", {"a": "two"}, "should be integer"),
    ("add", {"a": True}, "should be integer"),
    ("boom", {}, "RuntimeError: kaboom"),
])
def test_call_errors_become_tool_errors(name, args, msg):
    with pytest.raises(ToolError, match=msg):
        make_registry().call(name, args)


def test_output_is_truncated():
    out = make_registry().call("big", {})
    assert out.startswith("x" * 50) and "truncated" in out


def test_unsupported_param_type_rejected():
    def bad(items: list) -> str:
        """x"""
        return ""

    with pytest.raises(TypeError):
        ToolRegistry().register(bad)


@pytest.mark.parametrize("expr,expected", [
    ("4821 * 377", "1817517"),
    ("(2+3)**2 / 5", "5"),
    ("7 / 2", "3.5"),
    ("-3 + 2^3", "5"),
])
def test_calculate(expr, expected):
    assert calculate(expr) == expected


@pytest.mark.parametrize("expr", ["__import__('os').system('dir')", "open('x')", "2 ** 99999", "abc"])
def test_calculate_rejects_non_arithmetic(expr):
    with pytest.raises(ToolError):
        calculate(expr)


def test_file_tools(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "notes.txt").write_text("hello world", encoding="utf-8")
    listing = list_directory(str(tmp_path))
    assert "[dir]  sub" in listing and "[file] notes.txt" in listing
    assert "contains 1 folders and 1 files" in listing
    assert read_text_file(str(tmp_path / "notes.txt")) == "hello world"
    assert "showing 5 of 11" in read_text_file(str(tmp_path / "notes.txt"), max_chars=5)
    with pytest.raises(ToolError):
        read_text_file(str(tmp_path / "missing.txt"))
    with pytest.raises(ToolError):
        list_directory(str(tmp_path / "notes.txt"))

