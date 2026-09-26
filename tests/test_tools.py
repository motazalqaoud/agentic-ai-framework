import pytest

from agentic.tools.base import Tool, ToolRegistry, tool
from agentic.tools.builtin import calculator, current_datetime, read_file, write_file


def test_tool_decorator_derives_name_and_description():
    @tool(description="Adds two numbers")
    def add(a: float, b: float) -> float:
        return a + b

    assert isinstance(add, Tool)
    assert add.name == "add"
    assert add.description == "Adds two numbers"
    assert add.parameters == {"a": "float", "b": "float"}
    assert add(a=2, b=3) == 5


def test_registry_prevents_duplicate_names():
    registry = ToolRegistry([calculator])
    with pytest.raises(ValueError):
        registry.register(calculator)


def test_registry_lookup_and_render():
    registry = ToolRegistry([calculator, current_datetime])
    assert registry.get("calculator") is calculator
    assert registry.get("nonexistent") is None
    rendered = registry.render_for_prompt()
    assert "calculator" in rendered
    assert "current_datetime" in rendered


def test_calculator_basic_arithmetic():
    assert calculator(expression="2 + 3 * 4") == "14"
    assert calculator(expression="(2 + 3) * 4") == "20"
    assert calculator(expression="-5 + 10") == "5"


def test_calculator_rejects_unsafe_expressions():
    with pytest.raises(ValueError):
        calculator(expression="__import__('os').system('echo hi')")


def test_write_then_read_file(tmp_path):
    target = tmp_path / "nested" / "out.txt"
    result = write_file(path=str(target), content="hello world")
    assert "wrote" in result
    assert target.read_text() == "hello world"

    read_back = read_file(path=str(target))
    assert read_back == "hello world"


def test_read_file_missing_path_reports_error(tmp_path):
    missing = tmp_path / "nope.txt"
    assert "error" in read_file(path=str(missing))
