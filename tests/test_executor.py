from agentic.executor import Executor
from agentic.llm.client import MockLLMClient
from agentic.tools.base import ToolRegistry
from agentic.tools.builtin import calculator


def test_executor_runs_each_planned_step_as_its_own_agent():
    llm = MockLLMClient(
        scripted_responses=[
            # planner decomposition
            "1. Compute 2 + 2\n2. Compute the previous result times 10",
            # subtask 1
            'Thought: add.\nAction: calculator\nAction Input: {"expression": "2 + 2"}',
            "Thought: done.\nFinal Answer: 4",
            # subtask 2
            'Thought: multiply.\nAction: calculator\nAction Input: {"expression": "4 * 10"}',
            "Thought: done.\nFinal Answer: 40",
        ]
    )
    tools = ToolRegistry([calculator])
    executor = Executor(llm=llm, tools=tools)

    report = executor.run("compute 2 + 2 then multiply by 10")

    assert report.succeeded
    assert len(report.results) == 2
    assert report.results[0].final_answer == "4"
    assert report.results[1].final_answer == "40"
    assert "4" in report.summary()
    assert "40" in report.summary()


def test_executor_reports_failure_when_a_subtask_never_completes():
    llm = MockLLMClient(
        scripted_responses=[
            "1. An impossible step",
            # subtask never emits Final Answer within its step budget
            'Thought: stuck.\nAction: calculator\nAction Input: {"expression": "1 + 1"}',
        ]
    )
    tools = ToolRegistry([calculator])
    executor = Executor(llm=llm, tools=tools, max_steps_per_task=1)

    report = executor.run("do the impossible")

    assert not report.succeeded
    assert report.results[0].final_answer is None
