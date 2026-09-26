from agentic import Agent, MockLLMClient, ToolRegistry
from agentic.tools.builtin import calculator


def _agent(scripted_responses):
    llm = MockLLMClient(scripted_responses=scripted_responses)
    tools = ToolRegistry([calculator])
    return Agent(llm=llm, tools=tools, max_steps=5), llm


def test_single_tool_call_then_final_answer():
    agent, llm = _agent(
        [
            'Thought: compute it.\nAction: calculator\nAction Input: {"expression": "2 + 2"}',
            "Thought: got it.\nFinal Answer: 4",
        ]
    )
    result = agent.run("what is 2 + 2?")

    assert result.succeeded
    assert result.final_answer == "4"
    assert result.step_count == 2
    assert result.steps[0].action == "calculator"
    assert result.steps[0].observation == "4"


def test_immediate_final_answer_needs_no_tool_call():
    agent, _ = _agent(["Thought: trivial.\nFinal Answer: hello"])
    result = agent.run("say hello")

    assert result.succeeded
    assert result.final_answer == "hello"
    assert result.step_count == 1


def test_unknown_tool_becomes_an_observation_not_a_crash():
    agent, _ = _agent(
        [
            'Thought: try a tool that does not exist.\nAction: nonexistent_tool\nAction Input: {}',
            "Thought: fine, I will answer directly.\nFinal Answer: fallback answer",
        ]
    )
    result = agent.run("do something")

    assert result.succeeded
    assert "unknown tool" in result.steps[0].observation
    assert result.final_answer == "fallback answer"


def test_exhausting_max_steps_without_final_answer_is_not_success():
    # Every scripted response calls a tool, never emits Final Answer.
    scripted = ['Thought: keep going.\nAction: calculator\nAction Input: {"expression": "1 + 1"}'] * 3
    llm = MockLLMClient(scripted_responses=scripted)
    tools = ToolRegistry([calculator])
    agent = Agent(llm=llm, tools=tools, max_steps=3)

    result = agent.run("never finish")

    assert not result.succeeded
    assert result.final_answer is None
    assert result.step_count == 3


def test_memory_transcript_includes_observations():
    agent, llm = _agent(
        [
            'Thought: compute it.\nAction: calculator\nAction Input: {"expression": "3 * 3"}',
            "Thought: got it.\nFinal Answer: 9",
        ]
    )
    agent.run("what is 3 * 3?")

    transcript = agent.memory.as_messages()
    observation_messages = [m for m in transcript if "Observation:" in m["content"]]
    assert len(observation_messages) == 1
    assert "9" in observation_messages[0]["content"]
