from agentic.llm.client import MockLLMClient
from agentic.planner import Planner


def test_planner_parses_numbered_list():
    llm = MockLLMClient(scripted_responses=["1. Search for X\n2. Summarize findings\n3. Write to file"])
    plan = Planner(llm).plan("research X and write it up")

    assert plan.goal == "research X and write it up"
    assert len(plan) == 3
    assert plan.steps == ["Search for X", "Summarize findings", "Write to file"]


def test_planner_falls_back_to_single_step_goal_when_unparseable():
    llm = MockLLMClient(scripted_responses=["I refuse to make a list today."])
    plan = Planner(llm).plan("do the thing")

    assert list(plan) == ["do the thing"]


def test_planner_ignores_non_numbered_lines():
    llm = MockLLMClient(
        scripted_responses=["Here is my plan:\n1. First step\nSome commentary\n2. Second step"]
    )
    plan = Planner(llm).plan("goal")

    assert plan.steps == ["First step", "Second step"]
