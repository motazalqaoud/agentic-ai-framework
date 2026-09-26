"""Planner: breaks a high-level goal into an ordered list of subtasks before execution.

This is an optional layer above the raw ReAct Agent. For simple goals ("what's 2+2"),
skip it and call Agent.run directly. For a goal that clearly has multiple independent
phases ("research X, then summarize it, then write it to a file"), planning first keeps
the agent from wandering: each subtask becomes its own bounded Agent.run call with a
fresh, focused instruction, and the Executor (see executor.py) chains them together.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .llm.client import LLMClient

_PLANNER_SYSTEM_PROMPT = """You are a task-decomposition planner.
Given a goal, break it into a short ordered list of concrete, independently-actionable
subtasks. Respond with ONLY a numbered list, one subtask per line, nothing else.
Keep it to at most 6 steps. Each step should be a complete, self-contained instruction
that could be executed on its own with no additional context beyond the goal."""


@dataclass
class Plan:
    goal: str
    steps: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)


class Planner:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(self, goal: str) -> Plan:
        raw = self.llm.complete(
            system_prompt=_PLANNER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Goal: {goal}"}],
        )
        steps = self._parse_numbered_list(raw)
        if not steps:
            # Degenerate case (e.g. a mock client with nothing scripted, or a goal too
            # simple to decompose): fall back to a single-step plan of the goal itself.
            steps = [goal]
        return Plan(goal=goal, steps=steps)

    @staticmethod
    def _parse_numbered_list(text: str) -> List[str]:
        steps = []
        for line in text.splitlines():
            match = re.match(r"\s*\d+[\.\)]\s*(.+)", line)
            if match:
                steps.append(match.group(1).strip())
        return steps
