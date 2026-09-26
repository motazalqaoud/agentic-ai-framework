"""Executor: chains a Plan (see planner.py) into a sequence of bounded Agent.run calls.

Each subtask gets its own fresh Agent memory, seeded with the subtask instruction plus a
short summary of what prior subtasks produced -- this keeps each step focused instead of
accumulating one giant, drifting transcript across the whole goal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .agent import Agent, AgentResult
from .llm.client import LLMClient
from .memory import Memory
from .planner import Plan, Planner
from .tools.base import ToolRegistry


@dataclass
class ExecutionReport:
    plan: Plan
    results: List[AgentResult] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return bool(self.results) and all(r.succeeded for r in self.results)

    def summary(self) -> str:
        lines = [f"Goal: {self.plan.goal}", ""]
        for i, (step, result) in enumerate(zip(self.plan.steps, self.results), start=1):
            status = "done" if result.succeeded else "incomplete"
            answer = result.final_answer or "(no answer)"
            lines.append(f"{i}. [{status}] {step}\n   -> {answer}")
        return "\n".join(lines)


class Executor:
    def __init__(self, llm: LLMClient, tools: ToolRegistry, max_steps_per_task: int = 8):
        self.llm = llm
        self.tools = tools
        self.max_steps_per_task = max_steps_per_task
        self.planner = Planner(llm)

    def run(self, goal: str, plan: Optional[Plan] = None) -> ExecutionReport:
        plan = plan or self.planner.plan(goal)
        results: List[AgentResult] = []
        prior_summary = ""

        for step in plan.steps:
            instruction = step if not prior_summary else f"{step}\n\nContext from prior steps:\n{prior_summary}"
            agent = Agent(
                llm=self.llm,
                tools=self.tools,
                memory=Memory(),
                max_steps=self.max_steps_per_task,
            )
            result = agent.run(instruction)
            results.append(result)
            if result.final_answer:
                prior_summary += f"- {step}: {result.final_answer}\n"

        return ExecutionReport(plan=plan, results=results)
