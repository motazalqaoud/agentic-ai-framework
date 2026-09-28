"""Agent: a ReAct-style (Reason + Act) tool-calling loop.

At each step the agent's LLM produces text in a fixed structure:

    Thought: <reasoning about what to do next>
    Action: <tool name>
    Action Input: <JSON object of arguments>

or, when it's done:

    Thought: <reasoning>
    Final Answer: <the answer to return>

The agent parses that structure, executes the requested tool (if any), appends the
observation to the running transcript, and repeats -- up to `max_steps` -- until a
Final Answer is produced or the step budget runs out.

This text-based protocol (rather than a provider-specific function-calling API) is
deliberate: it keeps the loop provider-agnostic and easy to unit-test with a
MockLLMClient, and it's the same shape whether the backing model supports native
tool-use or not.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .llm.client import LLMClient
from .memory import Memory
from .tools.base import ToolRegistry

_SYSTEM_PROMPT_TEMPLATE = """You are an autonomous agent that solves tasks by reasoning \
step by step and calling tools when needed.

Available tools:
{tools}

Respond using EXACTLY this format at every step:

Thought: <your reasoning about what to do next>
Action: <one tool name from the list above>
Action Input: <a JSON object of arguments for that tool>

Once you have enough information to answer, respond instead with:

Thought: <your reasoning>
Final Answer: <the final answer to the goal>

Never call a tool that is not in the list. Never fabricate an observation -- wait for it
to be provided to you after each Action. One Thought/Action pair per response."""

_ACTION_RE = re.compile(r"Action:\s*(?P<name>\w+)\s*", re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(?P<input>\{.*\})", re.IGNORECASE | re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(?P<answer>.+)", re.IGNORECASE | re.DOTALL)


@dataclass
class AgentStep:
    """One iteration of the loop: what the model said, what was executed, and the result."""

    raw_response: str
    thought: Optional[str] = None
    action: Optional[str] = None
    action_input: Optional[dict] = None
    observation: Optional[str] = None
    final_answer: Optional[str] = None
    malformed: bool = False
    """True when the model's response contained neither an Action nor a Final Answer."""


@dataclass
class AgentResult:
    goal: str
    final_answer: Optional[str]
    steps: List[AgentStep] = field(default_factory=list)
    succeeded: bool = False

    @property
    def step_count(self) -> int:
        return len(self.steps)


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        memory: Optional[Memory] = None,
        max_steps: int = 8,
    ):
        self.llm = llm
        self.tools = tools
        self.memory = memory or Memory()
        self.max_steps = max_steps

    def run(self, goal: str) -> AgentResult:
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(tools=self.tools.render_for_prompt())
        self.memory.add("user", goal)

        steps: List[AgentStep] = []
        for _ in range(self.max_steps):
            raw = self.llm.complete(system_prompt, self.memory.as_messages())
            step = self._parse(raw)
            self.memory.add("assistant", raw)

            if step.final_answer is not None:
                steps.append(step)
                return AgentResult(goal=goal, final_answer=step.final_answer, steps=steps, succeeded=True)

            if step.action is not None:
                step.observation = self._execute(step.action, step.action_input or {})
                self.memory.add("user", f"Observation: {step.observation}")
            else:
                # Neither an Action nor a Final Answer: the model drifted from the
                # protocol. Say so explicitly rather than silently burning steps --
                # an unguided retry usually drifts again.
                step.malformed = True
                self.memory.add(
                    "user",
                    "Your last response did not contain a valid 'Action:' line or a "
                    "'Final Answer:' line. Reply again using exactly one of:\n"
                    "Thought: ...\nAction: <tool name>\nAction Input: {\"key\": \"value\"}\n"
                    "or\nThought: ...\nFinal Answer: ...",
                )

            steps.append(step)

        return AgentResult(goal=goal, final_answer=None, steps=steps, succeeded=False)

    def _execute(self, action: str, action_input: dict) -> str:
        t = self.tools.get(action)
        if t is None:
            return f"error: unknown tool '{action}'. Available tools: {', '.join(self.tools.names())}"
        try:
            return str(t(**action_input))
        except Exception as exc:  # tool errors become observations, not crashes
            return f"error running '{action}': {exc}"

    @staticmethod
    def _parse(raw: str) -> AgentStep:
        step = AgentStep(raw_response=raw)

        thought_match = re.search(r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer):|\Z)", raw, re.DOTALL)
        if thought_match:
            step.thought = thought_match.group(1).strip()

        final_match = _FINAL_ANSWER_RE.search(raw)
        if final_match:
            step.final_answer = final_match.group("answer").strip()
            return step

        action_match = _ACTION_RE.search(raw)
        if action_match:
            step.action = action_match.group("name").strip()

        input_match = _ACTION_INPUT_RE.search(raw)
        if input_match:
            try:
                step.action_input = json.loads(input_match.group("input"))
            except json.JSONDecodeError:
                step.action_input = {}

        return step
