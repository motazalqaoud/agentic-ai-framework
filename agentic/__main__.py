"""Command-line entry point: run an agent against a goal without writing any code.

    python -m agentic "What is 17 * 23?"
    python -m agentic --plan "Look up X, then write a summary to notes.txt"
    python -m agentic --demo "What is (12 * 7) + 5?"

By default this uses the Anthropic Messages API and requires ANTHROPIC_API_KEY.
Pass --demo to run against the scripted offline MockLLMClient instead, which
needs no key and no network (useful for seeing the loop's shape).
"""

from __future__ import annotations

import argparse
import sys

from .agent import Agent
from .executor import Executor
from .memory import Memory
from .tools.base import ToolRegistry
from .tools.builtin import (
    calculator,
    current_datetime,
    http_get,
    read_file,
    web_search_stub,
    write_file,
)

_DEMO_SCRIPT = [
    'Thought: I should compute this with the calculator.\n'
    'Action: calculator\n'
    'Action Input: {"expression": "(12 * 7) + 5"}',
    "Thought: I have the answer.\nFinal Answer: 89",
]


def build_default_registry() -> ToolRegistry:
    """The tool set the CLI exposes by default."""
    return ToolRegistry(
        [calculator, read_file, write_file, current_datetime, http_get, web_search_stub]
    )


def _make_client(demo: bool, model: str):
    if demo:
        from .llm.client import MockLLMClient

        return MockLLMClient(scripted_responses=list(_DEMO_SCRIPT))

    from .llm.client import AnthropicLLMClient

    return AnthropicLLMClient(model=model)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m agentic",
        description="Run a tool-using agent against a goal.",
    )
    parser.add_argument("goal", help="What you want the agent to accomplish")
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Decompose the goal into subtasks first (Planner + Executor) instead of one flat loop",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use the scripted offline mock client (no API key, no network)",
    )
    parser.add_argument("--model", default="claude-sonnet-5", help="Anthropic model id")
    parser.add_argument("--max-steps", type=int, default=8, help="Step budget per agent run")
    parser.add_argument("--quiet", action="store_true", help="Print only the final answer")
    args = parser.parse_args(argv)

    try:
        llm = _make_client(args.demo, args.model)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    tools = build_default_registry()

    if args.plan:
        executor = Executor(llm=llm, tools=tools, max_steps_per_task=args.max_steps)
        report = executor.run(args.goal)
        if args.quiet:
            last = report.results[-1].final_answer if report.results else None
            print(last or "(no answer)")
        else:
            print(report.summary())
        return 0 if report.succeeded else 1

    agent = Agent(llm=llm, tools=tools, memory=Memory(), max_steps=args.max_steps)
    result = agent.run(args.goal)

    if not args.quiet:
        for i, step in enumerate(result.steps, start=1):
            print(f"\n--- step {i} ---")
            if step.thought:
                print(f"thought: {step.thought}")
            if step.action:
                print(f"action:  {step.action}({step.action_input})")
                print(f"result:  {step.observation}")
        print()

    print(result.final_answer or "(no final answer -- step budget exhausted)")
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
