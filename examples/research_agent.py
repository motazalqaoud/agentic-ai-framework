"""Example: Planner + Executor chaining multiple subtasks toward one goal.

A goal like "research X, then write a summary to a file" doesn't fit neatly into a
single ReAct loop -- it has two distinct phases with different tools in play. The
Planner breaks it into ordered subtasks; the Executor runs each as its own bounded
Agent, carrying forward a short summary of what each prior step produced.

Runs fully offline against a MockLLMClient. Swap in `agentic.AnthropicLLMClient()`
(with ANTHROPIC_API_KEY set) for a real run -- the planner's output format is plain
text, not scripted, so it works unchanged against a real model.
"""

from agentic import ToolRegistry
from agentic.executor import Executor
from agentic.llm.client import MockLLMClient
from agentic.tools.builtin import web_search_stub, write_file


def main() -> None:
    llm = MockLLMClient(
        scripted_responses=[
            # 1: the planner's decomposition
            "1. Search for what an agentic AI framework is\n"
            "2. Write a one-paragraph summary of the findings to notes.txt",
            # 2: subtask 1 -- the research step
            'Thought: I should search for a definition.\n'
            'Action: web_search_stub\n'
            'Action Input: {"query": "what is an agentic AI framework"}',
            "Thought: I have enough to summarize.\n"
            "Final Answer: An agentic AI framework provides the reasoning loop, tool "
            "registry, and memory an LLM needs to autonomously pursue a goal.",
            # 3: subtask 2 -- the write step
            'Thought: I will write the summary to a file.\n'
            'Action: write_file\n'
            'Action Input: {"path": "/tmp/agentic_example_notes.txt", '
            '"content": "An agentic AI framework provides the reasoning loop, tool registry, '
            'and memory an LLM needs to autonomously pursue a goal."}',
            "Thought: Done.\nFinal Answer: Summary written to /tmp/agentic_example_notes.txt",
        ]
    )
    tools = ToolRegistry([web_search_stub, write_file])
    executor = Executor(llm=llm, tools=tools)

    report = executor.run("Research what an agentic AI framework is and write a summary to a file")
    print(report.summary())
    print(f"\nOverall succeeded: {report.succeeded}")


if __name__ == "__main__":
    main()
