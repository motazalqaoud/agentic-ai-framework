"""Example: a single Agent solving a two-step arithmetic task with the calculator tool.

Runs fully offline against a MockLLMClient scripted with the exact ReAct turns a real
model would produce -- this demonstrates the loop mechanics without requiring an API key.
Swap in `agentic.AnthropicLLMClient()` (with ANTHROPIC_API_KEY set) for a real run.
"""

from agentic import Agent, MockLLMClient, ToolRegistry
from agentic.tools.builtin import calculator


def main() -> None:
    llm = MockLLMClient(
        scripted_responses=[
            'Thought: I need to compute 12 * 7 first.\n'
            'Action: calculator\n'
            'Action Input: {"expression": "12 * 7"}',
            'Thought: Now add 5 to that result.\n'
            'Action: calculator\n'
            'Action Input: {"expression": "84 + 5"}',
            'Thought: I have the final result.\nFinal Answer: 89',
        ]
    )
    tools = ToolRegistry([calculator])
    agent = Agent(llm=llm, tools=tools, max_steps=5)

    result = agent.run("What is (12 * 7) + 5?")

    print(f"Goal: {result.goal}")
    for i, step in enumerate(result.steps, start=1):
        print(f"\nStep {i}")
        print(f"  Thought: {step.thought}")
        if step.action:
            print(f"  Action: {step.action}({step.action_input}) -> {step.observation}")
    print(f"\nFinal answer: {result.final_answer}  (succeeded={result.succeeded})")


if __name__ == "__main__":
    main()
