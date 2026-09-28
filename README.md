# agentic-ai-framework

[![tests](https://github.com/motazalqaoud/agentic-ai-framework/actions/workflows/tests.yml/badge.svg)](https://github.com/motazalqaoud/agentic-ai-framework/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-26%20passing-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-motazalqaoud-black)](https://github.com/motazalqaoud)

> A small, dependency-light framework for building tool-using LLM agents: a ReAct
> reasoning loop, a task planner, a tool registry, and bounded memory -- fully testable
> offline, with no API key required to run the test suite or the examples.

## Why this project

Most "agent framework" tutorials wire an LLM directly to a handful of tools and call it
done, with no way to verify the control flow without burning real API calls. This one
separates the **agent loop** (parse a model's reasoning, dispatch a tool call, feed back
the observation, repeat) from the **LLM backend** behind a two-method interface, so the
entire loop -- multi-step reasoning, tool dispatch, error handling, step-budget
exhaustion -- is exercised by a deterministic, scripted `MockLLMClient` in CI. Swapping
in a real model (Claude, or anything else) is a one-line change.

## Architecture

```
Goal ──▶ Planner (optional) ──▶ ordered subtasks ──▶ Executor
                                                         │
                                                         ▼
                                              Agent (ReAct loop) ×N
                                                         │
                                    ┌────────────────────┼────────────────────┐
                                    ▼                    ▼                    ▼
                                 LLMClient           ToolRegistry           Memory
                            (Anthropic / Mock)     (calculator, files,   (short-term buffer
                                                     search, ...)        + persisted KV store)
```

- **Agent** (`agentic/agent.py`) -- the core loop. At each step the LLM emits
  `Thought` / `Action` / `Action Input`, or a `Final Answer`; the agent parses that
  structure with regex (no provider-specific function-calling API required), executes
  the requested tool, and feeds the result back as an `Observation`.
- **Planner** (`agentic/planner.py`) -- for goals with multiple distinct phases, asks the
  LLM to decompose the goal into an ordered list of subtasks up front.
- **Executor** (`agentic/executor.py`) -- runs a `Plan` by giving each subtask its own
  fresh `Agent` and bounded step budget, carrying forward a short summary of what prior
  subtasks produced.
- **ToolRegistry / @tool** (`agentic/tools/base.py`) -- turn any Python function into a
  tool with one decorator; name, parameter types, and description are derived from the
  function's signature and docstring.
- **Memory** (`agentic/memory.py`) -- a bounded short-term conversation buffer (the goal
  is never evicted, oldest turns are) plus an optional JSON-persisted long-term
  key/value store an agent can read and write across runs.
- **LLMClient** (`agentic/llm/client.py`) -- a two-method interface
  (`complete(system_prompt, messages) -> str`). Ships with `AnthropicLLMClient` (a thin
  wrapper over the Anthropic Messages API) and `MockLLMClient` (a scripted, offline
  client used by every test and example in this repo).

## What's different from a typical agent-framework demo

| Common tutorial | This repo |
|---|---|
| Agent loop only runnable against a live API | Fully testable offline via `MockLLMClient` -- no API key needed |
| One flat loop for every goal | Optional `Planner` + `Executor` decompose multi-phase goals into bounded, focused sub-agents |
| Tools defined ad hoc, no consistent shape | `@tool` decorator derives name/params/description from the function itself |
| Unbounded conversation history | `Memory` caps the short-term buffer and always preserves the original goal |
| Provider-specific function-calling API | Plain-text `Thought/Action/Action Input` protocol -- works the same whether or not the backing model has native tool-use |

## Repository structure

```
agentic-ai-framework/
├── agentic/
│   ├── agent.py           ReAct loop: parses Thought/Action/Final Answer, dispatches tools
│   ├── executor.py         Runs a Plan as a sequence of bounded Agent.run calls
│   ├── planner.py          Decomposes a goal into an ordered list of subtasks
│   ├── memory.py           Bounded short-term buffer + persisted long-term key/value store
│   ├── __main__.py         CLI entry point (`agentic "goal"` / `python -m agentic`)
│   ├── tools/
│   │   ├── base.py          Tool dataclass, @tool decorator, ToolRegistry
│   │   └── builtin.py       calculator, file I/O, list_directory, http_get, datetime, search stub
│   └── llm/
│       └── client.py        LLMClient interface, AnthropicLLMClient, MockLLMClient
├── examples/
│   ├── math_agent.py        Single Agent solving a two-step arithmetic task
│   └── research_agent.py    Planner + Executor chaining a multi-phase goal
├── tests/                   26 tests, all offline (MockLLMClient, no network, no API key)
├── docs/architecture.md     How the layers fit together, and why
├── .github/workflows/       CI: tests on Python 3.9-3.12
├── CONTRIBUTING.md
├── requirements.txt
└── pyproject.toml
```

## Setup

```bash
git clone https://github.com/motazalqaoud/agentic-ai-framework
cd agentic-ai-framework
pip install -e ".[dev]"        # framework + pytest, no API key needed
```

To run against a real Claude model instead of the mock client:

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=sk-...
```

## Quickstart — command line

No code required. The package installs an `agentic` command:

```bash
agentic "What is 17 * 23, and is that number prime?"
agentic --plan "Read notes.txt, summarize it, and write the summary to summary.txt"
agentic --demo "What is (12 * 7) + 5?"     # offline, no API key needed
```

`--plan` routes the goal through the Planner and Executor instead of one flat
loop. `--demo` runs against the scripted mock client, so you can see the loop's
shape without a key. `python -m agentic ...` works identically.

## Quickstart — library

```python
from agentic import Agent, ToolRegistry, AnthropicLLMClient
from agentic.tools.builtin import calculator, web_search_stub

tools = ToolRegistry([calculator, web_search_stub])
agent = Agent(llm=AnthropicLLMClient(), tools=tools, max_steps=8)

result = agent.run("What is 17 * 23, and is that number prime?")
print(result.final_answer)
```

Inspect what actually happened:

```python
for step in result.steps:
    print(step.thought, step.action, step.observation, step.malformed)
```

Registering your own tool is one decorator:

```python
from agentic.tools.base import tool

@tool(description="Look up the current price of a stock ticker")
def stock_price(ticker: str) -> str:
    ...  # call a real market-data API here
```

## Running the examples

Both examples run fully offline against a scripted `MockLLMClient` -- no API key needed:

```bash
python examples/math_agent.py       # single Agent, ReAct loop over the calculator tool
python examples/research_agent.py   # Planner + Executor over a two-phase goal
```

## Tests

```bash
pytest tests/ -v
```

26 tests covering the agent loop (tool dispatch, unknown-tool handling, step-budget
exhaustion, final-answer parsing, format-drift recovery), the tool registry and
built-in tools, bounded and persisted memory, plan parsing, and multi-step
execution -- all offline, and all run in CI against Python 3.9 through 3.12.

Architecture and design rationale: [docs/architecture.md](docs/architecture.md).
Contributing: [CONTRIBUTING.md](CONTRIBUTING.md).

## Limitations and disclaimer

**This is a portfolio and reference project, not a production agent runtime.** The
`Thought/Action/Action Input` protocol is parsed with regex against free-text model
output rather than a provider's structured tool-use API, which is simpler and
provider-agnostic but less robust to a model that drifts from the format -- a
production system would likely prefer native function-calling where the backing model
supports it. The bundled tools (`calculator`, `web_search_stub`, file I/O) are examples,
not hardened utilities: the calculator restricts `eval` to arithmetic-only AST nodes,
and `web_search_stub` is an explicit placeholder for a real search API. There is no
sandboxing around tool execution -- don't register a tool with side effects you
wouldn't want an LLM invoking without human review.

## Tech stack

| Tool | Purpose |
|---|---|
| Python stdlib only | Core framework has zero required runtime dependencies |
| `anthropic` (optional) | Real LLM backend via the Claude Messages API |
| `pytest` | Test suite (21 tests, fully offline) |

## About the author

**Motaz Alqaoud, PhD**
- PhD in Biomedical Engineering with focus on medical image analysis and deep learning
- Senior AI/ML Engineer specializing in medical imaging, segmentation models, and clinical AI systems
- GitHub: [@motazalqaoud](https://github.com/motazalqaoud)
- LinkedIn: [linkedin.com/in/motazalqaoud](https://linkedin.com/in/motazalqaoud)

## License

MIT -- see [LICENSE](LICENSE).
