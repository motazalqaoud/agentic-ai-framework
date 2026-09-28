# Contributing

Thanks for taking a look. This is a small project, so the bar is simple: keep it
dependency-light, and keep everything testable offline.

## Development setup

```bash
git clone https://github.com/motazalqaoud/agentic-ai-framework
cd agentic-ai-framework
pip install -e ".[dev]"
pytest tests/ -v
```

No API key is needed for development. The entire test suite runs against
`MockLLMClient`.

## The two rules

**1. The core stays dependency-free.** `agentic/` imports only the standard
library. `anthropic` is an optional extra, imported lazily inside
`AnthropicLLMClient`. If a change needs a new required dependency, it probably
belongs in a user's own tool rather than in the framework.

**2. New behaviour needs an offline test.** If you add a code path, there should
be a test that reaches it with a scripted `MockLLMClient` — no network, no key.
That property is the main thing this project is trying to demonstrate, so a PR
that breaks it will be hard to accept.

## Adding a tool

Tools are plain functions with a decorator:

```python
from agentic.tools.base import tool

@tool(description="One line describing what this does, shown to the model")
def my_tool(some_arg: str) -> str:
    return "result"
```

Name, parameter names and types come from the signature automatically. Return a
string. **Return errors as strings rather than raising** — the agent turns a
returned error into an observation it can react to, while an unhandled exception
is caught generically and loses the detail.

If it's broadly useful, add it to `agentic/tools/builtin.py` and export it from
`agentic/tools/__init__.py`; otherwise keep it in your own code and register it
with `ToolRegistry([...])`.

## Supporting another LLM provider

Implement one method:

```python
from agentic.llm.client import LLMClient

class MyProviderClient(LLMClient):
    def complete(self, system_prompt: str, messages: list[dict]) -> str:
        ...
```

Import the provider SDK lazily inside `__init__` so it stays optional, and mirror
the structure of `AnthropicLLMClient`.

## Style

Follow what's already there: type hints on public functions, a docstring
explaining *why* where the reasoning isn't obvious from the code, and comments
reserved for non-obvious decisions rather than restating the line below them.

## Before opening a PR

```bash
pytest tests/ -v
python examples/math_agent.py
python examples/research_agent.py
```

All three should pass with no network access. CI runs the same thing on Python
3.9 through 3.12.
