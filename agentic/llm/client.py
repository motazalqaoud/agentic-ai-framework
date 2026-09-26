"""LLM client abstraction.

The agent loop only depends on the tiny `LLMClient` interface below, so it can run against:
  - AnthropicLLMClient   a real Claude model, via the Anthropic Messages API
  - MockLLMClient        a deterministic, offline, scripted client used by tests/examples/CI

This keeps the framework runnable and fully testable without an API key.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Optional


class LLMClient(ABC):
    """Minimal interface the agent loop needs from an LLM backend."""

    @abstractmethod
    def complete(self, system_prompt: str, messages: List[dict]) -> str:
        """Return the assistant's next raw text completion given a system prompt and history.

        `messages` is a list of {"role": "user"|"assistant", "content": str} dicts.
        Implementations should return plain text -- the agent loop itself is
        responsible for parsing the ReAct-style Thought/Action/Final Answer structure
        out of that text.
        """
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """A scripted, deterministic client for tests, demos, and CI (no network, no API key).

    Construct it with a list of canned responses; each call to `complete` pops the next
    one off the front. This is enough to exercise the full agent loop -- tool selection,
    observation feeding, multi-step reasoning, final-answer termination -- without ever
    calling a real model.
    """

    def __init__(self, scripted_responses: Optional[List[str]] = None):
        self._responses = list(scripted_responses or [])
        self.calls: List[dict] = []  # recorded for assertions in tests

    def queue(self, response: str) -> None:
        self._responses.append(response)

    def complete(self, system_prompt: str, messages: List[dict]) -> str:
        self.calls.append({"system_prompt": system_prompt, "messages": list(messages)})
        if not self._responses:
            return "Thought: I don't have a scripted response left.\nFinal Answer: (mock client exhausted)"
        return self._responses.pop(0)


class AnthropicLLMClient(LLMClient):
    """Thin wrapper around the Anthropic Messages API.

    Requires the `anthropic` package and an `ANTHROPIC_API_KEY` environment variable
    (or pass `api_key` explicitly). Kept intentionally thin -- swap in any other
    provider's SDK by implementing the same `complete` signature.
    """

    def __init__(self, model: str = "claude-sonnet-5", api_key: Optional[str] = None, max_tokens: int = 1024):
        try:
            import anthropic  # noqa: F401  (imported lazily so the package is optional)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "AnthropicLLMClient requires the 'anthropic' package: pip install anthropic"
            ) from exc

        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system_prompt: str, messages: List[dict]) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=messages,
        )
        return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
