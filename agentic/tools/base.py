"""Tool registration: a decorator + registry, no framework-specific base classes required.

A tool is just a Python callable with a name, a one-line description, and a JSON-schema-ish
parameter spec, auto-derived from the function's signature and docstring where possible.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class Tool:
    """A single callable tool the agent can invoke."""

    name: str
    description: str
    func: Callable[..., Any]
    parameters: Dict[str, str] = field(default_factory=dict)

    def __call__(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)

    def signature_hint(self) -> str:
        """Human/LLM-readable one-liner: name(param: type, ...) -- description"""
        params = ", ".join(f"{k}: {v}" for k, v in self.parameters.items())
        return f"{self.name}({params}) -- {self.description}"


class ToolRegistry:
    """Holds the tools available to one Agent instance."""

    def __init__(self, tools: Optional[list[Tool]] = None):
        self._tools: Dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, t: Tool) -> None:
        if t.name in self._tools:
            raise ValueError(f"Tool '{t.name}' is already registered")
        self._tools[t.name] = t

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def render_for_prompt(self) -> str:
        """Render the registry as a block of text suitable for an LLM system prompt."""
        if not self._tools:
            return "(no tools registered)"
        return "\n".join(f"- {t.signature_hint()}" for t in self._tools.values())


def _annotation_name(annotation: Any) -> str:
    """Render a parameter annotation as a short type name, whether it's a real type
    object or a string (functions using `from __future__ import annotations`, or a
    plain string forward reference, both end up with `annotation` as a str)."""
    if annotation is inspect.Parameter.empty:
        return "Any"
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__name__", str(annotation))


def tool(name: Optional[str] = None, description: Optional[str] = None):
    """Decorator: turn a function into a Tool and return the Tool (not the raw function).

    Usage:
        @tool(description="Add two numbers")
        def add(a: float, b: float) -> float:
            return a + b
    """

    def decorator(func: Callable[..., Any]) -> Tool:
        tool_name = name or func.__name__
        doc = description or (inspect.getdoc(func) or "").splitlines()[0] if (description or inspect.getdoc(func)) else ""
        sig = inspect.signature(func)
        params = {p.name: _annotation_name(p.annotation) for p in sig.parameters.values()}
        return Tool(name=tool_name, description=doc or "(no description)", func=func, parameters=params)

    return decorator
