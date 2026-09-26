"""agentic — a small, dependency-light framework for building tool-using LLM agents.

Public API:
    Agent           the ReAct-style reasoning loop
    Tool, tool      register a Python function as a callable tool
    ToolRegistry     lookup / dispatch for registered tools
    Memory          short-term buffer + persisted long-term key/value store
    Planner         breaks a high-level goal into an ordered list of subtasks
    LLMClient        abstract client interface
    MockLLMClient    deterministic, offline client used by tests and examples
    AnthropicLLMClient  thin wrapper around the Anthropic Messages API
"""

from .agent import Agent, AgentResult, AgentStep
from .memory import Memory
from .planner import Planner, Plan
from .tools.base import Tool, tool, ToolRegistry
from .llm.client import LLMClient, MockLLMClient, AnthropicLLMClient

__all__ = [
    "Agent",
    "AgentResult",
    "AgentStep",
    "Memory",
    "Planner",
    "Plan",
    "Tool",
    "tool",
    "ToolRegistry",
    "LLMClient",
    "MockLLMClient",
    "AnthropicLLMClient",
]

__version__ = "0.1.0"
