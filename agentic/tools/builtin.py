"""A handful of example tools shipped with the framework.

None of these are meant to be production-hardened (the calculator uses a restricted
`eval`, the web search is a stub) -- they exist to make the framework runnable and
testable out of the box, and to show the shape a real tool takes.
"""

from __future__ import annotations

import ast
import datetime
import operator
import pathlib
from typing import Any, Callable, Dict

from .base import tool

_ALLOWED_BINOPS: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> Any:
    """Evaluate a restricted arithmetic AST -- numbers and +-*/%** only, no names or calls."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval(node.operand)
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool(description="Evaluate a basic arithmetic expression (+ - * / // % **). No variables or function calls.")
def calculator(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    result = _safe_eval(tree.body)
    return str(result)


@tool(description="Read a UTF-8 text file and return its contents (truncated to 8000 chars).")
def read_file(path: str) -> str:
    p = pathlib.Path(path)
    if not p.exists():
        return f"error: no such file: {path}"
    text = p.read_text(errors="replace")
    return text if len(text) <= 8000 else text[:8000] + "\n...[truncated]"


@tool(description="Write text to a file, creating parent directories as needed. Overwrites existing content.")
def write_file(path: str, content: str) -> str:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} chars to {path}"


@tool(description="Return the current UTC date and time in ISO 8601 format.")
def current_datetime() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@tool(description="List the entries in a directory. Returns one name per line, directories marked with a trailing slash.")
def list_directory(path: str) -> str:
    p = pathlib.Path(path)
    if not p.exists():
        return f"error: no such directory: {path}"
    if not p.is_dir():
        return f"error: not a directory: {path}"
    entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
    if not entries:
        return "(empty directory)"
    return "\n".join(f"{e.name}/" if e.is_dir() else e.name for e in entries[:200])


@tool(description="Fetch a URL over HTTP(S) and return the response body as text (truncated to 8000 chars).")
def http_get(url: str) -> str:
    import urllib.error
    import urllib.request

    if not url.lower().startswith(("http://", "https://")):
        return "error: url must start with http:// or https://"
    request = urllib.request.Request(url, headers={"User-Agent": "agentic-ai-framework/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read(1_000_000)
    except urllib.error.HTTPError as exc:
        return f"error: HTTP {exc.code} for {url}"
    except Exception as exc:  # network errors become observations, not crashes
        return f"error fetching {url}: {exc}"
    text = raw.decode("utf-8", errors="replace")
    return text if len(text) <= 8000 else text[:8000] + "\n...[truncated]"


@tool(description="Stub web search -- returns a fixed placeholder. Replace with a real search API in production.")
def web_search_stub(query: str) -> str:
    return (
        f"[web_search_stub] No live search configured. "
        f"To wire up a real search, replace agentic.tools.builtin.web_search_stub "
        f"with a call to your search provider of choice, keeping the same "
        f"(query: str) -> str signature. Query was: {query!r}"
    )
