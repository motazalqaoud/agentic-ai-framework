"""Memory: a short-term conversation buffer plus an optionally-persisted long-term store.

Deliberately simple -- no vector index, no embeddings. Most agent tasks fail from bad
tool selection or losing track of the goal, not from lacking semantic recall over a huge
history. This gives the agent a bounded, inspectable working memory, and a flat
key-value long-term store an agent can read/write across runs.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional


class Memory:
    def __init__(self, max_turns: int = 20, persist_path: Optional[str] = None):
        self.max_turns = max_turns
        self._buffer: List[dict] = []
        self._persist_path = pathlib.Path(persist_path) if persist_path else None
        self._long_term: Dict[str, Any] = {}
        if self._persist_path and self._persist_path.exists():
            self._long_term = json.loads(self._persist_path.read_text())

    # -- short-term (this run's conversation) --------------------------------------

    def add(self, role: str, content: str) -> None:
        self._buffer.append({"role": role, "content": content})
        # keep the buffer bounded: drop oldest turns first, always keep the system goal (index 0)
        if len(self._buffer) > self.max_turns:
            self._buffer = [self._buffer[0]] + self._buffer[-(self.max_turns - 1):]

    def as_messages(self) -> List[dict]:
        return list(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    # -- long-term (persisted across runs, if persist_path was given) -------------

    def remember(self, key: str, value: Any) -> None:
        self._long_term[key] = value
        self._flush()

    def recall(self, key: str, default: Any = None) -> Any:
        return self._long_term.get(key, default)

    def forget(self, key: str) -> None:
        self._long_term.pop(key, None)
        self._flush()

    def _flush(self) -> None:
        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(json.dumps(self._long_term, indent=2))
