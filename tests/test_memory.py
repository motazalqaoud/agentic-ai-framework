import json

from agentic.memory import Memory


def test_short_term_buffer_accumulates():
    m = Memory()
    m.add("user", "goal")
    m.add("assistant", "thinking")
    assert m.as_messages() == [
        {"role": "user", "content": "goal"},
        {"role": "assistant", "content": "thinking"},
    ]


def test_short_term_buffer_is_bounded_and_keeps_first_turn():
    m = Memory(max_turns=3)
    m.add("user", "original goal")
    for i in range(10):
        m.add("assistant", f"turn {i}")

    messages = m.as_messages()
    assert len(messages) == 3
    assert messages[0]["content"] == "original goal"  # goal is never evicted
    assert messages[-1]["content"] == "turn 9"  # most recent turn kept


def test_long_term_memory_roundtrips_without_persistence():
    m = Memory()
    m.remember("favorite_color", "blue")
    assert m.recall("favorite_color") == "blue"
    assert m.recall("missing_key", default="fallback") == "fallback"
    m.forget("favorite_color")
    assert m.recall("favorite_color") is None


def test_long_term_memory_persists_to_disk(tmp_path):
    path = tmp_path / "memory.json"

    m1 = Memory(persist_path=str(path))
    m1.remember("count", 42)
    assert json.loads(path.read_text()) == {"count": 42}

    # A fresh Memory instance pointed at the same file picks up prior state.
    m2 = Memory(persist_path=str(path))
    assert m2.recall("count") == 42
