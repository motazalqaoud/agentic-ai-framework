# Architecture

This document explains how the pieces fit together and why the boundaries are
drawn where they are. For usage, see the [README](../README.md).

## The shape of the problem

An LLM agent is a loop around a model that cannot do anything by itself. The
model emits text; something has to read that text, decide it means "call this
tool with these arguments," actually call it, and hand the result back. Nearly
all the interesting failure modes live in that loop, not in the model:

- the model asks for a tool that doesn't exist
- the tool raises an exception
- the model never decides it's finished
- the model drifts out of the expected response format
- the conversation grows until the original goal falls out of context

This framework's design goal is that **every one of those cases is reachable in
a unit test, without a network call.**

## Layers

```
        ┌──────────────────────────────────────────────┐
Goal ──▶│ Planner (optional)                           │
        │   goal → ordered subtasks                    │
        └───────────────────┬──────────────────────────┘
                            │
        ┌───────────────────▼──────────────────────────┐
        │ Executor (optional)                          │
        │   runs each subtask as its own Agent,        │
        │   carrying forward a summary of prior ones   │
        └───────────────────┬──────────────────────────┘
                            │
        ┌───────────────────▼──────────────────────────┐
        │ Agent — the ReAct loop                       │
        │   parse → dispatch → observe → repeat        │
        └───┬──────────────┬──────────────┬────────────┘
            │              │              │
       ┌────▼────┐   ┌─────▼──────┐  ┌────▼─────┐
       │LLMClient│   │ToolRegistry│  │  Memory  │
       └─────────┘   └────────────┘  └──────────┘
```

Each layer below `Agent` is a dependency injected at construction time. Nothing
reaches around a layer: `Agent` never imports `anthropic`, `ToolRegistry` never
knows what a model is, `Memory` never sees a tool.

## Agent: the loop

`Agent.run(goal)` iterates up to `max_steps` times. Each iteration:

1. Ask the `LLMClient` for the next completion, given the system prompt (which
   embeds the rendered tool list) plus the conversation so far from `Memory`.
2. Parse the response with `Agent._parse` into an `AgentStep`.
3. If it contains `Final Answer:` → return a successful `AgentResult`.
4. If it contains `Action:` → run the tool, append `Observation: ...` to memory.
5. If it contains neither → mark the step `malformed`, append an explicit
   correction to memory telling the model the exact format expected.

Step 5 matters more than it looks. An unguided retry after a format drift tends
to drift again; naming the violation is what recovers the run. `AgentStep.malformed`
exposes this to callers so drift can be measured rather than silently absorbed.

Running out of steps is **not** an exception. It returns `AgentResult(succeeded=False,
final_answer=None)`, because "the agent didn't finish" is an ordinary outcome that
callers need to branch on, not a crash.

### Why a text protocol instead of native tool-calling

The loop parses `Thought:` / `Action:` / `Action Input:` out of free text with
regexes, rather than using a provider's structured function-calling API.

The trade-off is deliberate. A provider's tool-calling API is more robust to
format drift — but it also makes the loop untestable without that provider, ties
the framework to one vendor's schema, and does not work at all against models
without native tool support. The text protocol is identical across every backend,
and it is what makes `MockLLMClient` possible.

If you are building on a model with reliable native tool-calling and you do not
need the offline-testability property, using that API instead is a reasonable
change — it is contained entirely within `Agent._parse` and `Agent.run`.

## ToolRegistry: dispatch

A `Tool` is a name, a description, a parameter map, and a callable. The `@tool`
decorator builds one from an ordinary function, reading the parameter names and
types from its signature and the description from its docstring's first line, so
registering a tool is one line and the prompt-facing description cannot drift
out of sync with the implementation.

`ToolRegistry.render_for_prompt()` is the single place tools become text for the
model. Tool errors are caught in `Agent._execute` and returned **as observations**,
never raised — a tool that fails is information the agent should be able to react
to, not a crash that discards the run.

## Memory: bounded context

Short-term memory is a list of `{role, content}` turns, capped at `max_turns`.
When the cap is exceeded, the **oldest turns are dropped but index 0 is always
kept** — index 0 is the goal, and an agent that forgets its goal mid-run fails in
a confusing, hard-to-debug way.

Long-term memory is a flat key/value store, optionally persisted to JSON. There
is deliberately no vector index: in practice agents fail from poor tool selection
and losing the thread of the goal far more often than from lacking semantic recall
over a long history, and a flat inspectable store is much easier to reason about.

## Planner and Executor: multi-phase goals

A single ReAct loop handles "compute X" well. It handles "research X, summarize
it, then write the summary to a file" poorly — the transcript grows, early context
gets evicted, and the agent loses the plot.

`Planner` asks the model for a numbered list of subtasks and parses it. If the
response can't be parsed as a list, it falls back to a single-step plan containing
the original goal — degrading to plain `Agent` behaviour rather than failing.

`Executor` then runs each subtask with a **fresh `Agent` and fresh `Memory`**,
seeded with that subtask plus a short running summary of what earlier subtasks
produced. Each step gets a clean, focused context instead of one long drifting one.

## LLMClient: the seam

The entire interface is one method:

```python
def complete(self, system_prompt: str, messages: list[dict]) -> str: ...
```

`AnthropicLLMClient` wraps the Messages API. `MockLLMClient` pops from a list of
scripted responses and records every call for assertions.

That seam is why the test suite covers tool dispatch, unknown tools, tool
exceptions, step-budget exhaustion, format drift, plan parsing, and multi-step
execution with no API key and no network. Supporting another provider means
implementing one method.

## What this framework deliberately does not do

- **No sandboxing of tool execution.** Tools run with the privileges of the host
  process. Don't register a tool with side effects you wouldn't want an LLM
  invoking unreviewed.
- **No streaming.** `complete` returns a finished string; the loop needs the whole
  response before it can parse an action out of it.
- **No automatic retries or backoff** around the model call. That belongs in the
  `LLMClient` implementation, where the provider's error taxonomy is known.
- **No concurrency.** Subtasks run sequentially, because each one's output feeds
  the next one's context.
