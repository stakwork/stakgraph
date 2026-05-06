# Context Management Next Steps

## Current State

The repo agent now supports `contextMode: "compiled"`.

Compiled mode keeps two tiers of session memory:

- Short-term memory: the last 4 text-only user/assistant messages from the session.
- Medium-term memory: a structured `.context.json` sidecar with `summary`, `goals`, `decisions`, `importantRefs`, `checked`, `openQuestions`, `nextSteps`, and `warnings`.

The context compiler updates after each successful compiled-mode turn. It writes:

- `.context.json`: latest compiled session state.
- `.context.timeline.jsonl`: before/after snapshots, diff, summary token usage, changed summary flag, and a compact preview of the new turn.

The sessions API exposes `context_state` and `context_timeline`, and the benchmark UI shows a small `ctx` button on turns that have timeline entries.

The current comparison script runs full vs compiled sessions and prints token deltas plus answer snippets. It does not yet score memory preservation or consume `context_timeline`.

## Recommendation

Do not optimize efficiency first. The next priority is to prove that compiled memory preserves the right information and makes failures visible. Efficiency tuning should come after we can detect memory loss.

## Todo List

- [ ] Add measurable context quality checks to the comparison flow.
- [ ] Add context observability that explains what changed and why it matters.
- [ ] Use findings to tune the context schema only where failures show gaps.
- [ ] Tune efficiency after quality is stable.
- [ ] Consider long-term or retrieval memory only after session-level compiled memory is reliable.

## 1. Action Plan: Make Context Quality Measurable

### Goal

Measure whether compiled mode preserves the working facts needed to continue a session, not just whether it uses fewer tokens.

### Current Gap

`scripts/compare-context-modes.py` compares token usage and answer snippets, but it does not answer:

- Did compiled mode remember the user's constraints?
- Did it keep key files, functions, endpoints, commands, and decisions?
- Did it avoid repeating things already checked?
- Did it preserve open questions or known blockers?
- Did it drop anything that later mattered?

### Proposed Work

- [ ] Extend the comparison script to fetch each compiled session via `GET /sessions/:id` after the run.
- [ ] Save `context_state` and `context_timeline` into the existing results output.
- [ ] Add a deterministic quality report for each compiled run:
  - context entries per turn
  - total compiled-context summarization tokens
  - final compiled state token estimate or character count
  - counts for goals, decisions, refs, checked items, open questions, next steps, warnings
  - added/removed counts per turn from `diff`
- [ ] Add scenario-specific expected facts for each benchmark prompt set.
  - Example: auth prompt should preserve challenge/verify terms and key auth files/functions if discovered.
  - Example: payment prompt should preserve key payment flow functions and already checked files.
  - Example: admin endpoint prompt should preserve endpoint refs and reconcile details.
- [ ] Add a simple pass/fail memory retention check using exact/substring match against expected facts.
- [ ] Keep LLM scoring optional, but make deterministic checks the default so the benchmark is cheap and repeatable.
- [ ] Print a concise report that separates:
  - answer quality
  - memory quality
  - token cost
  - context summarization overhead

### Deliverable

An updated comparison script that produces a clear table like:

```text
MEMORY QUALITY
Turn  Mode      Expected Kept  Missing  Repeated Checks  Refs  Decisions  Warnings
1     compiled  6/7            1        0                8     2          0
2     compiled  9/10           1        0                13    4          1
3     compiled  12/12          0        0                20    5          1
```

### Success Criteria

- We can tell when compiled mode saves tokens but loses important context.
- We can compare full vs compiled quality without manually reading every answer.
- Each benchmark run leaves enough artifacts to debug what the context compiler kept or dropped.

## 2. Action Plan: Improve Observability

### Goal

Make context updates easy to inspect from the benchmark UI and API so memory problems are obvious during a session review.

### Current Gap

The UI now exposes context snapshots through a per-turn `ctx` modal, but the modal is mostly a readout of the final `after` state. It does not yet make the most important debugging questions obvious:

- What changed on this turn?
- What was added vs removed by field?
- Did the summary change?
- How much did the context update cost?
- What raw turn content caused the update?
- Which retained refs came from this turn versus earlier turns?

### Proposed Work

- [ ] Improve the `ctx` modal to show a compact "Changed this turn" section first.
  - Added goals, decisions, refs, checked items, open questions, next steps, warnings.
  - Removed items grouped separately.
  - `changedSummary` shown as a small summary badge.
- [ ] Add a token row in the modal:
  - agent turn tokens from `step_meta`
  - context summary tokens from `context_timeline[].usage`
  - combined turn cost if available
- [ ] Show `newMessagesPreview` in a collapsed details block so reviewers can inspect what the compiler saw.
- [ ] Add an API/debug curl path to inspect timeline-only data quickly.
  - Existing `GET /sessions/:id` already contains the data, so this can start as a documented jq/python one-liner rather than a new endpoint.
- [ ] Add a session-level context summary chip in the session header.
  - Number of context timeline entries.
  - Total context summarization tokens.
  - Latest context update timestamp.
- [ ] Make empty states explicit.
  - If no `ctx` buttons appear, show whether the session was not run with `contextMode: "compiled"` or simply has no timeline file.

### Deliverable

A session review experience where a developer can answer, in under a minute:

- Did compiled context run for this session?
- Which turns changed memory?
- What did each turn add or remove?
- How much did memory maintenance cost?
- Is the compiled state missing an important fact?

### Success Criteria

- Reviewing `context_timeline` no longer requires opening raw JSONL files.
- The UI makes memory loss and over-retention visible.
- The same data remains accessible from curl for quick debugging.

## Later: Efficiency Tuning

Only tune these after quality checks are in place:

- Recent raw message count: currently 4.
- Tool output preview size in compiler prompt: currently 2000 chars per tool result.
- Generic compact message part limit: currently 4000 chars.
- Context update frequency: currently every successful compiled-mode turn.
- Schema size and field redundancy.

Efficiency work should optimize against measured quality, not just lower tokens.