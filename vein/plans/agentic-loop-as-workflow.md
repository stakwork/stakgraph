# Agentic loop as workflow — tools, services, and visibility

Context: `mcp/src/lab/gitsee/steps/boot-and-exercise.ts` (~1050 LOC) is a great
result but a structural dead end for eval + self-improvement. It **forks** the
core `agent` step's entire `ToolLoopAgent` loop (and copy-pastes `textEdit`,
`buildPreamble`, `resolveInCwd`, cache-control wiring, the forced-final-answer
salvage) purely to add custom tools — because the core `agent` step has **no
extension point for tools** (its toolset is hardcoded; `toolFilter` only
*subsets* it — `agent.ts:541`). The file fuses five concerns:

1. the agent loop (a fork of `core/agent`)
2. a stateful browser driver (`BrowserSession`)
3. a boot/runner lifecycle (pm2/staklink/compose + snapshot-diff teardown)
4. a vision judge (`assessScreenshot`)
5. the QA *policy* (the giant system prompt + tool descriptions)

For eval we want to tune **#5 alone** while holding 1–4 fixed — and we want to
**see each iteration in the UI**. Today the whole loop is one opaque node.

The plan: make **tools** and the **loop** first-class vein constructs so the
monolith dissolves into small, versioned, individually-evaluable pieces, and the
run becomes visible on the canvas. Every design question below has a **firm
decision** — there are no open wrinkles. Two small, generic vein-core changes are
required (see "Core changes" at the end); everything else is additive.

## Implementation status

- **DONE — vein-core foundation** (420 tests green, tsc clean): `StepContext.registry`
  (populated in `dispatchStep`), the generic `services.onRunEnd(runId)` `finally`
  hook in `runWorkflow`, and the `agent` step's `agentTools` (registry step-types
  as LLM tools via the exported, unit-tested `buildRegistryTools`, with nested
  `step.start`/`step.end` emit). `agent.run` now consumes `ctx`.
- **DONE — visualization MVP (layer 1)**: `boot-and-exercise.ts` now wraps every
  tool's `execute` to emit a nested run event (`<stepPath>/NNN-<tool>`), so each
  iteration's tool calls show in the UI events panel / drill-down. Uses only
  pre-existing `ctx.emit`/`ctx.path` → runs against the current copied vein dep,
  **no `refresh-vein` required**. This satisfies the headline ask ("see each
  iteration in the UI") without touching the loop's behavior.
- **TODO — layer 2**: §2 services (browser/stack/vision as per-run factories +
  `LabServices.onRunEnd`), §1b tool-steps, §3 the C2 loop workflow. Build after
  the layer-1 MVP is confirmed in the UI (it validates the event-path approach
  end-to-end before the bigger decomposition rests on it).

The rest of this doc is the design + sequencing for layer 2.

---

## The core insight: visualization == nested-path run events

Everything the UI draws (node status, run drill-down, I/O flyout) is **derived
from `events.jsonl`**, where each event carries a hierarchical `path`:

- a normal step emits `step.start`/`step.end` at `${basePath}/${id}`
  (`runner.ts:353,376`)
- `foreach`/`loop` iterations emit at `${path}#${i}` (`runner.ts:522-651`)
- `subflow` runs children inline under the same runId with nested paths
  (`wf/subflowId/childId`); the UI strips the prefix and re-keys to a sub-canvas

**No path events → nothing to render.** The agent run is invisible because the
entire `ToolLoopAgent` loop runs *inside one step's `run()`*: the only events
emitted are the single `step.start`/`step.end` for that node. Tool calls are
anonymous closures only `console.log`ged (`agent.ts:571-578`,
`boot-and-exercise.ts:977-984`), never `ctx.emit`ed.

**DECISION — two complementary routes, both adopted:**

- **C2 (loop as real steps):** express boot → drive → observe → assess as real
  `loop`/`subflow` steps. The runner *already* emits the nested events and the
  existing drill-down renders them — **zero new emit code, zero UI change.** This
  is the committed surface for per-iteration visibility (it directly satisfies
  "see each iteration in the UI").
- **A2 (tools as steps + per-tool emit):** the inner diagnose-and-fix agent emits
  a `step.start`/`step.end` per tool call at `${ctx.path}/i${n}/${toolType}`, so
  even that node's tool calls appear in the **events panel**. `ctx.emit`/`ctx.path`
  already reach every leaf step (`runner.ts:478-485`).

**DECISION — the bespoke "agent node → dynamic sub-canvas" is OUT OF SCOPE.** C2
delivers loop-level drill-down for free via the existing subflow/foreach
machinery, and A2's tool calls render in the events panel keyed by nested path.
That fully satisfies the requirement; building a canvas from dynamic (non-static-
workflow) events is unnecessary and is explicitly cut, not deferred.

---

## 1. Tools as a first-class entity (the unlock)

### 1a. Inject tools into the core `agent` step (minimal, ships first)
`agent` accepts an extra `tools` list it merges with its built-ins. boot-and-
exercise stops forking the loop — it *configures* the core agent. Kills ~600
lines of duplication. **Fully self-contained in `agent.ts`** (no core change).

### 1b. Tools ARE steps (the "vein" version)
A tool call and a step call are the same shape: named unit, Zod input, output.
`agent` takes `agentTools: ["browser/click", "gitsee/boot", ...]` — registry
step-types whose `input` schema becomes the LLM tool schema and whose `run` is
the executor. Payoffs: each tool is independently **versioned / inspectable /
editable / testable** via the existing `/steps` API + UI; tool calls become
**nested run events**; eval can target a **single tool**.

**DECISION — give `agent` registry access via `StepContext.registry`, keep the
loop in `agent.ts`.** Add an optional `registry: StepRegistry` to `StepContext`
(`core.ts:44-51`), populated by the runner in `dispatchStep` (`runner.ts:478`).
The `agent` step (now consuming `ctx` — today `run(cfg)` ignores it,
`agent.ts:413`) looks up each `agentTools` type, builds the LLM tool from
`def.input` + `def.description`, and on a tool call executes
`def.run(input, childCtx)` where `childCtx = { ...ctx, path:
\`${ctx.path}/i${n}/${type}\` }`, emitting `step.start`/`step.end` around it.

*Rejected:* promoting `agent` to a runner-handled container step (à la
`subflow`). It would fracture the AI-SDK `ToolLoopAgent` loop into `runner.ts`.
Keeping the loop in the step file with read-only `registry` access is simpler and
keeps the runner generic. *Rejected:* a `ctx.runStep` closure over the internal
`executeStep` — tools don't need retry/onError, so the extra contract isn't worth
it. (Re-derive `executeStep` reuse later only if a tool ever needs retry.)

### 1c. Toolkits (ergonomics)
Mount `browser/*` as a bundle instead of listing names. **DECISION:** deferred to
after 1b lands and proves the per-name list is annoying — pure sugar, no new
capability.

---

## 2. Stateful resources become services

`BrowserSession` (`boot-and-exercise.ts:418-595`) and the pm2/staklink/compose
lifecycle (`736-809`) are infrastructure, not agent logic. They move into
`LabServices` so the §1b tool-steps are thin wrappers, and so they're swappable
for eval (real Playwright ↔ cassette; staklink ↔ inline boot; swap the vision
model).

**DECISION — three services, with per-run sessioning:**

- **`browser`** — `services.browser.session(runId)` returns/creates a per-run
  page session (the current `BrowserSession`, lifted verbatim).
- **`stack`** — `services.stack.session(runId, workspacePath)` owns boot
  (compose up + staklink/pm2 + wait-for-port) and teardown (the snapshot-diff
  container cleanup).
- **`vision`** — stateless `assessScreenshot(...)` (`615-659`); a swappable
  judge, evaluable against a labeled (screenshot+logs → verdict) dataset.

**DECISION — sessions are factories keyed by `runId`, NOT singletons.**
`createLabVein` builds the services bag once, shared across all runs
(`createLabVein.ts:70`); a live page / booted stack is per-run mutable state, and
the optimize loop runs many evals concurrently. Each service holds a
`Map<runId, session>`; tool-steps get `runId` from `ctx.runId`. This makes
concurrent runs collision-free by construction.

**DECISION — teardown is guaranteed by a generic `services.onRunEnd(runId)`
lifecycle hook.** The runner calls `await (services as any)?.onRunEnd?.(runId)` in
a `finally` wrapping `executeFlow` in `runWorkflow` (`runner.ts:110-155`), so it
fires on **both** success and error, for **every** run path (detached,
`optimizer.run`, tests) — `runWorkflow` is the single choke point, and subflows
reuse the parent runId so it fires once per top-level run. `LabServices.onRunEnd`
disposes that run's browser + stack sessions. This **exactly preserves today's
`try/finally` guarantee** (`boot-and-exercise.ts:1032-1049`) while staying generic
(the runner never names "stack"/"browser").

*Rejected:* a new vein-level guaranteed-`finally` *step*. It's a bigger, more
general core change than needed; `onRunEnd` is ~3 lines and sufficient. (A
finally-step can be proposed independently later if other workflows want it.)

**Hard-kill caveat (unchanged from today):** `onRunEnd` runs in-process, so a
`SIGKILL` of the host still skips it — identical to today's `finally`.
`mcp/src/lab/gitsee/cleanup.ts` remains the rescue for that case.

---

## 3. The loop as a visible workflow (C2)

Re-express `gitsee-setup-and-run.yaml`'s opaque `run` node as a real loop:

```
loop  until: "{{ $current.working }}"   maxIterations: {{ params.maxIters }}
  body: subflow "gitsee-qa-iteration"
    boot             (stack tool-step)
    drive+snapshot   (browser tool-steps)
    observe+assess   (browser + vision tool-steps → STRUCTURED outputs)
    diagnose-and-fix (agent step; tools: fs + bash + read_logs + re-snapshot)
    -> returns { working, ... }
```

Every phase is a canvas node, separately inspectable and **evaluable**.

**DECISION — the iteration is a `subflow` used as the `loop` body.** vein's `loop`
body is a single `Step` (`loop.ts:23`), so the multi-step iteration is its own
workflow (`gitsee-qa-iteration`), and `until` reads `$current.working` (the
subflow's output). This uses only existing vein features — no code change, just
the committed structure.

**DECISION — keep the inner agent autonomous within each iteration.** C2 fixes
only the *outer* rhythm; `diagnose-and-fix` keeps a generous tool budget (fs +
bash + read_logs + re-snapshot) so it still investigates freely. An **A/B vs the
free-form monolith is a required validation gate** (see Validation) before
retiring `boot-and-exercise.ts`.

---

## 4. Observations/verdicts as first-class artifacts

**DECISION:** `observe`, `assess-ui`, `read_logs` are tool-steps returning
**structured** outputs (not strings). That makes each signal independently
scorable, replayable, and cacheable, and lets us build an eval dataset for the
vision judge itself — turning "self-improvement" into something measured, not
vibed.

---

## 5. Harness vs policy split (the eval payoff)

**DECISION — a hard line, enforced by where things live:**

- **Harness (fixed):** the `browser`/`stack`/`vision` services + the tool-steps +
  `onRunEnd` teardown.
- **Policy (tunable, in `params`):** the agent `system` prompt, which
  `agentTools` are mounted, `model`, and `maxIters`.

A `gitsee-setup-optimize` loop then sweeps policy with the harness held constant
— mirroring how `gitsee-optimize` already tunes the explorer `system` prompt
(impossible for boot-and-exercise today because policy is welded into the TS file).

---

## Core changes (the complete list — both small + generic)

1. **`StepContext.registry?: StepRegistry`** (`core.ts`), populated by the runner
   in `dispatchStep` (`runner.ts:478`). Optional, ignored by steps that don't use
   it (exactly like `services`). Enables §1b.
2. **`services.onRunEnd?(runId)` hook** invoked in a `finally` around
   `executeFlow` in `runWorkflow` (`runner.ts:110-155`). Optional + generic.
   Enables §2 teardown.

Everything else is additive: new lab services, new tool-steps, two new lab
workflows (`gitsee-qa-iteration`, the rewritten `gitsee-setup-and-run`), and the
deletion of `boot-and-exercise.ts` once §3's A/B passes.

---

## Sequencing

1. **§1a** — inject tools into `core/agent` (self-contained; ends the fork; ~600
   LOC deleted). Lowest risk, immediate payoff.
2. **Core change #1** (`StepContext.registry`) + **§1b** — tools = steps; per-tool
   emit.
3. **§2** — `browser`/`stack`/`vision` services as per-run factories +
   **core change #2** (`onRunEnd`) for guaranteed teardown.
4. **§3 (C2)** — `gitsee-qa-iteration` + rewritten `gitsee-setup-and-run`;
   visualization lands here via the runner's existing events (no UI work).
5. **§4 / §5** — structured signals + harness/policy split → `gitsee-setup-optimize`.

## Validation (the gates, not open risks)

- **C2 quality A/B** (§3): run the decomposed loop vs the current monolith on the
  `hive` + `heroku-node` workspaces; require parity (working-rate + iterations)
  before deleting `boot-and-exercise.ts`. If the fixed outer rhythm regresses,
  widen the inner agent's per-iteration budget rather than reverting.
- **Concurrency** (§2): a 2-run overlap test asserting browser/stack sessions
  don't collide and `onRunEnd` disposes the correct run's sessions.
- **Teardown** (§2): kill-path test — force a mid-run error and assert compose +
  spawned containers are gone (parity with today's `finally`).
- **Live agent** (§1b): one real chat/optimize turn confirming the agent drives
  the registry tool-steps and the events panel shows per-iteration tool calls.
