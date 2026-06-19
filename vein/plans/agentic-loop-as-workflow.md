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
run becomes visible on the canvas.

**Status: nothing below is built.** This is the design + sequencing doc.

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
anonymous closures that are only `console.log`ged (`agent.ts:571-578`,
`boot-and-exercise.ts:977-984`), never `ctx.emit`ed.

There are **two routes to visibility**, which compose:

- **C2 (loop as real steps):** express boot → drive → observe → assess as real
  `loop`/`subflow`/`foreach` steps. The runner *already* emits all the nested
  events and the existing drill-down renders them — **zero new emit code, zero
  UI change.** Cost: the outer loop is deterministic (less agent autonomy).
- **A2 (tools as steps + per-tool emit):** keep an agentic node but have it
  `emit` a `step.start`/`step.end` per tool call at
  `${ctx.path}/iter${n}/${toolType}`. `ctx.emit`/`ctx.path` already reach every
  leaf step (`runner.ts:478-485`). Cost: a vein-core change (below) + a UI tweak
  for the dynamic sub-canvas.

Recommended: **C2 for the outer skeleton (free, immediate), A2 for the tool
calls inside the inner diagnose-and-fix agent** (so even that node isn't black).

---

## 1. Tools as a first-class entity (the unlock)

### 1a. Inject tools into the core `agent` step (minimal)
`agent` accepts an extra `tools` list it merges with its built-ins. boot-and-
exercise stops forking the loop — it *configures* the core agent. Kills ~600
lines of duplication immediately. This is the cheap first step and is **fully
self-contained in `agent.ts`** (no core change).

### 1b. Tools ARE steps (the "vein" version)
A tool call and a step call are the same shape: named unit, Zod input, output.
Let `agent` take `agentTools: ["browser/click", "gitsee/boot", ...]` — registry
step-types whose `input` schema becomes the LLM tool schema and whose `run` is
the executor. Payoffs:

- each tool is independently **versioned / inspectable / editable / testable**
  via the existing vein UI + `/steps` API
- the agent's tool calls become **nested run events** → visible in the run
  drill-down (the visualization win)
- eval can target a **single tool** (is `assess-ui` a good judge? is `boot`
  reliable?)

**WRINKLE (must own): a leaf step's `ctx` has no registry.** `StepContext`
(`core.ts:44-51`) is `{ runId, path, scope, input, emit, services }` — no way to
look up another step def by type. `subflow`/`foreach`/`loop` only reach other
steps because the runner handles them specially (`SELF_RESOLVING_STEPS`,
`runner.ts:301`). Two ways to give `agent` registry access:

- **(b, preferred) promote `agent` to a runner-aware container step**: add it to
  `SELF_RESOLVING_STEPS`, handle it in `dispatchStep`, and pass it the registry +
  a child-exec helper (the same machinery `executeSubflow` uses). Mirrors the
  existing pattern; keeps `StepContext` clean.
- (a) add `registry` to `StepContext`. Simpler diff, but widens the contract for
  every step.

Either way **`agent.run` must start consuming `ctx`** (today `async run(cfg)`
ignores it — `agent.ts:413`).

### 1c. Toolkits (ergonomics, later)
Mount `browser/*` as a bundle instead of listing 9 names. Only after 1b.

---

## 2. Stateful resources become services

`BrowserSession` (`boot-and-exercise.ts:418-595`) and the pm2/staklink/compose
lifecycle (`736-809`) are infrastructure, not agent logic. Move them into
`LabServices` so the tool-steps in §1b are thin wrappers (`ctx.services.browser.
click(ref)`), and so they're swappable for eval (real Playwright ↔ cassette;
staklink ↔ inline boot; swap the vision model).

- **`browser` service** — per-page session.
- **`stack`/`runner` service** — boot + teardown of the app + backing services.
- **`vision`/`judge` service** — `assessScreenshot` (`615-659`); swappable model;
  evaluable against a labeled (screenshot+logs → verdict) dataset.

**WRINKLE (must own): services are singletons, this state is per-run.**
`createLabVein` builds the services bag **once**, shared across all runs
(`createLabVein.ts:70`). A live browser page / booted stack is per-run mutable
state — concurrent runs would collide on one shared page. So `browser`/`stack`
must be **session factories keyed by runId** (`services.browser.session(runId)`),
not a singleton holding one page. Each session owns its own lifecycle.

**WRINKLE (must own): teardown guarantee is lost on decomposition.** Today one
`try/finally` (`boot-and-exercise.ts:1032-1049`) guarantees pm2 + compose +
snapshot-diff container cleanup even on error — critical, because a leaked
`supabase start` stack is ~12 containers. vein has **no workflow-level
"finally"** (only `when`/`onError` per step). Decomposing into steps risks
leaking docker stacks on kill/error. Options to evaluate:

- a dedicated teardown step that runs on both success and error (needs a
  guaranteed-run / `finally`-style semantic vein doesn't have yet — possible
  small core add), **or**
- the `stack` service registers its sessions and `createVein` tears down live
  sessions on `run.end`/`run.error` (service-owned lifecycle, no workflow
  change), **or**
- keep boot+teardown inside ONE step (a `stack`-lifecycle step that brings up,
  yields, and tears down) and only decompose the drive/observe/fix loop.

The `cleanup.ts` script (the manual rescue for killed runs) stays regardless.

---

## 3. The loop as a visible workflow (C2)

Re-express `gitsee-setup-and-run.yaml`'s opaque `run` node as a real loop:

```
loop (until $current.working || maxIters):
  body: subflow "gitsee-qa-iteration"
    boot            (stack service tool-step)
    drive+snapshot  (browser tool-steps)
    observe+assess  (browser + vision tool-steps → STRUCTURED outputs)
    diagnose-and-fix (agent step, narrow tools: fs + bash + read_logs)
    -> returns { working, ... }
```

Every phase becomes a canvas node, separately inspectable and **evaluable**.

**WRINKLE (must own): `loop` body is a single `Step`** (`loop.ts:23`). The
multi-step iteration must be wrapped in a `subflow` body; `until` references
`$current` (the subflow's output, e.g. `$current.working`). This is supported
today — just a structural requirement, not a code change.

**Tension to weigh honestly:** C2 trades agent autonomy for structure. The magic
today may partly BE the agent fluidly interleaving bash + browser in an order we
didn't anticipate. Mitigation: keep the inner `diagnose-and-fix` agent's tool
budget generous (fs + bash + read_logs + re-snapshot) so it still investigates
freely within each iteration; only the *outer* rhythm is fixed.

---

## 4. Observations/verdicts as first-class artifacts

`browser_observe`, `assess_ui`, `read_logs` emit *signals*. As tool-steps with
**structured** outputs they become independently scorable, replayable, and
cacheable — and you can build an eval dataset for the vision judge itself.
This is what makes "continuous self-improvement" measurable instead of vibes.

---

## 5. Harness vs policy split (the eval payoff)

Draw a hard line so an optimize loop can sweep policy while the harness stays
constant:

- **Harness (fixed):** services (browser/stack/vision) + tool-steps + teardown.
- **Policy (tunable, lives in `params`):** system prompt, which tools are
  mounted, model, loop strategy, maxIters.

This mirrors how `gitsee-optimize` already tunes the explorer `system` prompt —
impossible for boot-and-exercise today because policy is welded into the TS file.

---

## Suggested sequencing

1. **§1a** — inject tools into `core/agent` (self-contained; ends the fork;
   ~600 LOC of duplication deleted). Lowest risk, highest immediate payoff.
2. **§2 browser + stack services** as per-run factories; decide the teardown
   strategy (the riskiest open question — pick before decomposing).
3. **§3 (C2)** — re-express the loop as `loop`+`subflow`; visualization via the
   runner's existing events, **zero UI change**. This delivers "see each
   iteration in the UI."
4. **§1b** — promote `agent` to a registry-aware container step; tools = steps;
   per-tool emit for the inner agent node.
5. **§4 / §5** — structured signals + harness/policy split → a
   `gitsee-setup-optimize` loop.
6. **Tier-2 UI**: dynamic agent sub-canvas built from events (only if the events
   panel proves insufficient).

## Open questions / risks

- **Teardown semantics** (§2) — the one genuinely unsolved design choice. A
  vein-level guaranteed-`finally` step would be broadly useful but is a core add.
- **Does C2 regress quality** vs the free-form loop? Needs an A/B once §3 lands.
- **A2 sub-canvas** — building a canvas from dynamic events (no static child
  flow) is new UI territory; defer until Tier-1 (events panel) is proven thin.
- **Concurrency** — per-run service sessions assume runs may overlap (the
  optimize loop runs many evals); verify the factory keying before relying on it.
