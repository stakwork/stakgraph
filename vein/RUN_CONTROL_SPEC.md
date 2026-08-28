# Run Control — Spec

Cancel, pause/resume, and durable resume for runs — including the nested
run TREES the lab produces (an evolve run that launches generation runs
that launch candidate runs), where today the only control is killing the
server and orphaning every in-flight log as "stale".

Status: DRAFT. Nothing in this document is implemented.

---

## 1. Problem

A long optimization run (harvey-evolve: hours, tens of dollars, ten+
nested runs) currently offers zero control after launch:

- **No cancel.** A run discovered to be misconfigured at generation 1
  burns the remaining budget anyway.
- **No pause.** Server maintenance means choosing between waiting hours
  or killing everything.
- **No recovery.** A crash or restart loses ALL in-flight progress. The
  event log survives (append-only JSONL, every step's output recorded)
  but nothing can consume it: the run can only be started over from
  scratch, re-spending everything already spent.
- **No tree awareness.** Even with a per-run kill, the lab's runs are
  trees (`harvey-evolve` → `optimizer.run` → `harvey-evolve-gen` →
  `meta/run-workflow` → `harvey-produce-ai`). Each nested launch is an
  independent `runWorkflow` invocation sharing nothing — stopping a
  parent must stop its descendants or it stops nothing.

The runner's own architecture makes this tractable: `runWorkflow` is the
single choke point every run path goes through, every step boundary is
already instrumented (step.start/step.end events), and completed step
outputs are already durably journaled. Run control is mostly about
exposing seams the engine already has.

---

## 2. Design overview

Three features, strictly layered — each rung is independently shippable
and each reuses the previous rung's plumbing:

1. **Cancel** — cooperatively stop a run tree at the next step boundary;
   finalize honestly as `cancelled`.
2. **Pause / resume (in-memory)** — park a run tree at the next step
   boundary of every active branch; release it later. Does not survive a
   restart.
3. **Durable resume (journal replay)** — continue an interrupted run
   after a crash/restart by replaying completed steps' outputs from the
   event log and executing from the first incomplete step.

Pause (2) + durable resume (3) compose into the real operational win:
pause, wait for quiescence, restart the server, resume — a long run
survives a deploy.

### 2.1 One principle: cooperative, at boundaries

The leaves of a run tree cannot be frozen: an LLM call mid-stream, a
`uv`-spawned grader subprocess, an HTTP request. All control is therefore
**cooperative**: the current unit of work completes (and is paid for),
and the run checks for a control signal before starting the next unit.

Checkpoints, from coarse to fine:

| boundary                    | where                                      | rung |
|-----------------------------|--------------------------------------------|------|
| between DAG steps           | `executeFlow.runStep`, before `executeStep`| 1, 2 |
| between loop/foreach iters  | `executeLoop` / `executeForeach`           | 1, 2 |
| between retry attempts      | `executeStep` retry sleep                  | 1, 2 |
| between agent tool calls    | agent step's tool loop                     | 2    |
| between code-step iters     | opt-in via `ctx.control` (e.g. evolve-loop)| 1, 2 |

A run tree paused at these boundaries quiesces within one step / one
tool call per active branch — bounded, predictable, and the money spent
on the in-flight unit is not wasted (its output lands in the journal).

### 2.2 One mechanism: the RunController

```ts
type ControlState = "running" | "pausing" | "paused" | "cancelling";

interface RunController {
  readonly runId: string;
  readonly workflow: string;
  readonly parent?: RunController;      // tree linkage
  readonly children: Set<RunController>;
  state: ControlState;                  // own state; effective state
                                        // inherits the strictest ancestor
  /** The cooperative checkpoint. Resolves immediately when running;
   *  blocks while (effectively) paused; throws CancelledError when
   *  (effectively) cancelling. Every boundary in §2.1 awaits this. */
  checkpoint(): Promise<void>;
  /** True when this run AND all descendants are parked at a boundary
   *  (drives "safe to restart now"). */
  quiesced(): boolean;
  cancel(): void;                       // idempotent; propagates to children
  pause(): void;                        //                    "
  resume(): void;                       //                    "
}
```

- **Registry.** `createVein` holds `controllers: Map<runId, RunController>`
  (superseding today's `activeRuns: Set<string>` — a controller's presence
  IS "in-flight", so the `"running" | "stale"` listing fallback reads this
  map). Registered/unregistered exactly where `trackRun` is called today:
  `launchDetached`, `vein.run`, authoring's `runWorkflow`.
- **Tree linkage.** Nested launches attach to the launching run's
  controller. The parent runId travels the same paths the services bag
  already does: `RunOptions.parentRunId`, set by `meta/run-workflow` and
  the optimizer capability from their calling step's `ctx.runId`.
  Subflows/foreach/loop need nothing — they share the parent's runId and
  therefore its controller.
- **Effective state.** `checkpoint()` walks ancestors: any ancestor
  cancelling → throw; any ancestor pausing/paused → block. Controls
  therefore apply to WHOLE SUBTREES: cancelling the evolve run cancels
  its generations and candidates; pausing a single candidate pauses only
  that candidate.
- **Threading.** The controller rides `RunOptions` into `runWorkflow` and
  is exposed to steps as `ctx.control` (optional, like `ctx.registry` —
  absent outside the runner, e.g. unit tests). Code steps with long
  internal loops (evolve-loop's generations) call
  `await ctx.control?.checkpoint()` per iteration; steps that ignore it
  simply remain coarse-grained (control applies at their step boundary).

---

## 3. Rung 1 — Cancel

**Semantics.** `controller.cancel()` flips the subtree to `cancelling`.
Every branch throws `CancelledError` at its next boundary. `runWorkflow`
catches it as a distinct outcome (NOT the generic error path):

- emit `run.cancelled` (terminal event; `store.tailRun`'s `isTerminal`
  gains this type so SSE tails end),
- `store.finalize` with `status: "cancelled"` — run.json exists, the run
  is never "stale", partial outputs stay inspectable in the log,
- `onRunEnd` teardown fires as on any other exit (browsers/stacks are
  disposed).

**Ordering.** Children are cancelled by the same effective-state walk —
no separate fan-out message. A parent blocked awaiting a child's result
(optimizer.run, meta/run-workflow) sees the child return
`{ status: "cancelled" }` and then hits its own checkpoint. Steps that
launch children treat a cancelled child like a failed one (the existing
error paths), except the enclosing run is itself cancelling and will not
proceed.

**In-flight work.** v1 does not kill the current unit: the LLM call or
grader subprocess completes, its output is journaled, THEN the branch
stops. v1.5 (optional): thread an `AbortSignal` (aborted on cancel only,
never pause) into the agent step's AI-SDK calls and `services.http` for
faster, cheaper teardown of the most expensive leaves. Subprocess
graders keep run-to-completion semantics.

**API/UI.** `POST /workflows/:name/runs/:runId/cancel` (404 unknown, 409
already terminal). UI: a Cancel button in the topbar of an active run's
view; confirm dialog states the subtree consequence ("cancels N nested
runs").

**Status vocabulary.** `RunSummary.status` gains `"cancelled"`. The runs
listing keeps synthesizing `"running" | "stale"` for summary-less dirs
(from the controllers map), and adds `"paused"` (§4).

---

## 4. Rung 2 — Pause / resume (in-memory)

**Semantics.** `controller.pause()` → subtree state `pausing`; each
branch blocks inside `checkpoint()` at its next boundary; when all
branches are parked, `quiesced()` is true and state reads `paused`.
`resume()` releases every parked checkpoint. No events are lost, no
state is discarded — the run's promises simply stop advancing.

**Extra boundary: the agent tool loop.** A 200-step author agent must
park between tool calls, not after its whole session. The agent step
checks `ctx.control` between AI-SDK steps (via its per-step hook, e.g.
`prepareStep`/`onStepFinish`) — the in-flight LLM call finishes, the next
one doesn't start. This is the single highest-value checkpoint in the
lab's workloads.

**Observability.** Emit `run.paused` / `run.resumed` (non-terminal) so
the event log records the gap — otherwise a resumed run's step durations
silently include parked time and poison any timing analysis. The runs
listing reports `paused`; SSE tails stay open (heartbeats already exist).

**Honest limits.** In-memory only: a paused run dies with the process
(becoming exactly the orphan that §5 resurrects — pause then restart
WITHOUT rung 3 is still data loss). Wall-clock-sensitive steps (`wait`,
external polling) resume where they left off; external systems don't
pause with us. Pause does not release per-run resources (a gitsee
browser/stack stays up while parked) — pausing to save money only stops
NEW spend, chiefly LLM calls.

---

## 5. Rung 3 — Durable resume (journal replay)

The event log already contains everything a resume needs: every
completed step's `step.end` carries its `output`, keyed by a
deterministic `path` (`wf/stepId`, `wf/stepId#iter/...` for
foreach/loop bodies, synthetic `wf/stepId#gen` rows for iterative code
steps). Resume = re-run the workflow against that journal.

**Mechanism.**

```
resume(workflow, runId):
  events   = store.getRunEvents(workflow, runId)      # the journal
  journal  = { path → output } for every step.end     # completed units
  input    = the run.start event's recorded input
  re-invoke runWorkflow with { runId, journal }        # SAME runId
```

Execution proceeds normally, except `executeStep` consults the journal
first: a step whose `path` has a journaled output **replays** it (emit
`step.replayed` with the output; resolve the DAG promise; zero cost, no
side effects re-executed) instead of executing. The first path NOT in
the journal executes live, and everything downstream follows. Scope,
templates, and skip/gate logic are reconstructed naturally because they
only ever consume step outputs.

### 5.1 Hard stops are the same case

Nothing above requires a graceful shutdown. The journal is written AS
THE RUN HAPPENS — every event is appended (and awaited) before execution
continues — so after a SIGKILL, an OOM kill, or a power cut, the disk
already holds the completed prefix plus at most one dangling
`step.start` per active branch. Resume replays the prefix and re-executes
the interrupted units. Crash recovery is therefore rung 3 verbatim; a
graceful pause-then-restart merely avoids re-spending the in-flight
step. Three crash-specific hardenings:

- **Torn tail.** A process killed mid-append can leave a truncated final
  JSONL line (step outputs are large; single-write atomicity is not
  guaranteed). The journal reader must skip an unparseable trailing line
  — it belongs to an incomplete unit by definition.
- **Durability policy.** `appendFile` lands in the page cache; a whole-
  SYSTEM crash (not a process kill) can lose recently "written" events.
  Option: fsync on `step.end` / run-level events only (the events worth
  money). Acceptable default: no fsync — losing the cache tail just
  rewinds resume a little further back, which is correct, merely less
  thrifty.
- **At-least-once side effects.** Re-executed steps REDO their external
  effects: an author agent crashed mid-session republishes (an extra
  candidate version — benign, versions are append-only), a re-run
  meta/run-workflow launches a fresh child run. This is the §6
  re-execution contract doing its job; steps whose effects must not
  double need their own idempotency (none in the current lab do).

Optionally, the server can offer auto-resume of summary-less runs on
boot (off by default — a human choosing Resume on a "stale" run is the
right v1 ergonomics).

### 5.2 Resume after failure — retry from the failed step

A run that finalized `status: "error"` (an infra hiccup: a 529 from the
provider that outlived retries, a grader subprocess OOM) is the same
journal with a terminal summary on top. The replay mechanics need
NOTHING new: the failed step has no journaled `step.end`, so plain
resume replays the completed prefix and re-executes exactly the failed
step (fresh retry budget, same onError config) and everything
downstream. A failed foreach iteration re-runs alone — completed
iterations replay by their `#i` paths. What failure-resume actually
adds is lifecycle bookkeeping:

- **The terminal-summary guard relaxes.** Resume refuses only
  SUCCESSFUL runs (nothing to resume — unless `from` below). `error`
  and `cancelled` runs are resumable; on the resumed run's completion,
  `store.finalize` supersedes the old summary (the log keeps the
  original `run.error` + `run.resumed` marker, so history stays
  honest).
- **Tail terminality.** `run.error`/`run.cancelled` are no longer
  unconditionally terminal: a later `run.resumed` in the log reopens
  the stream (historical tails scan ahead; live tails consult the
  controllers map). Without this, the UI would freeze a resumed run's
  event panel at the old failure.
- **`from`: forced invalidation (the "re-run from this step" gesture).**
  Resume accepts an optional step path: that path, its transitive
  dependents, and its iteration children are DROPPED from the journal
  before replay, forcing re-execution even though they completed. This
  covers the step that returned garbage without erroring (a judge that
  produced empty criteria, a fetch that 200'd with an error page).
  With `from`, even a successful run is resumable — "re-grade from
  candeval onward" costs the grades, not the memo.

UI: a failed run's view offers **Resume** (retry the failed step); a
step node's flyout offers **Re-run from here** (resume with
`from: <path>`). Both show what will replay vs re-execute before
confirming — the journal makes that computable upfront. Resume CONTINUES the original run: it
appends to the same JSONL (after a `run.resumed` marker event) and keeps
the runId — critical because artifact directories are keyed by runId
(`artifacts/<runId>/...`): a drafted memo written before the crash is
still on disk at the path the journaled outputs reference. A fresh-runId
"resume" would silently dangle every recorded `outputDir`.

**Nested runs.** A parent step that launched a child run
(meta/run-workflow, optimizer.run) either has a journaled output — child
completed, replayed, child untouched — or doesn't, in which case the
step re-executes and launches a FRESH child run (the interrupted child's
partial log remains on disk as history; v1 does not recursively resume
children — see Non-goals for why not yet).

**Iterative code steps.** A step like `harvey/evolve-loop` is one DAG
step wrapping N generations; all-or-nothing replay would forfeit
completed generations. Such steps already emit per-iteration synthetic
`step.end` events (`…/evolve#3`) carrying full outputs. Contract: the
runner hands the step its own journal slice (`ctx.journal`: path →
output, scoped under the step's path); the step reconstructs completed
iterations from it and continues from the first missing one. Steps that
ignore `ctx.journal` just replay coarse (whole-step or re-run) —
correct, merely less thrifty.

**Agent steps.** v1: an agent step with no journaled output re-runs its
whole session (bounded loss: one step). Future: journal the message
transcript per tool call and resume mid-session — deliberately out of
v1, it drags in provider-state questions the rungs below don't need.

**Validity guards.** Resume refuses when: a live controller exists for
the runId (it's not dead, pause it instead); the workflow's current
content hash differs from the one recorded at `run.start` (the runner
must record it there — replaying outputs into a DIFFERENT DAG is
undefined; power users may override with an explicit flag); or the run
finished successfully with no `from` invalidation (§5.2 — `error` and
`cancelled` runs ARE resumable). Registry drift (a custom step edited
between crash and resume) is allowed but WARNED — steps re-executing
post-resume use current code, same as any new run.

**API/UI.** `POST /workflows/:name/runs/:runId/resume` with optional
`{ from: <stepPath> }` (§5.2). Valid on controller-less runs that are
summary-less ("stale" — crashed), `error`, or `cancelled`; a `success`
run needs `from`. UI: the "stale" badge becomes a Resume affordance.
This retroactively gives "stale" a purpose: it is (part of) the set of
resumable runs.

---

## 6. Step-author contract (additions)

- `ctx.control?: RunController` — await `checkpoint()` inside long
  internal loops; ignore it and your step remains a coarse unit.
- `ctx.journal?: Record<string, unknown>` — on resume, your own prior
  synthetic `step.end` outputs; consume to skip completed iterations.
- Emit per-iteration synthetic `step.end` events with FULL outputs if
  you want thrifty resume (evolve-loop already complies).
- Side-effectful steps must remain safe to re-execute after a crash
  mid-step (the journal only protects COMPLETED units). This is already
  the implicit contract — retries exist — now it's explicit.

## 7. Invariants

- **Measurement discipline untouched.** Control flows through the
  runner, not the workflow surface: authors/candidates cannot pause,
  cancel, or resume anything (no meta/* exposure; the grader and the
  §6-EVOLVE_SPEC firewalls are unaffected).
- **Terminal statuses are honest.** `cancelled` is never conflated with
  `error`; replayed steps are `step.replayed`, never fake `step.end`
  timings; paused gaps are visible in the log.
- **`runWorkflow` stays the single choke point.** All three rungs live
  in the runner + the launch sites; no per-step-type special cases
  beyond the opt-in contract in §6.

## 8. Non-goals (v1)

- Preemptive freezing of in-flight LLM calls/subprocesses, or process
  snapshotting (CRIU-style).
- Recursive resume of interrupted CHILD runs (parent re-launches fresh;
  a child's completed work is recoverable in a later iteration by
  making optimizer.run/meta-run-workflow journal-aware).
- Mid-session agent resume (transcript journaling).
- Cross-machine migration of paused runs.
- Scheduling/quotas ("pause when spend exceeds $X") — trivially layered
  on `pause()` later, out of scope here.

## 9. Rollout

1. RunController + registry + tree linkage; cancel; `cancelled` status;
   UI button. (Subsumes today's `activeRuns`/`trackRun`.)
2. Pause/resume + agent tool-loop checkpoint + paused/quiesced surfacing.
3. Content-hash recording at `run.start` (ship early — resume needs
   history to exist); journal replay; `ctx.journal`; evolve-loop
   iteration resume; failure-resume + `from` invalidation (§5.2); UI
   resume-from-stale, resume-failed, re-run-from-here.

Each rung lands with runner tests (cancel/pause mid-DAG, mid-foreach,
mid-retry; resume replay correctness incl. skip/gate reconstruction) and
one lab-level integration: cancel a nested optimizer run; pause/resume
an agent step between tool calls; resume a killed two-generation evolve
run and verify generation 0 replays for free.
