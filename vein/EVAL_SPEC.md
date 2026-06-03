# Evals & Self-Improving Workflows — Spec

How vein supports **evals** and **self-improving loops**: define a goal, run a
workflow, score its output against an expected gold standard, and iterate on
the workflow until the output is good.

Builds on `SPEC.md` (the engine). The guiding idea:

> **Everything is a workflow + its `params`.** The target pipeline, the
> scorer, and the eval harness are all workflows; the dataset (expected gold),
> the rubric, and the tunable prompts are all `params` — which vein already
> stores, versions, and edits in the UI. There are **no** new
> `Dataset`/`Rubric`/`Experiment` resource types. The eval is just a
> **measurement substrate**: `run target → collect → score → a number`.
> Whoever consumes that number (a human, the chat agent, or a thin loop) is
> the "optimizer".

---

## 1. Status (what's built)

| Piece | State |
| --- | --- |
| Keyed `paramOverrides` engine primitive (§4) | **done** (`runner.ts`, on `/run` + `run()`) |
| Params **visible + editable + persistable** in the UI (§3) | **done** (`ParamsFlyout`; Publish writes params into the new version) |
| Bootstrap prompt paramified | **done** (`bootstrap-then-process` `params`) |
| `concepts/collect-for-eval` — run output = scoreable concept set | **done** (final step of `bootstrap-then-process`) |
| `eval/score` (generic recall-based step) + `concepts-eval-score` workflow (§5) | **done** (step in `mcp/src/lab/eval`, workflow in `concepts`) |
| `concepts-eval` — bootstrap-only eval loop (§6) | **done** (`mcp/src/lab/concepts`) |
| First real run (recall 0.8) — surfaced the overfitting/insight problem | **done** |
| Background-job model: detached runs + reattach (§8) | **done** (`launchDetached` + `FileRunStore.tailEvents` + `…/runs/:runId/stream`) |
| `eval/reflect` — the "propose" step (param-changer, aggregate-aware) | **done** (`eval/reflect` + `concepts-eval-reflect` workflow) |
| Optimize loop `eval→keep→reflect` as a detached job (§7, §11.4) | **done, single-repo** (`eval/optimize` + `concepts-optimize`; runs `concepts-eval` per generation via `paramOverrides`) |
| Agent *builds* evals; background job *runs* them (§7) | partial (loop runs autonomously; agent-side `run_eval` tool + system prompt still todo) |
| Multi-repo dataset + train/val split (§4, §11.2) — the real overfit fix | todo (reflect already takes a multi-repo `results[]`; needs the `examples[]` dataset + batch eval) |
| Reproducible/isolated runs: snapshots + namespacing (§9) | todo |

Manual eval works **today**: open `concepts-eval`, set the `expected` gold in
the Params flyout, Run with a repo, read the score; edit the bootstrap prompt
(`bootstrap-then-process` params or a `paramOverrides`), re-run.

---

## 2. The model

```
                 paramOverrides / new version
                          │  (the candidate)
                          ▼
   repo ─► TARGET workflow ─► collect ─► { markdown, concepts }
                                              │
                expected gold (a param) ──►  SCORE ─► { score, missing, … }
```

- **Target** — the pipeline being improved (e.g. `bootstrap-then-process`).
- **Collect** — a final step (`collect-for-eval`) turns graph state into the
  scoreable run output. Without it the run records the wrong thing.
- **Score** — `concepts-eval-score`, an LLM judge that matches produced vs expected.
- **Candidate** — what changes between runs (see §4). Param tweak or a whole
  new workflow version.
- **Harness** — `concepts-eval` chains the above into one run that returns a
  number (§6).

Everything above is a workflow or a `param`. No new resource types.

---

## 3. Eval config = `params` (no new resources)

The rubric, the expected gold, and the dataset all live as `params` on
ordinary workflows. **Changing a param = a new workflow version** — editing in
the `ParamsFlyout` marks the workflow dirty; **Publish** writes the params into
the next version's YAML (`POST /workflows/:name` already accepts
`{ steps, params }`). So "setting up an eval in the UI" needs no new storage,
API, or resource — just the editable Params flyout (done).

- **Rubric** → the `rubric` param of `concepts-eval-score`.
- **Expected gold** → an `expected` param (one repo) or an `examples` array
  param (many), authored in the same markdown shape `collect-for-eval` emits.

**Naming convention (generic vs experiment).** The eval *mechanism* is
domain-agnostic and reusable; the *config* is per-experiment. So:

- **`eval/*` steps** — the generic primitives, no domain baked in:
  `eval/score` (match produced↔expected by a `rubric`), `eval/reflect`
  (propose a better prompt from the aggregate), `eval/optimize` (the loop).
  Seeded once (`mcp/src/lab/eval`), reused by every experiment.
- **`<experiment>-…` workflows** — wire those steps with the experiment's
  rubric / task / dataset, and live with the experiment. The concepts
  experiment ships `concepts-eval` (harness), `concepts-eval-score` (rubric),
  `concepts-eval-reflect` (task+guidance), `concepts-optimize` (the wired loop)
  in `mcp/src/lab/concepts/workflows`. A future experiment `foo` adds
  `foo-eval`, `foo-eval-score`, … reusing the same `eval/*` steps.

---

## 4. The candidate: a workflow *version* (not just params)

What an optimizer evolves is **a version of the target workflow**. Two kinds,
on a spectrum:

1. **Param tuning** (cheap, mechanical) — change prompt/threshold values for
   known knobs. Reaches knobs at any depth via the keyed-override primitive:

   **`paramOverrides`** (`runner.ts`) — opt-in, name-addressed, threaded
   recursively so it reaches a knob inside a nested subflow (e.g.
   `concepts/decide` two levels down). `params` (flat) only hits the entry
   flow; `paramOverrides[workflowName]` hits that flow at any depth.
   ```jsonc
   POST /workflows/concepts-eval/run
   { "input": { "owner": "x", "repo": "y" },
     "paramOverrides": { "bootstrap-then-process": { "bootstrapPrompt": "…" } } }
   ```

2. **Structural evolution** (the interesting one) — change the *shape* of the
   workflow: add steps, new data sources, new tools. E.g. instead of only
   reading code, bootstrap could fetch the last 200 PR titles, or inspect the
   org's other repos, to recover historical context. A workflow can't author
   itself — **this is inherently the agent's job** (§7). Each structural
   variant is just a **new workflow version**, which versioning already
   captures and the eval already scores.

Param tuning is the degenerate case of structural evolution where only the
params changed. Both reduce to "author a candidate version → eval it → keep
the winner."

---

## 5. The scorer (recall-based)

`eval/score` is a self-contained LLM-judge step (dynamic `import("ai")`,
structured output). The judge **matches** each expected capability to a
produced concept semantically and reports `matched / missing / spurious`; the
step computes a **continuous, recall-weighted** score (F-beta, β=2) from the
counts — deterministic and informative as a gradient.

```jsonc
// output
{ "score": 0.62, "recall": 0.6, "precision": 0.75,
  "matched":  [{ "expected": "Real-time Chat", "produced": "Messaging" }],
  "missing":  ["Payment Processing", "Notifications"],
  "spurious": ["Redis Caching"],
  "reason": "…", "insight": "…", "markdown": "…" }
```

The matching **criteria** (what counts as a match / spurious / naming
standards) live in the `rubric` param; the **math** lives in the step. The
`missing` list is the actionable signal — it's exactly what the next candidate
must learn to recover.

---

## 6. The harness: `concepts-eval`

One workflow that runs the whole eval as a single scoreable run:

```
reset    → clear the repo's concepts (so bootstrap re-runs fresh)
produce  → bootstrap-then-process, lookbackDays: 0  (BOOTSTRAP ONLY, no PRs)
score    → concepts-eval-score(actual = produce.markdown, expected = params.expected)
```

`lookbackDays: 0` makes the checkpoint "now" so no PRs/commits are processed —
isolating the bootstrap agent. `reset` is needed because the `is-new-repo`
gate only bootstraps an empty repo. Output: `{ score, missing, … }`. The
expected gold is the `expected` param.

---

## 7. Who runs the loop: the agent *builds*, a background job *runs*

The chat agent should **build** evals, not **be** the loop. Two roles:

- **Agent = builder** (interactive, short-lived). Designs the experiment:
  authors target-workflow variants, the dataset (`examples` param), the rubric,
  the reflect prompt + its generalization rules, the train/val split. Uses tools
  it largely already has (`create_workflow`, `create_step`, `edit_step`,
  params). It can kick a run off and inspect results, but it is **not** the
  thing sitting in the hot loop.
- **Background optimize run = executor** (autonomous, runs for hours). Runs
  `eval → reflect → select` over many candidates × many repos, persisting each
  generation, decoupled from any chat session. `reflect` (the LLM proposing the
  next param edit) runs **inside this job** — and per §4 it must see the
  **multi-repo aggregate**, not one repo, or it overfits (the wrong-insight
  problem).

The orchestrator is thin, generic engine code that sequences workflow runs in a
loop and tracks best-on-val; the *intelligence* lives in the artifacts the agent
authored (rubric, reflect prompt, dataset). Every piece it runs is a workflow.

- **Param tuning** → fully autonomous in the background job ("go and go and go").
- **Structural evolution** (authoring new steps/data sources) → stays
  agent-interactive for now (a workflow can't write code); later, a headless
  authoring-agent job.

Still todo: a `run_eval` tool (clean score for the agent) + a system-prompt
section; and the background-job model (§8) the executor depends on.

---

## 8. Background jobs: detached runs + reattach (engine prereq) — **done**

A multi-hour optimize loop can't be a request-scoped run. Previously a run was
launched by `POST /run` and streamed over that one request: it executes
**server-side** and **persists every event to an append-only JSONL file**
(durable, queryable after the fact) — but closing the connection, a server
restart, or proxy timeouts made long runs fragile. The fix is an evolution of
the existing run model, not a rewrite. **Shipped** as a clean cutover (the old
streaming `POST /run` is gone):

1. **Launch detached** — `POST /run` starts `runWorkflow` **without awaiting it
   in the request** (`launchDetached` in `createVein.ts`) and returns
   `{ runId }` (202) immediately. The run's liveness is decoupled from any
   connection; there is no `onEvent` on the launch (all viewing is read back
   from the persisted log).
2. **Reattach (no client polling)** — `GET /workflows/:name/runs/:runId/stream`
   (SSE) **tails the events file** via `FileRunStore.tailEvents`: replay offset
   0 → EOF (history), then follow appends, stop + send final `done` on the
   terminal event (`run.end`/`run.error`). The web UI's `api.runWorkflow`
   chains the two (launch → `streamRun`) so callers keep the same interface.

Why file-tail: the append-only log **is** the ordered source of truth, so the
history→live join is naturally **race-free** (read to EOF, follow from EOF) — no
sequence numbers, no dedupe. One code path serves completed *and* in-flight runs,
it survives restarts, and it's process-agnostic. The only "polling" is the
server noticing appends (`fs.watch` / a tiny interval) — invisible to the client.
(An in-memory pub/sub is lower-latency but single-process and loses the live tail
on restart; not worth it for vein's filesystem-first model.)

**Restart-resume:** in-flight *execution* is in-memory, so a crash mid-run loses
the remaining work; the persisted log up to the crash survives. True resume
(re-pick an unfinished run) is a later add — for now a crashed optimize job is
restarted from its last persisted generation.

---

## 9. Reproducibility & isolation (todo)

For tight/automated loops the eval runs must be cheap and non-colliding:

- **Snapshots** — pin a dataset example's change-set so runs are deterministic
  and offline (`fetch-changes`/`fetch-content` read an injected `snapshot`
  instead of hitting GitHub). Today they only have a lower bound (the
  checkpoint); a snapshot freezes the upper bound too.
- **Namespacing + teardown** — concepts are keyed by `owner/repo` in Neo4j, so
  parallel candidates collide. A per-run `namespace` (suffix the repoId) +
  teardown isolates them. `concepts-eval` uses a coarse `reset` today, which
  is enough for sequential single-repo tuning.

---

## 10. The open thesis

The point of all this: **code-only bootstrap probably can't recover a repo's
~10 core user-facing capabilities** — that needs historical context (PR
titles, releases, org activity). The recall rubric will score code-only
bootstrap low and name the `missing` capabilities, which is the signal that
should push the agent to add history-aware steps. The eval turns "you need
historical context" from an opinion into a **measurable, falsifiable** claim —
and then drives building the workflow that fixes it.

---

## 11. What's next

1. **First real run** of `concepts-eval` on a small repo — **done** (recall 0.8;
   surfaced the overfitting/insight problem that motivates §4 + §7).
2. **Multi-repo eval** — `examples` array param + a Level-2 batch eval, with a
   train/val split (the structural fix for overfitting, §4/§7).
3. **Background-job model (§8)** — detached launch + file-tail reattach.
   **done.** The prerequisite for any long optimize run.
4. **The optimize loop** — `eval → keep best → reflect (generalizing) → repeat`,
   built as the `eval/optimize` step inside the detached `concepts-optimize`
   workflow (a background job, §8). **done for a single repo**; `eval/reflect`
   already takes a multi-repo `results[]`, so generalizing across a train set
   is unlocked once the dataset (item 2) lands. Still todo: best-on-**val** (vs
   train), auto-**promote** the winner (write the param default + publish a new
   version), and a `run_eval` tool + system-prompt section so the agent can
   build/inspect it.
5. **Reproducibility (§9)** — snapshots + namespacing for cheap, parallel runs.
