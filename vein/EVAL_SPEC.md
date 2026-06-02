# Evals & Self-Improving Experiments — Spec

A design spec for turning vein into a platform for **evals** and
**self-improving loops**: pin a goal, run a workflow against pinned
examples, score the output against an expected gold standard, and let an
optimizer evolve the workflow's tunable knobs (`params`) until the output
is good.

This builds on `SPEC.md` (the engine) and reuses its existing primitives
wherever possible. It is **not** a CLI — every operation is a first-class
API endpoint, visible in the UI, and (later) drivable by the AI agent.

> GOAL: AS SIMPLE AS POSSIBLE. **Everything is a workflow + its `params`.**
> The dataset (expected gold), the rubric, and the candidate knobs are all
> just `params` — which vein already stores, versions, and (now) edits in the
> UI. We do **not** add new `Dataset`/`Rubric`/`Experiment` resource types.
> The only new pieces are a couple of small **steps** (a collector and a
> scorer) and one engine primitive (keyed param overrides).

---

## 0. Status (what's implemented)

| Piece | State |
| --- | --- |
| Keyed `paramOverrides` engine primitive (§3) | **done** (`runner.ts`, exposed on `/run` + `run()`) |
| Params **visible + editable + persistable** in the UI (§4, §8) | **done** (`ParamsFlyout`; Publish writes params into the new version) |
| Bootstrap prompt paramified (§6.1) | **done** (`bootstrap-then-process` `params`) |
| `concepts/collect-for-eval` — run output = scoreable concept set (§5) | **done** (final step of `bootstrap-then-process`) |
| `eval/score` step + `eval-score` workflow (rubric in `params`) (§5) | **done** (`mcp/src/lab/eval`) |
| Snapshot capture/injection, namespacing + teardown (§6) | todo |
| `eval/reflect` + orchestrator + optimize stream (§5) | todo |
| Optimize UI: generations chart / diff / drill (§8) | todo |
| Agent tools (§9) | todo |

---

## 1. Motivation

Two pipelines we want to make "perfect" by iterating on prompts, not code:

1. **Concepts** (`mcp/src/lab/concepts`) — walk a repo's history and build
   a knowledge graph of ~10 user-facing "Concepts". The goal is a good
   **top-N concept set** per repo. Tunable surfaces: the bootstrap
   exploration prompt and the per-change `decide` prompt.
2. **Services agent** (`mcp/src/gitsee/agent`, services mode) — explore a
   repo and emit a working `pm2.config.js` + `docker-compose.yml`. (A full
   GEPA-style harness for this already existed at
   `mcp/src/gitsee/agent/eval/` — this spec generalizes that pattern into
   the platform. Not the v1 target, but the design must cover it.)

Both reduce to the same shape:

| Piece | Concepts | Services agent |
| --- | --- | --- |
| **Candidate** (what evolves) | bootstrap + decide prompts (`params`) | explorer + final_answer prompts |
| **Dataset** | repos pinned at a rev + gold concept list | repos + gold pm2/compose files |
| **Run** | run the target workflow with candidate params | run the agent with candidate prompts |
| **Score** | judge: semantic recall/precision vs gold | judge: would the config work? |
| **Reflect** | LLM proposes better params | LLM proposes better prompts |

The framework is **target-agnostic**: it does not care how many prompts a
target has or where they live, as long as they are exposed as `params`.

---

## 2. Design decisions (settled)

| # | Decision | Rationale |
| --- | --- | --- |
| D1 | **Thin orchestrator + everything-is-a-workflow.** `score`/`reflect` are vein workflows; the GEPA loop is thin engine code streamed via SSE. | Ports the proven harness; every sub-step is an inspectable run; no fragile new looping primitives. |
| D2 | **No new resource types. Eval config = workflow `params`.** The rubric, the expected gold, and the dataset (`examples`) are `params` on an eval/experiment workflow. | `params` are already workspace-stored, versioned, and (now) UI-editable. **Changing a param IS a new workflow version** — exactly the persistence/versioning we'd otherwise rebuild. |
| D3 | **Candidate scope = `params` only** (prompts/thresholds/selectors), as a **multi-scope union** across `{workflow, key}`. | Matches the existing "experiment surface"; smallest search space; covers multi-prompt targets. |
| D4 | **Nested/scoped param override** is added to the engine (keyed by workflow name). | The tunable prompts live in subflows; per-run `params` don't propagate today. General; opt-in; back-compat. |
| D5 | **Pin-by-snapshot** datasets. | Deterministic, offline, fast, cheap evals — no GitHub during optimization. |
| D6 | **Holistic reflection** for credit assignment (start). | Simplest; reflector sees all prompts + insights and may change any. Component sub-rubrics / staged optimization can come later. |
| D7 | **Human-driven first, agent-driven later.** | The agent layer is thin tool wrappers over the same endpoints; build them tool-friendly from day one. |
| D8 | **Params are editable + persistable in the UI** (the `ParamsFlyout`); edits publish a new workflow version. | This is what makes "set up an eval in the UI" real without any new resource/storage/API. |

---

## 3. New engine primitive: keyed param overrides

Today (`runner.ts`): `executeFlow` seeds `params = { ...workflow.params,
...paramsOverride }`, and the subflow call **does not** pass overrides
down — subflows reset to their own `params`. This is deliberate isolation,
but it means a candidate cannot reach a prompt that lives in a subflow
(e.g. `concepts/decide` inside `process-change`, two levels below the
entry workflow).

**Add an opt-in, name-addressed override that travels the whole tree:**

```ts
runWorkflow(flow, input, registry, {
  params,                                  // entry-flow only (unchanged)
  paramOverrides: {                        // NEW — keyed by workflow name
    "bootstrap-then-process": { bootstrapSystem, sizing },
    "process-change":         { systemPrompt, guidelines },
  },
})
```

- In `executeFlow`, the scope becomes:
  `{ ...workflow.params, ...(paramOverrides[workflow.name] ?? {}), ...(isEntry ? params : {}) }`
- The subflow execution (`runner.ts` ~673) threads `paramOverrides`
  recursively so it reaches every nested level.
- **Back-compat:** flat `params` keeps meaning "entry flow"; `paramOverrides`
  is additive and opt-in. Precedence: step `.default()` < flow `params`
  default < `paramOverrides[name]` < entry `params`.

This single primitive is also what makes **multi-scope candidates** free:
a candidate is just a `paramOverrides` map spanning several workflows.

---

## 4. No new resources — eval config is `params`

The earlier draft proposed `datasets/`, `rubrics/`, and `experiments/`
resource types (storage + API + UI). We **dropped that**: vein's `params`
already are the dynamic-state mechanism we need — workspace-stored, versioned,
and now UI-editable. So the rubric, the expected gold, and the dataset all
live as `params` on ordinary workflows.

> **Changing a param = a new workflow version.** Editing a param in the
> `ParamsFlyout` marks the workflow dirty; **Publish** writes the params into
> the next version's YAML (the existing `POST /workflows/:name` already
> accepts `{ steps, params }`). No new storage, API, or resource type.

### 4.1 Rubric → a `param` of the scorer workflow

The rubric (the fixed measuring stick — WHAT to weigh) is the `rubric` param
of `eval-score`. Editable in the UI; the optimizer never mutates it.

```yaml
# eval-score.yaml (seeded)  —  input: { actual, expected }
steps:
  - id: judge
    type: eval/score
    config: { actual: "{{ input.actual }}", expected: "{{ input.expected }}", rubric: "{{ params.rubric }}" }
params:
  rubric: |-
    Score produced Concepts vs the expected gold set. Match semantically, not
    by string. Weigh RECALL / PRECISION / NAMING / DESCRIPTION. …
```

The `eval/score` **step** is pure mechanism: it appends the strict
`SCORE / REASON / INSIGHT` contract and parses the verdict
(`{ score, reason, insight, markdown }`). The `INSIGHT` feeds reflection.

### 4.2 Dataset (expected gold) → a `param`

The "dataset" is just `params` too. For a single repo, the gold is an
`expected` param; for several, an `examples` array param (each
`{ input, expected }`). Authored/edited in the `ParamsFlyout` (JSON for the
array), versioned on Publish.

```yaml
# an "experiment" workflow's params
params:
  rubric: |- …                       # the measuring stick
  examples:
    - input:    { owner: stakwork, repo: hive, rev: a1b2c3, until: "2025-09-01" }
      expected: |
        # Concepts: stakwork/hive
        ## Real-time Chat
        Users can …
        ## Task Management
        …
```

`expected` is authored in the **same markdown shape** `collect-for-eval`
emits, so scoring is apples-to-apples (semantic, so PR/commit counts don't
matter). **Pin-by-snapshot (D5)** still applies for reproducibility — the
pinned change-set is carried in the example's `input` (or a `snapshot` field)
and injected into `fetch-changes`/`fetch-content`.

### 4.3 Experiment → a workflow whose params hold the dataset + rubric

An "experiment" is an ordinary workflow that orchestrates one eval pass:
`foreach` over `params.examples` → run the **target** (with the candidate's
`paramOverrides`) → `collect-for-eval` → `eval-score` against that example's
`expected`. The **candidate** (what the optimizer evolves) is the target's
own `params`, addressed via keyed `paramOverrides` (§3) — *not* stored on the
experiment; it's proposed per generation by reflection and promoted into the
target's `params` defaults on a win.

### 4.4 Generations (optimize history)

The one place a small new artifact is still useful: the per-generation record
the optimizer writes (candidate params tried, per-example scores, aggregate,
reflection). Persist these under the experiment workflow's run storage
(`runs/<ts>/gen-N/…`) — reusing the existing run store, not a new resource.

```jsonc
// gen-3: { candidate, results:[{exampleId, score, reason, insight}], aggregate }
```

---

## 5. The optimize loop

### 5.1 Two small workflows (everything-is-a-workflow, D1)

- **`eval-score`** (built) — input `{ actual, expected }`, `rubric` in
  `params`. The `eval/score` step is a self-contained LLM-judge (like vein's
  core `llm` step, via dynamic `import("ai")`) that enforces the
  `SCORE / REASON / INSIGHT` contract and returns
  `{ score, reason, insight, markdown }`.
- **`eval-reflect`** (todo) — input `{ candidate, results, history }`; one
  `llm` step; output a new candidate (`paramOverrides`-shaped JSON over the
  tunable keys). The reflection prompt is this workflow's own `params`, so
  it is itself editable. Holistic (D6): it sees **all** prompts + insights
  and may change any of them; given prior attempts so it doesn't repeat.

### 5.2 The orchestrator (thin engine code, SSE-streamed)

```
optimize(experiment):
  candidate = seed (from target's param defaults across all tunable scopes)
  best = candidate; bestScore = -1; history = []
  for gen in 0 .. budget.maxGenerations:
    results = []
    for example in dataset.examples:
      ns = repoId + "#" + <expRun>/<gen>/<exampleId>          # isolation (§6)
      run TARGET( { ...example.input, snapshot: example.snapshot, namespace: ns },
                  paramOverrides = candidateToOverrides(candidate) )   # a real run
      output = run experiment.output.extractor(ns)            # a real run
      teardown(ns)                                            # delete graph nodes
      s = run eval/score({ output, expected: example.expected, rubric })   # a real run
      results.push({ exampleId, output, score: s })
    aggregate = mean(results.score)
    persist gen-N; emit SSE generation event; track best
    if aggregate >= budget.perfectScore: break
    candidate = run eval/reflect({ candidate, results, history })  # a real run
    history.push(...)
  persist summary
```

Every `run …` is a normal `runWorkflow` call → persisted, inspectable in
the existing run flyout/canvas. Only the outer loop (budget, best-tracking,
reflection memory, early-stop) is engine code.

---

## 6. Stateful-target concerns (Concepts)

The Concepts target writes to Neo4j keyed by `repoId = owner/repo` rather
than returning a value. Three mechanical requirements:

- **Output extraction (W4).** An `outputExtractor` step
  (`concepts/collect-for-eval`) reads `getAllConcepts(namespacedRepoId)`
  and returns `{ concepts }`, so the scorer always sees a uniform
  `{ output, expected }`.
- **Isolation (W3).** Add an optional `namespace` to the concepts workflow
  input; steps compute
  `repoId = namespace ? \`${owner}/${repo}#${namespace}\` : \`${owner}/${repo}\``.
  Each candidate/generation/example run uses a distinct namespace so runs
  never collide; **teardown** deletes nodes for that namespaced repoId
  after scoring. Feasible: every storage method already takes a `repo` arg.
- **Snapshot injection (D5).** `fetch-changes`/`fetch-content` read
  `input.snapshot` when present (no GitHub).

### 6.1 Paramification (done)

The bootstrap prompt was hardcoded in `bootstrap.ts` (`buildBootstrapPrompt`,
the `systemOverride`). It is now lifted into `bootstrap-then-process`'s
`params` (`bootstrapSystem` + `bootstrapPrompt`, with `{slot}` placeholders
filled at runtime) and consumed by `concepts/bootstrap-explore` — mirroring
`concepts/decide`. The candidate surface is now a clean union of `params`
across the two workflows: `paramOverrides["bootstrap-then-process"]` (bootstrap)
+ `paramOverrides["process-change"]` (decide).

---

## 7. API surface

There are **no new resource endpoints** — datasets/rubrics/experiments are
workflows + params, so they use the existing surface:

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/workflows/:name/flow` | read a workflow incl. its `params` (done) |
| POST | `/workflows/:name` | publish a new version with `{ steps, params }` — persists edited rubric/expected/examples (done) |
| POST | `/workflows/:name/run` | run a target/experiment; accepts `params` **and** keyed `paramOverrides` (done) |

The only genuinely new endpoints are for the optimize loop (todo):

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/workflows/:name/optimize` | run the experiment workflow across `params.examples`, evolving the target's params; **SSE stream** of generation events |
| POST | `/workflows/:name/promote` | write the winning candidate into the target's `params` defaults + publish a new version (reuses versioning) |

---

## 8. UI surface

- **Setting up an eval = editing workflow params** (done). The `ParamsFlyout`
  (topbar **Params** button) edits a workflow's `rubric` / `expected` /
  `examples`; **Publish** persists them as a new version. No separate
  Datasets/Rubrics/Experiments UI.
- **Optimize view** (todo) — for an experiment workflow's optimize run:
  - **score-over-generations** chart,
  - **candidate diff** (params gen N vs N+1),
  - **per-example drill** (output vs expected + judge reason/insight),
  - **Promote winner** button.
- Reuses the existing run flyout/canvas: each target/score/reflect run is a
  normal run, openable from a generation.

---

## 9. Human-first / agent-later layering (D7)

The chat agent already has `create_workflow` and `run_workflow` (with a
`params` override) in `ai/tools.ts`. Because datasets/rubrics/experiments are
just workflows + params, the agent mostly **already has what it needs** — it
authors an experiment workflow and sets its `rubric`/`examples` params via
`create_workflow`. The **agent layer (v2)** adds only:

- `edit_workflow_params` (or reuse `create_workflow`) — set rubric/expected/examples
- `optimize` + `promote` — drive and land the loop
- `run_workflow` already accepts `paramOverrides` (done) for candidate trials

plus a system-prompt section. No new resource machinery. A human says "make
the concepts workflow produce a better top-10 set"; the agent authors the
experiment workflow (params = examples + rubric) and runs `optimize`.

---

## 10. Build order

| Phase | Scope | State |
| --- | --- | --- |
| **P0** | Engine: keyed `paramOverrides` (§3) | **done** |
| **P1** | Params **visible + editable + persistable** in the UI (§4, §8) — the "set up an eval" surface (replaces the old Dataset/Rubric/Experiment resources) | **done** |
| **P1b** | `collect-for-eval` (run output = concept set) + `eval-score` (scorer + rubric param) (§5) | **done**; bootstrap paramified (§6.1) **done** |
| **P2** | Snapshot capture + injection seam; namespacing + teardown (§6) | todo — reproducible, isolated eval runs |
| **P3** | An **experiment workflow** (foreach `examples` → target+collect+score) + `eval-reflect` + orchestrator + `/optimize` SSE (§5) | todo — the loop |
| **P4** | Optimize UI (chart, diff, drill) + `/promote` (§7–8) | todo |
| **P5** | Agent tools (§9) | todo |

Manual eval works **today** (P0–P1b): run `bootstrap-then-process` (optionally
with `paramOverrides`), copy the `result.markdown`, run `eval-score` with it +
`expected`. P3 automates that across `params.examples`.

First concrete experiment: **concepts prompt tuning** on
`process-repo-chronological` (skip bootstrap initially to reduce variables;
add it as a second tunable scope once the loop is green), `0/0.5/1`
scoring, 1–2 pinned repos.

---

## 11. Open items (decide before/at build time)

- **Score granularity:** keep `0/0.5/1` to start; revisit continuous `0..1`
  if the gradient is too coarse for reflection.
- **Credit assignment:** holistic (D6) first; component sub-rubrics or
  staged optimization later if multi-surface reflection plateaus.
- **First target:** `process-repo-chronological` vs full
  `bootstrap-then-process` (clone/ingest dependency). Start chronological.
- **Concurrency:** examples run sequentially in v1 (cost + Neo4j
  isolation simplicity); parallelize later via distinct namespaces.
