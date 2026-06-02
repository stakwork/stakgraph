# Evals & Self-Improving Experiments — Spec

A design spec for turning vein into a platform for **evals** and
**self-improving loops**: pin a goal, run a workflow against pinned
examples, score the output against an expected gold standard, and let an
optimizer evolve the workflow's tunable knobs (`params`) until the output
is good.

This builds on `SPEC.md` (the engine) and reuses its existing primitives
wherever possible. It is **not** a CLI — every operation is a first-class
API endpoint, visible in the UI, and (later) drivable by the AI agent.

> GOAL: AS SIMPLE AS POSSIBLE. Reuse runs, `params`, and versioning. Add
> the smallest set of new concepts: **Dataset, Rubric, Experiment**.

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
| D2 | **Resources live in the vein workspace**, versioned like workflows. | Consistent with how workflows/steps already work; one persistence model. |
| D3 | **Candidate scope = `params` only** (prompts/thresholds/selectors), as a **multi-scope union** across `{workflow, key}`. | Matches the existing "experiment surface"; smallest search space; covers multi-prompt targets. |
| D4 | **Nested/scoped param override** is added to the engine (keyed by workflow name). | The tunable prompts live in subflows; per-run `params` don't propagate today. General; opt-in; back-compat. |
| D5 | **Pin-by-snapshot** datasets. | Deterministic, offline, fast, cheap evals — no GitHub during optimization. |
| D6 | **Holistic reflection** for credit assignment (start). | Simplest; reflector sees all prompts + insights and may change any. Component sub-rubrics / staged optimization can come later. |
| D7 | **Human-driven first, agent-driven later.** | The agent layer is thin tool wrappers over the same endpoints; build them tool-friendly from day one. |

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

## 4. Resources

Stored in the workspace next to `workflows/` and `steps/`. JSON shapes are
intentionally simple (tool-friendly for the future agent layer).

```
workspace/
  workflows/   steps/                                   (exist)
  datasets/<name>/
     dataset.json                                       # metadata + example index
     examples/<exampleId>/
        input.json        # EXACTLY the target workflow's input
        expected.json     # the gold label
        snapshot.json     # captured change-set / fixtures (pin-by-snapshot)
        notes.txt         # optional
  rubrics/<name>.json                                   # judge prompt + scale
  experiments/<name>.json                               # the binding + budget
  experiments/<name>/runs/<ts>/
     gen-<N>/{ candidate.json, results.json, aggregate } 
     summary.json
```

### 4.1 Dataset

A reusable, re-labeled-rarely collection of **pinned examples**. Each
example's `input` is exactly what the target workflow takes; `expected` is
the human-authored gold; `snapshot` makes the run deterministic.

```jsonc
// datasets/concepts-goldset/dataset.json
{
  "name": "concepts-goldset",
  "kind": "concepts",              // informal tag: compatible targets/rubrics
  "examples": ["hive@a1b2c3"]
}
```
```jsonc
// datasets/concepts-goldset/examples/hive@a1b2c3/input.json
{ "owner": "stakwork", "repo": "hive",
  "rev": "a1b2c3d4",                       // pins the clone (working tree)
  "until": "2025-09-01T00:00:00Z" }        // upper bound on the change-set
```
```jsonc
// expected.json
{ "concepts": [
    { "name": "Real-time Chat",  "description": "Users can…" },
    { "name": "Task Management", "description": "…" }
    /* ~10 */ ] }
```
```jsonc
// snapshot.json — captured once at label time; bypasses GitHub at eval time
{ "changes": [ { "type": "pr", "id": "42", "data": { … }, "markdown": "…" }, … ] }
```

**Pin-by-snapshot (D5).** `fetch-changes` / `fetch-content` gain a seam:
when `input.snapshot` is present, return it instead of calling GitHub. A
**capture** workflow produces the snapshot from `{owner, repo, until}`.
Note: today `fetch-changes` only has a *lower* bound (the checkpoint); the
capture path introduces the upper bound (`until`) and freezes the result.

### 4.2 Rubric

The **fixed measuring stick** — how to turn `(output, expected)` into a
score. It is a prompt (editable in the UI) but the optimizer **never
mutates it**. Reusable across every experiment of the same `output_kind`.

```jsonc
// rubrics/concept-quality.json
{
  "name": "concept-quality",
  "output_kind": "concepts",
  "scale": [0, 0.5, 1],
  "judgePrompt": "Score produced Concepts vs the expected gold set. Match \
semantically, not by string. Weigh: RECALL (expected capabilities present), \
PRECISION (penalize junk / over-granular / implementation-named concepts), \
NAMING (capability-focused: 'Real-time Chat' good, 'Pusher WebSockets' bad), \
DESCRIPTION accuracy. Output SCORE / REASON / INSIGHT.",
  "aggregation": "mean"
}
```

Score format (ported from the prior harness): `SCORE` (0 / 0.5 / 1) +
`REASON` (what's right/wrong) + `INSIGHT` (one prompt-level suggestion).
The `INSIGHT` is what feeds reflection.

### 4.3 Experiment

Binds a **target** workflow + **dataset** + **rubric** + the **candidate
surface** + an **output extractor** + **isolation** + **budget**. This is
the entity you press "Optimize" on.

```jsonc
// experiments/concepts-prompt-tuning.json
{
  "name": "concepts-prompt-tuning",
  "target": "process-repo-chronological",     // workflow being optimized
  "dataset": "concepts-goldset",
  "rubric": "concept-quality",

  "candidate": {                              // multi-scope param union (D3)
    "tunable": [
      { "scope": "bootstrap-then-process", "keys": ["bootstrapSystem","sizing"] },
      { "scope": "process-change",         "keys": ["systemPrompt","guidelines"] }
    ],
    "seed": "from-workflow-defaults"
  },

  "output":    { "extractor": "concepts/collect-for-eval" }, // → { concepts }
  "isolation": { "namespace": "per-run", "teardown": true },
  "budget":    { "maxGenerations": 8, "perfectScore": 0.9 }
}
```

### 4.4 ExperimentRun (generations)

One `optimize` invocation produces a sequence of generations. This is
exactly the `runs/gen-N/` artifact layout the prior harness wrote.

```jsonc
// experiments/<name>/runs/<ts>/gen-3/results.json
[{ "exampleId": "hive@a1b2c3",
   "output": { "concepts": [ … ] },
   "score": 0.5, "reason": "…", "insight": "…" }]
// .../gen-3/candidate.json     → the params tried this generation
// .../summary.json             → { best_score, best_gen, history:[…] }
```

---

## 5. The optimize loop

### 5.1 Two built-in workflows (everything-is-a-workflow, D1)

- **`eval/score`** — input `{ output, expected, rubric }`; one `llm` step
  using `rubric.judgePrompt`; output `{ score, reason, insight }`.
- **`eval/reflect`** — input `{ candidate, results, history }`; one `llm`
  step; output a new candidate (`paramOverrides`-shaped JSON over the
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

### 6.1 Prerequisite paramification

To make the bootstrap prompt tunable (it is currently hardcoded in
`bootstrap.ts`: `buildBootstrapPrompt`, the `systemOverride`, sizing
thresholds), lift those into `concepts/bootstrap-explore` config sourced
from the workflow's `params`. `concepts/decide` is already paramified.
After this, the candidate surface is a clean union of `params` across the
two workflows.

---

## 7. API surface

Mirrors `/workflows`. All JSON in/out (tool-friendly).

| Method | Path | Purpose |
| --- | --- | --- |
| GET/POST | `/datasets`, `/datasets/:name` | list / read / create datasets |
| POST | `/datasets/:name/capture` | run the capture flow → write a snapshot example |
| GET/POST | `/rubrics`, `/rubrics/:name` | list / read / create rubrics |
| GET/POST | `/experiments`, `/experiments/:name` | list / read / create experiments |
| POST | `/experiments/:name/eval` | score current params once (no evolution) |
| POST | `/experiments/:name/optimize` | **SSE stream** of generation events |
| GET | `/experiments/:name/runs[/:ts]` | generation history |
| POST | `/experiments/:name/promote` | write winning candidate into target `params` defaults + publish a new workflow version |

`promote` closes the loop using the **existing** versioning machinery.

---

## 8. UI surface

- New **Experiments** section in the sidebar (alongside Workflows/Steps).
- Experiment detail:
  - **score-over-generations** chart,
  - **candidate diff** viewer (params gen N vs N+1),
  - **per-example drill** (output vs expected + judge reason/insight),
  - **Promote winner** button.
- Reuses the existing run flyout/canvas: each target/score/reflect run is a
  normal run, openable from a generation.

---

## 9. Human-first / agent-later layering (D7)

The chat agent already has `run_workflow` with a `params` override
(`ai/tools.ts`), so it can do crude one-off trials today. The eval
framework adds the substrate. The **agent layer (v2)** is then just thin
tools over the §7 endpoints:

- `create_dataset`, `capture_snapshot`
- `create_rubric`
- `create_experiment`
- `run_experiment` (eval / optimize)
- `get_experiment_run`
- `promote`

plus a system-prompt section. No engine changes — the human flow and the
agent flow call the same endpoints. A human says "make the concepts
workflow produce a better top-10 set"; the agent builds the dataset/rubric/
experiment and runs the optimization.

---

## 10. Build order

| Phase | Scope | Notes |
| --- | --- | --- |
| **P0** | Engine: keyed `paramOverrides` (§3) | Small, isolated, unit-testable in vein alone. |
| **P1** | Dataset/Rubric/Experiment resources + storage + read API + UI lists | No loop yet. Establishes the data model. |
| **P2** | Snapshot capture + injection seam; namespacing + teardown; paramify bootstrap (§6) | Concepts-specific prerequisite work. |
| **P3** | `eval/score` + `eval/reflect` workflows + orchestrator + `/eval` + `/optimize` SSE (§5) | The actual loop. Ports the prior GEPA harness. |
| **P4** | Optimize UI (chart, diff, drill) + `/promote` (§7–8) | Human-driven experience complete. |
| **P5** | Agent tools (§9) | Agent-driven layer. |

First concrete experiment: **`concepts-prompt-tuning`** on
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
