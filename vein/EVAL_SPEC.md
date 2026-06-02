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
| `eval/score` (recall-based) + `eval-score` workflow (§5) | **done** (`mcp/src/lab/eval`) |
| `eval-concepts` — bootstrap-only eval loop (§6) | **done** (`mcp/src/lab/concepts`) |
| Eval as a chat-agent capability (§7) | todo — the real "optimizer" |
| Reproducible/isolated runs: snapshots + namespacing (§8) | todo |
| Multi-repo dataset (`foreach` over `examples`) + thin param-sweep loop (§7) | todo |

Manual eval works **today**: open `eval-concepts`, set the `expected` gold in
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
- **Score** — `eval-score`, an LLM judge that matches produced vs expected.
- **Candidate** — what changes between runs (see §4). Param tweak or a whole
  new workflow version.
- **Harness** — `eval-concepts` chains the above into one run that returns a
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

- **Rubric** → the `rubric` param of `eval-score`.
- **Expected gold** → an `expected` param (one repo) or an `examples` array
  param (many), authored in the same markdown shape `collect-for-eval` emits.

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
   POST /workflows/eval-concepts/run
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

## 6. The harness: `eval-concepts`

One workflow that runs the whole eval as a single scoreable run:

```
reset    → clear the repo's concepts (so bootstrap re-runs fresh)
produce  → bootstrap-then-process, lookbackDays: 0  (BOOTSTRAP ONLY, no PRs)
score    → eval-score(actual = produce.markdown, expected = params.expected)
```

`lookbackDays: 0` makes the checkpoint "now" so no PRs/commits are processed —
isolating the bootstrap agent. `reset` is needed because the `is-new-repo`
gate only bootstraps an empty repo. Output: `{ score, missing, … }`. The
expected gold is the `expected` param.

---

## 7. The optimizer is the chat agent

vein already has a `ToolLoopAgent` (`src/ai/`) with `create_step`,
`edit_step`, `create_workflow`, `run_workflow`. Because targets/scorers/
datasets are all workflows + params, **the agent already has almost everything
it needs** to be the optimizer:

- read the goal + expected,
- author a candidate (edit a prompt param, or author a new step/workflow
  version — e.g. a `concepts/fetch-pr-titles` step using `ctx.services.octokit`),
- run `eval-concepts`, read `{ score, missing, insight }`,
- iterate, keeping the best version.

This is why we **don't** build a separate engine "GEPA orchestrator" as the
primary path: a workflow can't restructure itself, but the agent can — and
structural evolution (§4.2) is where the real gains are. The thin
deterministic loop is only worth it as an optional add for **unattended param
sweeps**, which the agent can even delegate.

What's missing is small (todo):
- a `run_eval` tool returning the score cleanly (vs parsing run output),
- a system-prompt section teaching the agent the eval loop,
- (later) a `promote`/`optimize` convenience for batch sweeps.

Target UX: *open the chat, say "here's the repo and the 10 concepts I expect —
make the workflow find them," and the agent iterates*, getting progressively
more sophisticated about the workflow it builds.

---

## 8. Reproducibility & isolation (todo)

For tight/automated loops the eval runs must be cheap and non-colliding:

- **Snapshots** — pin a dataset example's change-set so runs are deterministic
  and offline (`fetch-changes`/`fetch-content` read an injected `snapshot`
  instead of hitting GitHub). Today they only have a lower bound (the
  checkpoint); a snapshot freezes the upper bound too.
- **Namespacing + teardown** — concepts are keyed by `owner/repo` in Neo4j, so
  parallel candidates collide. A per-run `namespace` (suffix the repoId) +
  teardown isolates them. `eval-concepts` uses a coarse `reset` today, which
  is enough for sequential single-repo tuning.

---

## 9. The open thesis

The point of all this: **code-only bootstrap probably can't recover a repo's
~10 core user-facing capabilities** — that needs historical context (PR
titles, releases, org activity). The recall rubric will score code-only
bootstrap low and name the `missing` capabilities, which is the signal that
should push the agent to add history-aware steps. The eval turns "you need
historical context" from an opinion into a **measurable, falsifiable** claim —
and then drives building the workflow that fixes it.

---

## 10. What's next

1. **First real run** of `eval-concepts` on a small repo (needs Neo4j +
   GitHub token + LLM key) — validate the wiring and see a real score.
2. **Agent eval capability** (§7): `run_eval` tool + prompt → the chat-driven
   loop, including structural evolution.
3. **Reproducibility** (§8): snapshots + namespacing for cheap, parallel runs.
4. **Multi-repo + sweeps**: `examples` array param + an optional thin loop for
   unattended param sweeps.
