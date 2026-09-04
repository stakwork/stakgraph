# wfbench — the Workflow Editor benchmark harness as a vein workflow

Port of stakwork workflow 58313 ("Workflow Editor Agent Benchmark — Task
Runner", v189235) plus its nested workflows, rebuilt on the lab's existing
produce → run → judge → record substrate. Second purpose: use the same
harness to PORT stakwork workflows onto vein one task at a time, with a
rubric that checks the port is faithful.

## Decision: the author is an `agent` step with `agentTools: ["meta/*"]`

Not a "chat assistant as a step". Reasons:

- The chat builder (`vein/src/ai/tools.ts`) and `meta/*` are the SAME
  authoring capability (`services.authoring`). `meta/*` is its in-run form.
  This is exactly how `gaia-evolve-gen` / `harvey-evolve-gen` already author
  candidates, and it is what EVOLVE_SPEC §5.3.2 prescribes.
- The chat is a detached background job with its own store, dispatch
  notifications and auto-turn caps. Wrapping it as a step means polling a
  chat and parsing prose; the agent step gives schema-mode structured
  output, `cost`/`steps`, per-tool-call events on the canvas, retry/onError
  and run-control tree linkage for free.
- Stakwork 54419 "Workflow Editor JSON" is itself just a wrapper around an
  agent loop (54517). The vein twin of "54419 as a black box returning
  artifact pointers" is `agent` + `meta/*` returning `{ workflow, version }`.
- Chat-only tools the author does NOT need: `bash` (forbidden next to
  `meta/*`), `graph_query`, run control, `set_active_version`. The one
  worth adding is `validate_workflow` → new `meta/validate-workflow` step
  (thin wrapper over the same `validate(yaml, name)`).

## Step map (58313 → vein)

| 58313 (stakwork) | vein |
| --- | --- |
| `set_var` | workflow `input` + `params` |
| EvalSet / EvalRequirement / EvalTrigger roster (55741, 58114, 55740, `hop_check_trigger_exists`, `guard_first_run`) | `graph/create-node` (EvalSet), `graph/create-batch-triplet` (EvalRequirements), `graph/graph-neighbors` + `wfbench/trigger-edge` → `graph/create-triplet` with `HAS_BASELINE_TRIGGER` or `HAS_TRIGGER`. Ids: EvalSet = task slug verbatim, EvalRequirement = `<slug>::<criterion_id>` — Hive's `eval-nodes.ts` convention (Hive upserts the roster before dispatch and reads it back by those ids), not harvey's `<slug>-<id>` |
| `run_workflow_editor` (54419) | `agent` + `agentTools: ["meta/*"]`, `toolFilter: [str_replace_based_edit_tool]`, schema `{ workflow, version, summary, changes, missingSecrets }` |
| `extract_artifacts.py`, JSONPath for workflowId/version | schema output; never trust the echo — `meta/get-workflow` pin fallback (`vpin.version || vactive.version`) + the `published` gate vs `vbefore` (copy from `gaia-evolve-gen`) |
| 58414 WhileLoop ×15 + `api_fallback_fetch` | gone. `meta/publish-workflow` is synchronous; `meta/get-workflow { name, version }` returns the YAML |
| `wfbench_check_input_keys.py` | pure step `wfbench/check-input-keys` (declared `input` keys of the produced YAML vs the task's `workflow_input_json`) |
| 57425 Run Trigger + webhook wait | `meta/run-workflow { name, version, input }` — awaits, own runId, refuses non-`ai` workflows |
| `wfbench_classify_run_result.py` | pure step `wfbench/classify-run` over `run.status` / `run.output` / `run.runId` |
| `wfbench_build_produced_materials.py` | pure step `wfbench/build-materials` (produced YAML + agent-authored custom steps via `meta/get-step`, the engine's static validation via `meta/validate-workflow`, run output, expected output, launch payload) |
| `guard_materials_present` | `if n_materials > 0`, else harness error `no_materials_produced` |
| skill `harvey_lab_score_rubric` (all criteria, one call) | `foreach` criteria → subflow `wfbench-judge-criterion` (clone of `harvey-judge-criterion`: agent schema mode, materials in prompt / artifacts dir, NO tools) → `eval/aggregate-scores`. Judge crash = `{ error }` = honest FAIL |
| `guard_judge_ran`, `guard_valid_score` | `if` on aggregate output, else `judge_failed` |
| 58312 record (EvalTriggerOutput + CriterionResult + edges) | `eval/build-eval-chain` → `graph/create-batch-triplet` (idempotent on runId) |
| `resolve_webhook_payload` (4-way) + `post_result` | one `pack` depending on every branch (`||` chain) → `if webhookUrl` → `http POST`. Output == what was posted (fixes the 58313 `set_output` divergence) |

Webhook body: byte-compatible with what Hive parses today
(`RunnerScoreSchema`): success `{ task_slug, task_title, n_passed, n_total,
all_pass, pass_rate, judge_model, criteria_results }`, failure
`{ harness_error: true, error_type }` with no score fields.

## Files (v1 — BUILT; offline smoke green)

```
mcp/src/lab/wfbench/
  seed.ts                          # seedWfbenchSteps / seedWfbenchWorkflows (SEED_OPTS)
  smoke.ts                         # offline: seed, discover, validate YAML, drive every pure step,
                                   #   check graph payloads against JARVIS_ONTOLOGY
  steps/                           # all pure — no services, LLM, or graph
    normalize-task.ts              # Hive/58313 payload → canonical task (hard-fails early)
    build-roster.ts                # EvalSet / EvalRequirement×N / EvalTrigger payloads (58313 ids)
    trigger-edge.ts                # guard_first_run: HAS_BASELINE_TRIGGER vs HAS_TRIGGER
    resolve-candidate.ts           # vpin || vactive, published vs vbefore (never trust the echo)
    check-input-keys.ts            # input.<key> refs in the YAML vs the launch payload
    classify-run.ts                # launch_ok / completed / failed / harness error
    build-materials.ts             # judge materials → one markdown block (judge needs no tools)
    build-eval-output.ts           # EvalTriggerOutput + CriterionResult triplets (58312 ids)
    webhook-body.ts                # resolve_webhook_payload (exact Hive keys)
    pack-result.ts                 # passthrough / onError pack
  workflows/
    wfbench-run.yaml               # the 58313 twin
    wfbench-judge-criterion.yaml   # per-criterion judge subflow (agent schema mode)
```

Not built (deferred): `wfbench-batch` (foreach tasks), `wfbench-port-task`
(stakwork id → task; needs `STAK_CUSTOMER_TOKEN`) and the edit-existing
(`baseline_workflow`) path. The bench itself needs NO stakwork credentials:
tasks arrive as input.

`wfbench-run` input: `{ task_slug, task_title?, instructions, criteria,
workflow_input_json?, rerun_expected_output?, webhook_url? }`. Params:
`namespace`, `authorSystem`, `authorGuidance`, `authorModel`,
`authorMaxSteps`, `judgeModel`, `judgeConcurrency` (+ `judgeSystem` /
`judgePrompt` / `judgeMaxSteps` on the judge subflow).

Registered in `createLabVein.ts` after the artifact steps. Needs the graph
(mcp's Neo4j) and `ANTHROPIC_API_KEY`.

## Grant discipline

- author: `meta/*` + editor tool only. Never bash, never `graph/*`, never
  `wfbench/*`.
- judge: nothing useful to call (materials are pre-resolved into the
  prompt; the core `llm` step takes a Zod schema, not JSON Schema, so the
  judge is the `agent` step in schema mode with the editor tool over an
  empty dir). Cheap and non-gameable.
- produced workflow: runs via `meta/run-workflow` so it is necessarily
  publisher `ai`; the seeded harness surface can never be graded as a
  candidate.

## The port use case

Each stakwork workflow becomes ONE task for `wfbench-run`:

- `instructions` = the stakwork body JSON (`data.workflow` from
  `wfbench/stakwork-fetch`) + "port this to vein" + a translation
  cheat-sheet in `params.authorSystem`:
  SetVar → `input`/`params`; JSONBuilder / JSONPathParser / IfElseValue →
  `{{ }}` templates; Request → `http`; IfElseCondition → `if` + `when`;
  WhileLoop → `loop`; ForEachCondition → `foreach`; WorkflowRunner →
  `subflow`; Script → `meta/create-step`; Skill/LLM → `agent` / `llm`;
  `[#(step).output.x]` → `{{ step.x }}`; `%%SECRET%%` → secret NAMES from
  `meta/list-secrets` (the harvey seeder already did this mapping by hand
  for the deliver pipeline — same rules, now given to the agent).
- `workflow_input_json` + `rerun_expected_output` = one completed stakwork
  project's real input and output for that workflow (fetched, not
  invented). Faithfulness is then judged against a real run, not prose.
- `criteria` = derived once per workflow by `wfbench-port-task` (an agent
  in schema mode reading the stakwork body: same input keys, every step
  with a side effect present, same output shape, secrets by name, no
  hardcoded values) and kept as EvalRequirements after a human glance.
  Nothing in the produce path ever sees them.

`baseline_workflow` (Hive's v2 edit-existing case): `meta/publish-workflow`
refuses to overwrite a workflow the agent surface did not author, so the
harness first republishes the baseline YAML under `<name>-candidate` (now
publisher `ai`) and points the author at that. Cheap to support from day
one; 58313 v1 skipped it.

## Small vein/lab changes needed

1. DONE — `meta/validate-workflow` step (wraps the chat's `validate`).
2. DONE — `harvey/build-eval-chain`, `harvey/aggregate-scores`,
   `harvey/criterion-refs` promoted to `eval/*` (no aliases: harvey-score
   and the smoke were re-pointed in the same change; `workflow` is now a
   required input of build-eval-chain).

## Graph writes vs 58313 / 58312 (what differs, and why)

vein's graph backend rejects attributes the ontology does not declare, so:
`EvalSet.project_id` (an int in stakwork; vein runIds are strings) and the
`name` on EvalTrigger / EvalTriggerOutput are omitted (EvalTrigger gets
`agent: wfbench-run` as its title). 58312's `CriterionResult -HAS_CAUSE->
Workflow_version(material_ref_id)` is not written: the ontology has no
such relationship and the produced vein workflow is not a Workflow_version
node. Everything else — ids, edge types, EvalTriggerOutput's score fields,
CriterionResult's verdict/reasoning — matches, and the smoke asserts it.

## Not ported on purpose

- 58414's polling loop and the `api_fallback_fetch` (no async publish).
- 57425's webhook wait (`meta/run-workflow` awaits).
- 54419's `set_metadata` gap on the `workflow_id="new"` path.
- The graph "Page" blurb calling it a timing page.
