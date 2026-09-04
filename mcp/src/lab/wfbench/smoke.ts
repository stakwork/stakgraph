/**
 * wfbench OFFLINE smoke — no LLM, no graph, no network. Seeds the harness
 * into a throwaway workspace, discovers the steps, static-validates both
 * workflows through the authoring capability (the same check the author
 * agent gets via meta/validate-workflow), then drives every pure step with
 * fixtures and asserts the graph payloads match stakwork 58313 / 58312's id
 * conventions and only use attributes the jarvis ontology declares (vein's
 * graph backend rejects undeclared ones).
 *
 *   npx tsx src/lab/wfbench/smoke.ts
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { WorkspaceManager, buildRegistry, createVein, JARVIS_ONTOLOGY, type StepContext } from "vein";
import { seedWfbenchSteps, seedWfbenchWorkflows } from "./seed.js";
import { seedEvalSteps } from "../eval/seed.js";
import { seedArtifactSteps } from "../artifacts/seed.js";

const declared = (type: string): Set<string> => {
  const schema = JARVIS_ONTOLOGY.schemas.find((s) => s["type"] === type);
  assert.ok(schema, `ontology has no ${type}`);
  const meta = new Set(["type", "ref_id", "node_key", "parent", "domain", "type_description", "index", "icon", "primary_color", "secondary_color", "shape", "description_key", "title_key", "paid_properties", "is_system"]);
  return new Set(Object.keys(schema!).filter((k) => !meta.has(k)));
};
const assertDeclared = (type: string, data: Record<string, unknown>) => {
  const ok = declared(type);
  for (const k of Object.keys(data)) assert.ok(ok.has(k), `${type}.${k} is not declared on the ontology`);
};
const edgeAllowed = (source: string, edge: string, target: string) =>
  JARVIS_ONTOLOGY.edge_schemas.some((e) => e.source === source && e.edge === edge && e.target === target);

async function main() {
  const dir = mkdtempSync(join(process.cwd(), ".wfbench-smoke-"));
  try {
    // ── 1. seed + discover ───────────────────────────────────────────────
    const workspace = new WorkspaceManager(dir);
    await seedEvalSteps(workspace);
    await seedArtifactSteps(workspace);
    await seedWfbenchSteps(workspace);
    await seedWfbenchWorkflows(workspace);
    const { registry } = await buildRegistry(await workspace.materializeCustomSteps());
    const expectedSteps = [
      "wfbench/normalize-task",
      "wfbench/build-roster",
      "wfbench/trigger-edge",
      "wfbench/resolve-candidate",
      "wfbench/check-input-keys",
      "wfbench/classify-run",
      "wfbench/build-materials",
      "wfbench/build-eval-output",
      "wfbench/webhook-body",
      "pack",
      "eval/aggregate-scores",
      "eval/criterion-refs",
      "artifacts/dir",
      // vein lib/core steps the workflows lean on
      "graph/create-node",
      "graph/create-batch-triplet",
      "graph/create-triplet",
      "graph/graph-neighbors",
      "graph/register-namespace",
      "meta/get-workflow",
      "meta/run-workflow",
      "meta/get-step",
      "agent",
    ];
    for (const t of expectedSteps) assert.ok(registry[t], `registry missing ${t}`);
    console.log(`✔ seeded + discovered ${expectedSteps.length} steps`);

    // ── 2. static validation (what meta/validate-workflow runs) ──────────
    const vein = await createVein({ workspace, serveUi: false });
    const authoring = (vein.services as any).authoring;
    for (const name of ["wfbench-judge-criterion", "wfbench-run"]) {
      const entry = (await workspace.listWorkflows()).find((w) => w.name === name);
      assert.ok(entry, `workflow ${name} not seeded`);
      const yaml = await workspace.getWorkflowSource(name, entry!.activeVersion);
      const v = await authoring.validateWorkflow(yaml, name);
      assert.equal(v.ok, true, `${name}: ${JSON.stringify(v.errors, null, 2)}`);
      if (v.warnings.length) console.log(`  ${name} warnings:`, v.warnings.map((w: any) => `${w.path}: ${w.message}`));
      console.log(`✔ ${name} validates (${v.summary.steps} steps)`);
    }

    const run = (type: string, input: unknown, ctx?: StepContext) =>
      registry[type]!.run(registry[type]!.input.parse(input), ctx ?? ({} as StepContext));
    const ctx = { runId: "run42" } as StepContext;

    // ── 3. normalize-task ────────────────────────────────────────────────
    let task: any = await run("wfbench/normalize-task", {
      task_slug: "Fetch PR Titles!",
      instructions: "Given owner/repo, list the titles of open PRs.",
      criteria: JSON.stringify([
        { id: "c1", title: "Uses input keys", match_criteria: "reads owner and repo" },
        { id: "c1", title: "dup id", match_criteria: "…" },
        { title: "no id", match_criteria: "…", deliverables: ["out.json"] },
      ]),
      workflow_input_json: '{"owner":"stakwork","repo":"hive"}',
      rerun_expected_output: '{"titles":["a"]}',
    });
    assert.equal(task.task_slug, "fetch-pr-titles");
    assert.equal(task.task_title, "Fetch PR Titles!");
    assert.deepEqual(task.criteria.map((c: any) => c.id), ["c1", "c1-2", "c3"]);
    assert.deepEqual(task.criteria[2].deliverables, ["out.json"]);
    assert.deepEqual(task.workflow_input_keys, ["owner", "repo"]);
    assert.deepEqual(task.rerun_expected_output, { titles: ["a"] });
    await assert.rejects(() => run("wfbench/normalize-task", { task_slug: "x", instructions: "y", criteria: "[]" }), /non-empty/);
    await assert.rejects(
      () => run("wfbench/normalize-task", { task_slug: "x", instructions: "y", criteria: [{ id: "a" }], workflow_input_json: "[1]" }),
      /JSON object/,
    );
    console.log("✔ normalize-task (slug / criteria ids / JSON strings / hard fails)");

    // ── 4. build-roster (58313 ids, ontology-declared attrs only) ────────
    const roster: any = await run(
      "wfbench/build-roster",
      { task_slug: task.task_slug, task_title: task.task_title, instructions: task.instructions, criteria: task.criteria, workflow: "wfbench-run", workflow_version: "v3" },
      ctx,
    );
    assert.equal(roster.run_id, "run42");
    assert.deepEqual(roster.evalset, { node_type: "EvalSet", node_data: { id: "fetch-pr-titles", name: "Fetch PR Titles!" } });
    assertDeclared("EvalSet", roster.evalset.node_data);
    assert.equal(roster.requirement_triplets.length, 3);
    assert.deepEqual(roster.requirement_ids, ["fetch-pr-titles-c1", "fetch-pr-titles-c1-2", "fetch-pr-titles-c3"]);
    for (const [i, t] of roster.requirement_triplets.entries()) {
      assert.equal(t.edge_type, "HAS_REQUIREMENT");
      assert.deepEqual(t.source_data, { id: "fetch-pr-titles" });
      assert.equal(t.edge_data.order, i);
      assertDeclared("EvalRequirement", t.target_data);
      assert.ok(edgeAllowed("EvalSet", "HAS_REQUIREMENT", "EvalRequirement"));
    }
    assert.equal(roster.requirement_triplets[0].target_data.description, "reads owner and repo");
    assert.equal(roster.trigger_id, "fetch-pr-titles-run42");
    assert.equal(roster.trigger.node_type, "EvalTrigger");
    const trig = roster.trigger.node_data;
    assert.equal(trig.id, "fetch-pr-titles-run42");
    assert.equal(trig.project_id, "run42");
    assert.equal(trig.workflow_id, "wfbench-run");
    assert.equal(trig.workflow_version_id, "v3");
    assert.equal(trig.workflow_input, task.instructions);
    assertDeclared("EvalTrigger", trig);
    assert.ok(edgeAllowed("EvalSet", "HAS_TRIGGER", "EvalTrigger") && edgeAllowed("EvalSet", "HAS_BASELINE_TRIGGER", "EvalTrigger"));
    console.log("✔ build-roster (EvalSet / EvalRequirement×3 / EvalTrigger — 58313 ids, declared attrs)");

    // ── 5. trigger-edge (guard_first_run) ────────────────────────────────
    let e: any = await run("wfbench/trigger-edge", { neighbors: [], trigger_ref_id: "t1" });
    assert.deepEqual([e.edge_type, e.is_baseline, e.prior_triggers], ["HAS_BASELINE_TRIGGER", true, 0]);
    e = await run("wfbench/trigger-edge", { neighbors: [{ ref_id: "t1", node_type: "EvalTrigger" }], trigger_ref_id: "t1" });
    assert.equal(e.edge_type, "HAS_BASELINE_TRIGGER"); // own trigger excluded
    e = await run("wfbench/trigger-edge", { neighbors: [{ ref_id: "t0", node_type: "EvalTrigger" }, { ref_id: "r1", node_type: "EvalRequirement" }], trigger_ref_id: "t1" });
    assert.deepEqual([e.edge_type, e.prior_triggers], ["HAS_TRIGGER", 1]);
    e = await run("wfbench/trigger-edge", { neighbors: "graph/graph-neighbors: HTTP 500" });
    assert.deepEqual([e.edge_type, e.is_baseline, e.readable], ["HAS_TRIGGER", false, false]);
    console.log("✔ trigger-edge (baseline / own-excluded / prior / unreadable)");

    // ── 6. resolve-candidate (never trust the echo) ──────────────────────
    const author = { object: { workflow: "wfbench-fetch-pr-titles", version: "v2", summary: "s", changes: ["a"], customSteps: ["cand/x", 3] }, cost: 0.5, steps: 12 };
    let cand: any = await run("wfbench/resolve-candidate", {
      author, candidate: "wfbench-fetch-pr-titles",
      vbefore: { error: "not found" },
      vpin: { version: "v2", yaml: "name: x\nsteps: []\n" },
      vactive: { version: "v2", yaml: "name: x\nsteps: []\n" },
    });
    assert.deepEqual([cand.version, cand.published, cand.yaml, cand.customSteps, cand.authorCost], ["v2", true, "name: x\nsteps: []\n", ["cand/x"], 0.5]);
    cand = await run("wfbench/resolve-candidate", {
      author: { object: { version: "placeholder" } }, candidate: "c",
      vbefore: { version: "v1" }, vpin: { error: "Version placeholder not found" }, vactive: { version: "v2", yaml: "y" },
    });
    assert.deepEqual([cand.version, cand.published, cand.yaml], ["v2", true, "y"]); // echo bogus → active
    cand = await run("wfbench/resolve-candidate", {
      author: { error: "agent blew up" }, candidate: "c",
      vbefore: { version: "v1" }, vpin: { error: "x" }, vactive: { version: "v1", yaml: "y" },
    });
    assert.deepEqual([cand.version, cand.published, cand.yaml, cand.authorError], ["v1", false, "", "agent blew up"]); // nothing shipped
    console.log("✔ resolve-candidate (pin / active fallback / no-op publish / author error)");

    // ── 7. check-input-keys ──────────────────────────────────────────────
    const yaml = 'name: x\nsteps:\n  - id: a\n    type: http\n    config:\n      url: "https://api/{{ input.owner }}/{{ input["repo"] }}"\n';
    let keys: any = await run("wfbench/check-input-keys", { workflow_yaml: yaml, workflow_input: { owner: "o", repo: "r", extra: 1 } });
    assert.deepEqual([keys.keys_match, keys.referenced_keys, keys.missing, keys.unused, keys.error_type], [true, ["owner", "repo"], [], ["extra"], null]);
    assert.deepEqual(keys.launch_payload, { owner: "o", repo: "r", extra: 1 });
    keys = await run("wfbench/check-input-keys", { workflow_yaml: yaml, workflow_input: { owner: "o" } });
    assert.deepEqual([keys.keys_match, keys.missing, keys.error_type], [false, ["repo"], "input_keys_mismatch"]);
    keys = await run("wfbench/check-input-keys", { workflow_yaml: "", workflow_input: { owner: "o" } });
    assert.deepEqual([keys.keys_match, keys.error_type], [false, "no_workflow_produced"]);
    console.log("✔ check-input-keys (match+unused / missing / empty body)");

    // ── 8. classify-run ──────────────────────────────────────────────────
    let cls: any = await run("wfbench/classify-run", { gate_error_type: "input_keys_mismatch", gate_error: "m" });
    assert.deepEqual([cls.launch_ok, cls.execution_status, cls.error_type, cls.error], [false, "none", "input_keys_mismatch", "m"]);
    cls = await run("wfbench/classify-run", { run: { error: "not agent-authored" } });
    assert.deepEqual([cls.launch_ok, cls.error_type], [false, "launch_refused"]);
    cls = await run("wfbench/classify-run", { run: { runId: "child1", status: "success", output: { titles: ["a"] } } });
    assert.deepEqual([cls.launch_ok, cls.execution_status, cls.project_id, cls.error_type], [true, "completed", "child1", null]);
    assert.deepEqual(cls.run_output, { titles: ["a"] });
    const failedCls: any = await run("wfbench/classify-run", { run: { runId: "child2", status: "error", error: { message: "boom" } } });
    assert.deepEqual([failedCls.launch_ok, failedCls.execution_status, failedCls.error_type, failedCls.error], [true, "failed", "produced_workflow_failed", "boom"]);
    console.log("✔ classify-run (not launched / refused / completed / failed-still-launched)");

    // ── 9. build-materials ───────────────────────────────────────────────
    let mats: any = await run("wfbench/build-materials", {
      workflow: "wfbench-fetch-pr-titles", version: "v2", workflow_yaml: yaml,
      custom_steps: [{ type: "cand/x", code: "export default 1" }, { type: "cand/y", error: "not found" }],
      run_output: { titles: ["a"] }, execution_status: "completed", project_id: "child1",
      rerun_expected_output: { titles: ["a"] }, launch_payload: { owner: "o" }, instructions: "do it",
    });
    assert.equal(mats.n_materials, 2);
    assert.deepEqual(mats.materials.map((m: any) => m.type), ["WORKFLOW", "STEP", "LAUNCH_PAYLOAD", "RUN_OUTPUT", "EXPECTED_OUTPUT"]);
    assert.equal(mats.warnings.length, 1);
    assert.match(mats.materials_text, /### WORKFLOW: wfbench-fetch-pr-titles@v2\n\n```yaml/);
    assert.match(mats.materials_text, /### STEP: cand\/x/);
    assert.equal(mats.task_desc, "do it");
    mats = await run("wfbench/build-materials", { workflow: "w", workflow_yaml: "", instructions: "do it" });
    assert.equal(mats.n_materials, 0);
    assert.deepEqual(mats.materials.map((m: any) => m.type), ["LAUNCH_PAYLOAD", "RUN_OUTPUT"]);
    console.log("✔ build-materials (produced vs context / warnings / none)");

    // ── 10. judge zip → build-eval-output (58312 / 58115 chain) ──────────
    const scores: any = await run("eval/aggregate-scores", {
      rubric: task.criteria,
      results: [
        { object: { verdict: "pass", reasoning: "ok" }, cost: 0.1 },
        { error: "judge blew up" },
        { object: { verdict: "pass", reasoning: "fine" }, cost: 0.1 },
      ],
      judge_model: "claude-sonnet-5",
    });
    assert.deepEqual([scores.n_passed, scores.n_total, scores.all_pass], [2, 3, false]);
    const chain: any = await run("wfbench/build-eval-output", { task_slug: task.task_slug, scores, trigger_ref_id: "trig-ref", judge_model: "claude-sonnet-5" }, ctx);
    assert.equal(chain.scored, true);
    assert.equal(chain.output_id, "fetch-pr-titles-run42");
    assert.equal(chain.trigger_id, "fetch-pr-titles-run42");
    // 1 spine + 2 per criterion
    assert.equal(chain.triplets.length, 1 + 2 * 3);
    const spine = chain.triplets[0];
    assert.deepEqual([spine.source_ref_id, spine.target_type, spine.edge_type], ["trig-ref", "EvalTriggerOutput", "HAS_OUTPUT"]);
    assert.deepEqual(spine.target_data, {
      id: "fetch-pr-titles-run42", result: "fail", verdict: "fail", score: 2, max_score: 3, n_passed: 2, n_total: 3, judge_model: "claude-sonnet-5",
    });
    assertDeclared("EvalTriggerOutput", spine.target_data);
    assert.ok(edgeAllowed("EvalTrigger", "HAS_OUTPUT", "EvalTriggerOutput"));
    assert.deepEqual(chain.criterionSlots, [{ criterion_id: "c1", index: 1 }, { criterion_id: "c1-2", index: 3 }, { criterion_id: "c3", index: 5 }]);
    const fromOutput = chain.triplets[1];
    const fromReq = chain.triplets[2];
    assert.deepEqual([fromOutput.source_type, fromOutput.edge_type, fromOutput.target_type], ["EvalTriggerOutput", "HAS_CRITERION_RESULT", "CriterionResult"]);
    assert.deepEqual(fromOutput.source_data, spine.target_data); // identical object → batch dedupe hits
    assert.deepEqual([fromReq.source_type, fromReq.edge_type, fromReq.target_type], ["EvalRequirement", "HAS_CRITERION_RESULT", "CriterionResult"]);
    assert.deepEqual(fromReq.source_data, { id: "fetch-pr-titles-c1" });
    assert.deepEqual(fromReq.target_data, fromOutput.target_data);
    assert.deepEqual(fromOutput.target_data, { id: "fetch-pr-titles-run42-c1", criterion_id: "c1", title: "Uses input keys", verdict: "pass", reasoning: "ok" });
    assertDeclared("CriterionResult", fromOutput.target_data);
    assert.ok(edgeAllowed("EvalTriggerOutput", "HAS_CRITERION_RESULT", "CriterionResult") && edgeAllowed("EvalRequirement", "HAS_CRITERION_RESULT", "CriterionResult"));
    assert.equal(chain.triplets[3].target_data.verdict, "fail");
    assert.match(chain.triplets[3].target_data.reasoning, /judge error/);
    // inline trigger fallback + unscored
    const chain2: any = await run("wfbench/build-eval-output", { task_slug: "t", scores }, ctx);
    assert.deepEqual([chain2.triplets[0].source_type, chain2.triplets[0].source_data], ["EvalTrigger", { id: "t-run42" }]);
    const chain3: any = await run("wfbench/build-eval-output", { task_slug: "t", scores: { error: "refusing to zip" } }, ctx);
    assert.deepEqual([chain3.scored, chain3.triplets], [false, []]);
    // criterion-refs recovers the persisted CriterionResult refs by slot
    const refs: any = await run("eval/criterion-refs", {
      slots: chain.criterionSlots,
      record: { results: chain.triplets.map((_: any, i: number) => ({ target_ref_id: `ref${i}` })) },
    });
    assert.deepEqual(refs, [{ criterion_id: "c1", ref_id: "ref1" }, { criterion_id: "c1-2", ref_id: "ref3" }, { criterion_id: "c3", ref_id: "ref5" }]);
    console.log("✔ build-eval-output (EvalTriggerOutput + CriterionResult×3, 58312 ids, declared attrs, slots→refs)");

    // ── 11. webhook-body (resolve_webhook_payload) ───────────────────────
    const base = { task_slug: "fetch-pr-titles", task_title: "Fetch PR Titles!", judge_model: "claude-sonnet-5" };
    let body: any = await run("wfbench/webhook-body", { ...base, keys: { keys_match: false, error_type: "input_keys_mismatch", error: "m" }, cls: { launch_ok: false } });
    assert.deepEqual(body, { task_slug: "fetch-pr-titles", task_title: "Fetch PR Titles!", harness_error: true, error_type: "input_keys_mismatch", error: "m" });
    body = await run("wfbench/webhook-body", { ...base, keys: { keys_match: true }, cls: { launch_ok: false, error_type: "launch_refused", error: "r" } });
    assert.deepEqual([body.harness_error, body.error_type, body.n_passed], [true, "launch_refused", undefined]);
    body = await run("wfbench/webhook-body", { ...base, keys: { keys_match: true }, cls: { launch_ok: true }, mats: { n_materials: 0 } });
    assert.equal(body.error_type, "no_materials_produced");
    body = await run("wfbench/webhook-body", { ...base, keys: { keys_match: true }, cls: { launch_ok: true }, mats: { n_materials: 1 }, scores: { error: "refusing to zip" } });
    assert.deepEqual([body.error_type, body.error], ["judge_failed", "refusing to zip"]);
    body = await run("wfbench/webhook-body", { ...base, keys: { keys_match: true }, cls: { launch_ok: true }, mats: { n_materials: 1 } });
    assert.equal(body.error_type, "judge_failed"); // judge never ran
    body = await run("wfbench/webhook-body", { ...base, keys: { keys_match: true }, cls: { launch_ok: true, execution_status: "failed" }, mats: { n_materials: 1 }, scores });
    assert.deepEqual(Object.keys(body).sort(), ["all_pass", "criteria_results", "judge_model", "n_passed", "n_total", "pass_rate", "task_slug", "task_title"]);
    assert.deepEqual([body.n_passed, body.n_total, body.all_pass, body.judge_model, body.criteria_results.length], [2, 3, false, "claude-sonnet-5", 3]);
    assert.equal(body.harness_error, undefined);
    console.log("✔ webhook-body (keys / launch / no materials / judge failed ×2 / success — exact Hive keys)");

    // ── 12. pack-result ──────────────────────────────────────────────────
    assert.deepEqual(await run("pack", { a: 1, b: { c: 2 } }), { a: 1, b: { c: 2 } });
    console.log("✔ pack-result");

    console.log("\nALL WFBENCH SMOKE CHECKS PASSED");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

main().then(
  () => process.exit(0),
  (err) => {
    console.error(err);
    process.exit(1);
  },
);
