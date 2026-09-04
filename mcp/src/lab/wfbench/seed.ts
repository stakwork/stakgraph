import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceStore } from "vein";
import { SEED_OPTS, retireSteps } from "../seed-opts.js";

/**
 * wfbench — the Workflow Editor Agent Benchmark harness (the vein port of
 * stakwork workflow 58313; see plans/wfbench-harness.md). Pure plumbing
 * steps + the two workflows. Everything here is seeded UNSTAMPED (no
 * publisher arg → not "ai"), so the meta surface — i.e. the author agent
 * inside a benchmark run — can read but never edit, run, or overwrite the
 * harness. Content-hash reconciled (SEED_OPTS): a changed committed copy
 * wins at boot, an unchanged one leaves a workspace-side edit active.
 *
 * Steps (all pure — no services, no LLM, no graph; the graph writes are
 * vein's graph/* lib steps and the judge is the core agent step):
 *   wfbench/normalize-task     the Hive/58313 task payload → canonical task
 *   wfbench/build-roster       EvalSet / EvalRequirement / EvalTrigger payloads (58313 ids)
 *   wfbench/trigger-edge       HAS_BASELINE_TRIGGER vs HAS_TRIGGER (guard_first_run)
 *   wfbench/resolve-candidate  pin what the author actually shipped (never its echo)
 *   wfbench/check-input-keys   input-key contract gate (wfbench_check_input_keys.py)
 *   wfbench/classify-run       launch_ok / completed / failed (wfbench_classify_run_result.py)
 *   wfbench/build-materials    judge materials (wfbench_build_produced_materials.py)
 *   wfbench/build-eval-output  EvalTriggerOutput + CriterionResult triplets (58312)
 *   wfbench/webhook-body       the one Hive callback body (resolve_webhook_payload)
 *
 * GRANT DISCIPLINE: none of wfbench/* is ever granted to an agent's
 * agentTools — the author gets meta/* only, the judge gets nothing.
 */
const SEED_STEPS: Array<{ file: string; type: string }> = [
  { file: "normalize-task.ts", type: "wfbench/normalize-task" },
  { file: "build-roster.ts", type: "wfbench/build-roster" },
  { file: "trigger-edge.ts", type: "wfbench/trigger-edge" },
  { file: "resolve-candidate.ts", type: "wfbench/resolve-candidate" },
  { file: "check-input-keys.ts", type: "wfbench/check-input-keys" },
  { file: "classify-run.ts", type: "wfbench/classify-run" },
  { file: "build-materials.ts", type: "wfbench/build-materials" },
  { file: "build-eval-output.ts", type: "wfbench/build-eval-output" },
  { file: "webhook-body.ts", type: "wfbench/webhook-body" },
];

const SEED_WORKFLOWS: Array<{ name: string; description: string }> = [
  {
    name: "wfbench-judge-criterion",
    description:
      "wfbench: LLM-as-judge for ONE rubric criterion over the produced workflow's materials (agent schema mode, no tools). Run per criterion by wfbench-run; a crash packs { error } = honest FAIL. Input: { criterion, task_desc, materials_text }.",
  },
  {
    name: "wfbench-run",
    description:
      "Workflow Editor Agent Benchmark — Task Runner (stakwork 58313's twin): graph roster (EvalSet/EvalRequirement/EvalTrigger) -> meta/* author builds wfbench-<slug> -> input-key gate -> rerun via meta/run-workflow -> per-criterion judge -> record EvalTriggerOutput/CriterionResult -> POST the Hive callback. Input: { task_slug, task_title?, instructions, criteria, workflow_input_json?, rerun_expected_output?, webhook_url?, namespace? }.",
  },
];

// Types this seeder USED to publish (seeding is additive — see retireSteps).
const RETIRED_STEPS = ["wfbench/pack-result"]; // → vein core `pack`

const HERE = dirname(fileURLToPath(import.meta.url));

export async function seedWfbenchSteps(workspace: WorkspaceStore): Promise<void> {
  const dir = join(HERE, "steps");
  for (const { file, type } of SEED_STEPS) {
    try {
      const code = await readFile(join(dir, file), "utf-8");
      const { version, changed } = await workspace.publishStep(type, code, undefined, "wfbench-seed", SEED_OPTS);
      if (changed) console.log(`[wfbench] seeded step: ${type} @ ${version}`);
    } catch (err) {
      console.warn(`[wfbench] could not seed step "${type}":`, err instanceof Error ? err.message : err);
    }
  }
  await retireSteps(workspace, RETIRED_STEPS, "wfbench");
}

export async function seedWfbenchWorkflows(workspace: WorkspaceStore): Promise<void> {
  const dir = join(HERE, "workflows");
  for (const { name, description } of SEED_WORKFLOWS) {
    try {
      const yaml = await readFile(join(dir, `${name}.yaml`), "utf-8");
      const { version, changed } = await workspace.publishWorkflowByContent(name, yaml, description, "wfbench", undefined, SEED_OPTS);
      if (changed) console.log(`[wfbench] seeded workflow: ${name} @ ${version}`);
    } catch (err) {
      console.warn(`[wfbench] could not seed workflow "${name}":`, err instanceof Error ? err.message : err);
    }
  }
}
