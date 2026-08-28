import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceManager } from "vein";

/**
 * GAIA LAB steps + workflows — the harness that scored 5/5 on the first
 * level-1 batch, promoted from the workspace where the assistant authored it
 * (EVOLVE_SPEC §promote: winners become committed, diffable artifacts).
 * The grader itself stays in the in-code `gaia` service (`service.ts`, on
 * the LabServices bag) — these steps are THIN plumbing over
 * `ctx.services.gaia.*`; editing them can only break plumbing, never change
 * how scoring works.
 *
 * - `gaia/list-tasks`, `gaia/get-task` — task plumbing (gold stripped by the
 *   service). get-task stages a task's attached file into the run's
 *   artifacts dir so agent steps (cwd = artifacts dir) can read it.
 * - `gaia/evaluate` — the real leaderboard scorer. HARNESS-ONLY: grant only
 *   to harness workflows, never to a producing agent's `agentTools`.
 * - `gaia/pack-result`, `gaia/summarize-batch` — pure combiners.
 * - `gaia/digest-results` — aggregate graded results into the evolve loop's
 *   propose digest (verdict channel only; accuracy as `fitness`).
 *
 * The evolve harness (gaia-candidate-run / gaia-evolve-gen / gaia-evolve)
 * mirrors harvey's, driven by the generic `eval/evolve-loop`. All seeded
 * UNSTAMPED, so the meta surface can read but never edit, run, or
 * overwrite them.
 *
 * Seeding is content-hash reconciled: the committed copy is authoritative at
 * boot. A workspace-side evolution of these survives restarts only once it's
 * ported back here.
 */
const SEED_STEPS: Array<{ file: string; type: string }> = [
  { file: "list-tasks.ts", type: "gaia/list-tasks" },
  { file: "get-task.ts", type: "gaia/get-task" },
  { file: "evaluate.ts", type: "gaia/evaluate" },
  { file: "pack-result.ts", type: "gaia/pack-result" },
  { file: "summarize-batch.ts", type: "gaia/summarize-batch" },
  { file: "digest-results.ts", type: "gaia/digest-results" },
];

const HERE = dirname(fileURLToPath(import.meta.url));

const SEED_WORKFLOWS: Array<{ name: string; description: string }> = [
  {
    name: "gaia-produce",
    description:
      "Helper: stage a GAIA task's file (if any) into a fresh artifacts dir, run a tool-using agent to answer it, and pack the result. A failed agent (rare no-object-generated error, etc.) falls back to an empty answer instead of aborting the caller. Input: { taskId }. Output: { taskId, question, level, hasFile, stagedPath, answer, cost, steps, produceError }.",
  },
  {
    name: "gaia-run",
    description:
      "GAIA single-task harness: get-task -> agent produce -> real scorer. Input { taskId }. Output combines score + the produced answer/cost/steps.",
  },
  {
    name: "gaia-batch",
    description:
      "GAIA batch harness: list-tasks (by level) -> first `limit` -> produce per task via gaia-produce -> one gaia/evaluate call -> merged report {accuracy, byLevel, perTask, totalCost, totalSteps}.",
  },
  {
    name: "gaia-candidate-run",
    description:
      "Run an ai-stamped candidate produce workflow on ONE GAIA task via meta/run-workflow (own runId, fresh registry) and score its answer with the real scorer (fromRun unpack — a failed run scores as an honest zero). Input: { workflow, version?, taskId }. Output: { taskId, candidate, version, correct, answer, level, question, produceStatus, runResult, … }.",
  },
  {
    name: "gaia-evolve-gen",
    description:
      "ONE GENERATION of the gaia evolution loop: meta/* authoring agent publishes a candidate version -> gaia-candidate-run over the task set (pinned version) -> gaia/digest-results. Invoked by eval/evolve-loop with { tasks, mission, candidateName, generation, briefing }. Output: { candidate, generation, version, summary, changes, missingSecrets, authorCost, authorSteps, digest }.",
  },
  {
    name: "gaia-evolve",
    description:
      "GAIA authoring harness (hill-climb): baseline gaia-run over the task set -> digest -> eval/evolve-loop over gaia-evolve-gen generations (accuracy fitness, exact-match so improveMargin 0) -> report with best version vs baseline. TRAIN scores — validate the best version on held-out tasks before promoting. Input: { tasks: [taskId, …], mission, generations? }.",
  },
];

export async function seedGaiaWorkflows(workspace: WorkspaceManager): Promise<void> {
  const dir = join(HERE, "workflows");
  for (const { name, description } of SEED_WORKFLOWS) {
    try {
      const yaml = await readFile(join(dir, `${name}.yaml`), "utf-8");
      const { version, changed } = await workspace.publishWorkflowByContent(name, yaml, description, "gaia");
      if (changed) console.log(`[gaia] seeded workflow: ${name} @ ${version}`);
    } catch (err) {
      console.warn(`[gaia] could not seed workflow "${name}":`, err instanceof Error ? err.message : err);
    }
  }
}

export async function seedGaiaSteps(workspace: WorkspaceManager): Promise<void> {
  const dir = join(HERE, "steps");
  for (const { file, type } of SEED_STEPS) {
    try {
      const code = await readFile(join(dir, file), "utf-8");
      const { version, changed } = await workspace.publishStep(type, code, undefined, "gaia-seed");
      if (changed) console.log(`[gaia] seeded step: ${type} @ ${version}`);
    } catch (err) {
      console.warn(
        `[gaia] could not seed step "${type}":`,
        err instanceof Error ? err.message : err,
      );
    }
  }
}
