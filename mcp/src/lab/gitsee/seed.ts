import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceManager } from "vein";

/**
 * Workflow + step templates for the `gitsee` experiment (self-contained port of
 * gitsee "services" mode). Reconciled into the workspace by content hash on boot
 * — see concepts/seed.ts for the reconciliation contract (unchanged → no-op,
 * edited → new active version, prior versions archived).
 *
 * Steps are self-contained source (they import only `vein`, the third-party AI
 * SDK, Node builtins, and TYPE-ONLY imports that erase at runtime). The explore +
 * eval steps need no services. The QA tool-steps (gitsee/boot, browser-*, etc.)
 * DO reach a runtime `gitsee` services bag via `ctx.services.gitsee.*` — that bag
 * (`gitsee/services/`) is built + merged into `LabServices` in `createLabVein`,
 * not seeded here.
 */

const SEED_WORKFLOWS = [
  "gitsee-explore-services",
  // Product loop: clone → produce → boot-and-exercise (iterate a live app until
  // it runs). NOT an eval — the boot-and-exercise step writes/fixes.
  "gitsee-setup-and-run",
  // Eval/optimize stack (wires the generic eval/* steps with the gitsee
  // rubric / task / dataset), mirroring the concepts-* workflows.
  "gitsee-eval", // harness: produce setup files → score
  "gitsee-eval-score", // eval/score + setup-facts rubric
  "gitsee-eval-reflect", // eval/reflect + setup task/guidance
  "gitsee-optimize", // eval/optimize loop, wired to the above
];

const SEED_STEPS: Array<{ file: string; type: string }> = [
  { file: "clone-workspace.ts", type: "gitsee/clone-workspace" },
  // Exploration now runs on the vein-core `agent` step (gitsee-explore-services
  // wires clone → agent); there's no gitsee-specific explore step anymore.
  // Structured scorer (replaces eval/score for gitsee-eval-score): parses the
  // pm2 + compose pair and scores by name set-diffs vs the gold + an LLM residue.
  { file: "score-setup.ts", type: "gitsee/score-setup" },
  // Boot gate: actually runs the produced setup (compose up + staklink) and
  // checks the frontend renders in headless chromium — the dominant eval signal.
  { file: "verify-setup.ts", type: "gitsee/verify-setup" },
  // Product loop (NOT eval): boots the produced setup, DRIVES the live app in a
  // browser, observes failures, fixes the config/repo, reboots — until it runs.
  { file: "boot-and-exercise.ts", type: "gitsee/boot-and-exercise" },
  // QA tool-steps (the decomposed boot-and-exercise): thin, self-contained steps
  // that reach the per-run browser/stack/vision HARNESS via ctx.services.gitsee.*
  // (see ../services). Usable BOTH as workflow steps (the C2 gitsee-qa-iteration
  // loop) AND as agent tools (agentTools on the core `agent` step).
  { file: "stage-setup.ts", type: "gitsee/stage-setup" },
  { file: "boot.ts", type: "gitsee/boot" },
  { file: "browser-open.ts", type: "gitsee/browser-open" },
  { file: "browser-snapshot.ts", type: "gitsee/browser-snapshot" },
  { file: "browser-click.ts", type: "gitsee/browser-click" },
  { file: "browser-fill.ts", type: "gitsee/browser-fill" },
  { file: "browser-press.ts", type: "gitsee/browser-press" },
  { file: "browser-observe.ts", type: "gitsee/browser-observe" },
  { file: "assess-ui.ts", type: "gitsee/assess-ui" },
  { file: "read-logs.ts", type: "gitsee/read-logs" },
  { file: "finalize-setup.ts", type: "gitsee/finalize-setup" },
  // Captures the agent's repo edits as a replayable git diff (part of the
  // deliverable): the last step of gitsee-explore-services, passes the agent
  // output through and adds `diff`.
  { file: "capture-edits.ts", type: "gitsee/capture-edits" },
];

const HERE = dirname(fileURLToPath(import.meta.url));

export async function seedGitseeWorkflows(workspace: WorkspaceManager): Promise<void> {
  const dir = join(HERE, "workflows");
  for (const name of SEED_WORKFLOWS) {
    try {
      const yaml = await readFile(join(dir, `${name}.yaml`), "utf-8");
      const { version, changed } = await workspace.publishWorkflowByContent(name, yaml);
      if (changed) console.log(`[gitsee] seeded workflow: ${name} @ ${version}`);
    } catch (err) {
      console.warn(
        `[gitsee] could not seed workflow "${name}":`,
        err instanceof Error ? err.message : err,
      );
    }
  }
}

export async function seedGitseeSteps(workspace: WorkspaceManager): Promise<void> {
  const dir = join(HERE, "steps");
  for (const { file, type } of SEED_STEPS) {
    try {
      const code = await readFile(join(dir, file), "utf-8");
      const { version, changed } = await workspace.publishStep(type, code, undefined, "gitsee-seed");
      if (changed) console.log(`[gitsee] seeded step: ${type} @ ${version}`);
    } catch (err) {
      console.warn(
        `[gitsee] could not seed step "${type}":`,
        err instanceof Error ? err.message : err,
      );
    }
  }
}
