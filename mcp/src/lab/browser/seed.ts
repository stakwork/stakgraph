import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceStore } from "strut";
import { SEED_OPTS, retireSteps, retireWorkflows } from "../seed-opts.js";

/**
 * Workflow + step templates for browser use (NOT an experiment): every
 * Playwright call is one `browser/*` step, the same steps are an agent's tools
 * (`agentTools: ["browser/*"]`), and three workflows show the two ways in — a
 * declared route (`browser-capture`) and an explored one (`browser-explore`) —
 * plus a scheduled reader (`browser-watch`). Reconciled into the workspace by
 * content hash on boot; see seed-opts.ts for the contract.
 *
 * The steps are self-contained source: they import only `strut` and
 * `./_shared.js`, and reach the browser through `ctx.services.browser` — the
 * `BrowserService` (service.ts) that `createLabStrut` builds and merges into
 * `LabServices`, not seeded here.
 */

const SEED_WORKFLOWS = [
  // one browser/capture call: open, run the listed actions, shoot
  "browser-capture",
  // an agent over the browser/* tools, answering a goal with screenshots
  "browser-explore",
  // an automation: read a region, compare with the previous run, say what changed
  "browser-watch",
];

// Names this seeder USED to publish (seeding is additive — see retireWorkflows).
const RETIRED_WORKFLOWS: string[] = [];

/** One file per step under steps/, `<name>.ts` → `browser/<name>`. */
export const SEED_STEPS = [
  // a helper the registry skips and every step below imports
  "_shared",
  "open",
  "goto",
  "back",
  "close",
  // reads: the accessibility tree with refs, a region's text, a fact by
  // script, the page's console/network trouble
  "snapshot",
  "text",
  "evaluate",
  "observe",
  // actions, by ref (from the last snapshot) or selector
  "click",
  "fill",
  "type",
  "press",
  "select",
  "hover",
  "scroll",
  "wait",
  // the frame: saved under the run's artifacts, shown to an agent that asked
  "screenshot",
  // the declared compound: open → actions → settle → screenshot
  "capture",
  // the storage state, for a human to paste under Secrets once
  "state",
] as const;

// Step types this seeder USED to publish (see retireSteps).
const RETIRED_STEPS: string[] = [];

const HERE = dirname(fileURLToPath(import.meta.url));
export const STEPS_DIR = join(HERE, "steps");

export async function seedBrowserWorkflows(workspace: WorkspaceStore): Promise<void> {
  const dir = join(HERE, "workflows");
  for (const name of SEED_WORKFLOWS) {
    try {
      const yaml = await readFile(join(dir, `${name}.yaml`), "utf-8");
      const { version, changed } = await workspace.publishWorkflowByContent(name, yaml, undefined, "browser", undefined, SEED_OPTS);
      if (changed) console.log(`[browser] seeded workflow: ${name} @ ${version}`);
    } catch (err) {
      console.warn(`[browser] could not seed workflow "${name}":`, err instanceof Error ? err.message : err);
    }
  }
  await retireWorkflows(workspace, RETIRED_WORKFLOWS, "browser");
}

export async function seedBrowserSteps(workspace: WorkspaceStore): Promise<void> {
  for (const name of SEED_STEPS) {
    const type = `browser/${name}`;
    try {
      const code = await readFile(join(STEPS_DIR, `${name}.ts`), "utf-8");
      const { version, changed } = await workspace.publishStep(type, code, undefined, "browser-seed", SEED_OPTS);
      if (changed) console.log(`[browser] seeded step: ${type} @ ${version}`);
    } catch (err) {
      console.warn(`[browser] could not seed step "${type}":`, err instanceof Error ? err.message : err);
    }
  }
  await retireSteps(workspace, RETIRED_STEPS, "browser");
}
