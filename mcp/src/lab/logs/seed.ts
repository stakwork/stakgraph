import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { WorkspaceStore } from "strut";
import { SEED_OPTS, retireSteps, retireWorkflows } from "../seed-opts.js";

/**
 * logs — the logs agent (`/logs/agent`, mcp/src/log) as a strut workflow,
 * for the orchestrator to ask this swarm, or a peer swarm, what its logs say
 * (`meta/run-workflow` here, `strut/run-workflow { peer, workflow: "logs" }`
 * there). `job/dir` → `logs/context` (the endpoint's prompt assembly, this
 * deployment's swarm / pool / workspace, the time budget) → `agent` over the
 * six logs/* tools → `pack`.
 *
 * The steps are plumbing over the in-code `services.logs`
 * (./service.ts — createLabStrut builds it), which runs the endpoint's own
 * fetches and texts (log/tools.ts, log/context.ts) and, for the endpoint's
 * `bash`, three tools that run nothing: search, read, insights. Seeded
 * UNSTAMPED, content-hash reconciled (SEED_OPTS), like explore.
 */
const SEED_WORKFLOWS: Array<{ name: string; description: string }> = [
  {
    name: "logs",
    description:
      "Ask this swarm's logs a question: the logs agent (/logs/agent's, with its tools and time budget) fetches the swarm's CloudWatch log groups (Logs Insights, long lines recovered) and its app's Quickwit logs into files, searches and reads them (logs/search, logs/read), counts and groups with Logs Insights queries (logs/insights), and answers. Where to look is this deployment's (LOGS_SWARM_NAME, LOGS_POOL_NAME, LOGS_WORKSPACE_SLUG, from Secrets or env); what it may read is the swarm's IAM role's. Input: { prompt, session? } (+ `job` on the launch to keep the fetched files for a follow-up). Output: { answer, cost }. Open to other struts (`peer: true`): strut/run-workflow { peer, workflow: \"logs\", input: { prompt } }. No built-ins, no shell, no caller MCP servers.",
  },
];

// Names this seeder USED to publish (seeding is additive — see retireWorkflows).
const RETIRED_WORKFLOWS: string[] = [];

/** One file per step under steps/, `<name>.ts` → `logs/<name>` (`_shared` is the helper the registry skips). */
export const SEED_STEPS = [
  "_shared",
  "context",
  "fetch-cloudwatch",
  "list-cloudwatch-groups",
  "fetch-quickwit",
  "search",
  "read",
  "insights",
] as const;
const RETIRED_STEPS: string[] = [];

const HERE = dirname(fileURLToPath(import.meta.url));
const CATEGORY = "logs";
export const STEPS_DIR = join(HERE, "steps");

export async function seedLogsSteps(workspace: WorkspaceStore): Promise<void> {
  for (const name of SEED_STEPS) {
    const type = `logs/${name}`;
    try {
      const code = await readFile(join(STEPS_DIR, `${name}.ts`), "utf-8");
      const { version, changed } = await workspace.publishStep(type, code, undefined, "logs-seed", SEED_OPTS);
      if (changed) console.log(`[logs] seeded step: ${type} @ ${version}`);
    } catch (err) {
      console.warn(`[logs] could not seed step "${type}":`, err instanceof Error ? err.message : err);
    }
  }
  await retireSteps(workspace, RETIRED_STEPS, "logs");
}

export async function seedLogsWorkflows(workspace: WorkspaceStore): Promise<void> {
  const dir = join(HERE, "workflows");
  for (const { name, description } of SEED_WORKFLOWS) {
    try {
      const yaml = await readFile(join(dir, `${name}.yaml`), "utf-8");
      const { version, changed } = await workspace.publishWorkflowByContent(name, yaml, description, CATEGORY, undefined, SEED_OPTS);
      if (changed) console.log(`[logs] seeded workflow: ${name} @ ${version}`);
    } catch (err) {
      console.warn(`[logs] could not seed workflow "${name}":`, err instanceof Error ? err.message : err);
    }
  }
  await retireWorkflows(workspace, RETIRED_WORKFLOWS, "logs");
}
