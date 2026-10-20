import { z, defineStep, type StepContext } from "strut";
import { logsOf } from "./_shared.js";

/**
 * The logs agent's prompt and clock — the `/logs/agent` handler's prompt
 * assembly (log/context.ts) as a step, before the agent. Where to look is
 * THIS deployment's: its swarm name (→ the `/swarms/<id>` log group and its
 * streams), its pool (→ the workspace pods' group), its Hive workspace (→
 * the app's Quickwit logs), read through the secrets capability — set them
 * under Secrets or in env. A caller, who may be another strut, names
 * nothing; with none set the agent finds its groups with
 * logs/list-cloudwatch-groups. Starts this run's time budget and its logs
 * folder for every logs/* tool after it.
 */
export default defineStep({
  type: "logs/context",
  description:
    "Start a logs investigation: this run's logs folder (under `dir`) and time budget, and the agent's prompt — the question with where to look on this deployment: " +
    "its swarm's CloudWatch group and streams (LOGS_SWARM_NAME, e.g. swarm38), its pool's workspace pods (LOGS_POOL_NAME), its app's Quickwit logs (LOGS_WORKSPACE_SLUG), each read from Secrets or env and left out when unset. " +
    "The logs/* tools only work in a run that ran this first. Output: { prompt, logsDir }",
  input: z.object({
    prompt: z.string().min(1).describe("The question about the logs"),
    dir: z.string().describe("Where the run keeps its files (job/dir's path); fetched logs go in its logs/ folder"),
    budgetMinutes: z.number().positive().optional().describe("The agent's wall-clock budget, shown on every tool result (default 8)"),
  }),
  output: z.object({ prompt: z.string(), logsDir: z.string() }),
  async run(cfg, ctx: StepContext) {
    const secrets = (ctx.services as { secrets?: { get(name: string): Promise<string | undefined> } } | undefined)?.secrets;
    const read = async (name: string) => ((await secrets?.get(name)) ?? "").trim() || undefined;
    return logsOf(ctx).begin(ctx.runId, {
      dir: cfg.dir,
      prompt: cfg.prompt,
      swarmName: await read("LOGS_SWARM_NAME"),
      poolName: await read("LOGS_POOL_NAME"),
      workspaceSlug: await read("LOGS_WORKSPACE_SLUG"),
      ...(cfg.budgetMinutes ? { budgetMinutes: cfg.budgetMinutes } : {}),
    });
  },
});
