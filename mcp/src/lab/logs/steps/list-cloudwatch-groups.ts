import { z, defineStep, type StepContext } from "strut";
import { logsOf } from "./_shared.js";

/**
 * `/logs/agent`'s list_cloudwatch_groups as a step (log/tools.ts
 * `listCloudwatchGroupsText`, through services.logs). What it lists is what
 * this swarm's IAM role may see. The description and input are the tool's,
 * word for word.
 */
export default defineStep({
  type: "logs/list-cloudwatch-groups",
  description:
    "List available CloudWatch log groups. Use this to discover which log groups exist before fetching logs.",
  input: z.object({
    prefix: z
      .string()
      .optional()
      .describe(
        "Optional prefix to filter log groups (e.g. /ecs/ or /aws/lambda/)"
      ),
  }),
  output: z.string(),
  async run(cfg, ctx: StepContext) {
    return logsOf(ctx).listGroups(ctx.runId, cfg);
  },
});
