import { z, defineStep, type StepContext } from "strut";
import { cancelSignal, logsOf } from "./_shared.js";

/**
 * An agent's own CloudWatch Logs Insights query — the counting and grouping
 * the `/logs/agent` agent did with awk / sort / uniq in bash, run where the
 * logs are (log/cloudwatch.ts `runInsightsQuery`, through services.logs).
 * Rows inline (the first 200) and all of them in a file in the logs folder.
 * Read-only by construction; this swarm's IAM role decides which groups.
 */
export default defineStep({
  type: "logs/insights",
  description:
    "Run a CloudWatch Logs Insights query on one log group and return its rows — for counting, grouping and timing over many events without fetching them: " +
    '"stats count(*) by bin(5m)", "filter @message like /ERROR/ | stats count(*) by @logStream", "parse @message \\"took * ms\\" as ms | stats avg(ms), max(ms) by bin(1m)", "fields @timestamp, @message | filter @message like /OOM/ | sort @timestamp desc | limit 20". ' +
    "Shows the first 200 rows and saves every row to a file in the logs folder. A query without a limit returns up to 1000 rows.",
  input: z.object({
    log_group: z.string().describe("CloudWatch log group name (e.g. /swarms/38)"),
    query: z.string().min(1).describe("The Logs Insights query (fields / filter / parse / stats / sort / limit)"),
    minutes: z.number().optional().describe("How many minutes back to query (default 30)"),
  }),
  output: z.string(),
  async run(cfg, ctx: StepContext) {
    const cancel = cancelSignal(ctx);
    try {
      return await logsOf(ctx).insights(ctx.runId, cfg, cancel.signal);
    } finally {
      cancel.done();
    }
  },
});
