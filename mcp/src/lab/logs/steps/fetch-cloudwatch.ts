import { z, defineStep, type StepContext } from "strut";
import { cancelSignal, logsOf } from "./_shared.js";

/**
 * `/logs/agent`'s fetch_cloudwatch as a step (log/tools.ts
 * `fetchCloudwatchText`, through services.logs): Logs Insights, newest
 * first, long lines recovered through GetLogRecord, saved to the run's logs
 * folder. The description and input are the tool's, word for word — the
 * lab's logs tests fail when they drift apart.
 */
export default defineStep({
  type: "logs/fetch-cloudwatch",
  description:
    'Fetch logs from AWS CloudWatch using Logs Insights and save them to a local file for searching. Results are returned **newest-first**. Plain string filter patterns are supported (e.g. "ERROR"); JSON-style patterns (e.g. "{ $.statusCode = 500 }") are **not** supported — use a plain string instead.',
  input: z.object({
    log_group: z
      .string()
      .describe(
        "CloudWatch log group name (e.g. /swarms/38)"
      ),
    log_stream_names: z
      .array(z.string())
      .optional()
      .describe(
        'Filter to specific log streams (e.g. ["stakgraph.sphinx", "boltwall.sphinx"])'
      ),
    filter_pattern: z
      .string()
      .optional()
      .describe(
        'Plain string filter pattern (e.g. "ERROR", "147426952"). JSON-style patterns (e.g. "{ $.statusCode = 500 }") are not supported.'
      ),
    minutes: z
      .number()
      .optional()
      .describe("How many minutes back to fetch (default 30)"),
    limit: z
      .number()
      .optional()
      .describe("Max number of log events to fetch (default 10000)"),
  }),
  output: z.string(),
  async run(cfg, ctx: StepContext) {
    const cancel = cancelSignal(ctx);
    try {
      return await logsOf(ctx).fetchCloudwatch(ctx.runId, cfg, cancel.signal);
    } finally {
      cancel.done();
    }
  },
});
