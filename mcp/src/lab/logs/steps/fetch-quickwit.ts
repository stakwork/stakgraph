import { z, defineStep, type StepContext } from "strut";
import { logsOf } from "./_shared.js";

/**
 * `/logs/agent`'s fetch_quickwit as a step (log/tools.ts
 * `fetchQuickwitText`, through services.logs): the app's own request logs,
 * from this swarm's Quickwit, saved to the run's logs folder. The
 * description and input are the tool's, word for word.
 */
export default defineStep({
  type: "logs/fetch-quickwit",
  description:
    "Fetch endpoint logs from the user's actual application (such as vercel logs) and save them to a local file for searching. Supports Lucene query syntax.",
  input: z.object({
    query: z
      .string()
      .describe(
        'Lucene query string (e.g. "level:error AND service:payments")'
      ),
    max_hits: z
      .number()
      .optional()
      .describe("Maximum number of results (default 1000)"),
    start_timestamp: z
      .number()
      .optional()
      .describe("Start timestamp (Unix epoch seconds)"),
    end_timestamp: z
      .number()
      .optional()
      .describe("End timestamp (Unix epoch seconds)"),
  }),
  output: z.string(),
  async run(cfg, ctx: StepContext) {
    return logsOf(ctx).fetchQuickwit(ctx.runId, cfg);
  },
});
