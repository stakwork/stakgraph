import { z, defineStep, type StepContext } from "strut";
import { logsOf } from "./_shared.js";

/**
 * Read lines of one fetched log file — what the `/logs/agent` agent did
 * with `head` / `tail` / `sed -n` in bash: a numbered range, at most 500
 * lines, only inside the run's logs folder, capped and redacted
 * (services.logs `read`).
 */
export default defineStep({
  type: "logs/read",
  description:
    "Read lines of one log file fetched in this run, numbered (as logs_search numbers them). Lines `from` to `to` (1-based, inclusive): 200 lines from `from` by default, at most 500 per call. " +
    "Use it for what surrounds a match, or the first lines of a file (a CloudWatch file is newest first).",
  input: z.object({
    file: z.string().describe("The file, as a fetch result or logs_search names it"),
    from: z.number().optional().describe("First line (default 1)"),
    to: z.number().optional().describe("Last line (default from + 199)"),
  }),
  output: z.string(),
  async run(cfg, ctx: StepContext) {
    return logsOf(ctx).read(ctx.runId, cfg);
  },
});
