import { z, defineStep, type StepContext } from "strut";
import { logsOf } from "./_shared.js";

/**
 * Search the run's fetched logs — what the `/logs/agent` agent did with
 * `rg` / `grep` / `wc -l` in bash, without a shell: ripgrep with a fixed
 * argument list (the pattern after `--`), only inside the run's logs folder,
 * output capped and redacted (services.logs `search`).
 */
export default defineStep({
  type: "logs/search",
  description:
    "Search the log files fetched in this run (ripgrep: a regular expression unless fixed_strings). Prints file:line:text for each matching line — newest first within a CloudWatch file, as fetched. " +
    "Search every file, or name one. `context` adds lines around each match, `max_count` stops after that many matches per file, `count` prints only how many lines match. Output is capped; narrow the pattern or name a file when it is cut.",
  input: z.object({
    pattern: z.string().min(1).describe('What to find: a regular expression (e.g. "ERROR|FATAL", "timed out after \\d+"), or a plain string with fixed_strings'),
    file: z.string().optional().describe("One file from a fetch result (e.g. cw-swarms-38-….log); every fetched file when omitted"),
    ignore_case: z.boolean().optional().describe("Match regardless of case"),
    fixed_strings: z.boolean().optional().describe("Treat the pattern as a plain string, not a regex"),
    context: z.number().optional().describe("Lines of context before and after each match (0-20)"),
    max_count: z.number().optional().describe("Stop after this many matching lines per file"),
    count: z.boolean().optional().describe("Print only the number of matching lines per file"),
  }),
  output: z.string(),
  async run(cfg, ctx: StepContext) {
    return logsOf(ctx).search(ctx.runId, cfg);
  },
});
