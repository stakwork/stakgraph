import { z } from "zod";
import { defineStep } from "../../../core.js";
import { requireAuthoring } from "./_shared.js";

export default defineStep({
  type: "meta/search-runs",
  description:
    "Grep across the recent runs of an agent-authored workflow: match a regex against every event's JSON (inputs, outputs, errors) and get back (runId, event path, snippet) tuples plus a per-run frequency summary. The cross-run complement to meta/get-run — use it to answer 'which runs hit this, and how often?' (e.g. environment-gap signatures like 'command not found' or 'ModuleNotFoundError' — EVOLVE_SPEC §4.2), then meta/get-run to investigate one run. Note: tool outputs are truncated in the event log (~1500 chars), so a signature deep in long output can be missed. Run history of workflows the agent surface did not author is refused.",
  input: z.object({
    name: z.string().describe("Workflow name whose runs to search (must be agent-authored)"),
    pattern: z
      .string()
      .describe(
        "JavaScript regular expression to match against each event's JSON line, e.g. \"command not found|ModuleNotFoundError\".",
      ),
    runIds: z
      .array(z.string())
      .optional()
      .describe("Explicit run ids to search (e.g. one eval batch). Default: the newest runLimit runs."),
    runLimit: z
      .number()
      .int()
      .positive()
      .default(20)
      .describe("How many recent runs to scan when runIds is absent (default 20)."),
    maxMatches: z
      .number()
      .int()
      .positive()
      .default(50)
      .describe(
        "Cap on returned matches; scanning stops once reached (truncated: true). Narrow the pattern or run window rather than raising this (default 50).",
      ),
    ignoreCase: z.boolean().default(true).describe("Case-insensitive matching (default true)."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    return requireAuthoring(ctx.services).searchRuns(cfg.name, cfg.pattern, {
      runIds: cfg.runIds,
      runLimit: cfg.runLimit,
      maxMatches: cfg.maxMatches,
      ignoreCase: cfg.ignoreCase,
    });
  },
});
