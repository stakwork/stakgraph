import { z } from "zod";
import { defineStep } from "../../../core.js";
import { requireAuthoring } from "./_shared.js";

export default defineStep({
  type: "meta/list-runs",
  description:
    "List past runs of an agent-authored workflow (newest first) with status, duration, and timestamps — across ALL sessions, so earlier generations' candidate runs are inspectable. Then call meta/get-run for a specific run's detail. Run history of workflows the agent surface did not author is refused (their logs can record what graders were handed).",
  input: z.object({
    name: z.string().describe("Workflow name whose runs to list (must be agent-authored)"),
    limit: z.number().int().positive().default(20).describe("Max number of recent runs to return (default 20)."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    return requireAuthoring(ctx.services).listRuns(cfg.name, cfg.limit);
  },
});
