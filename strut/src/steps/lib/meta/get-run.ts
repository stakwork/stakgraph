import { z } from "zod";
import { defineStep } from "../../../core.js";
import { requireAuthoring } from "./_shared.js";

export default defineStep({
  type: "meta/get-run",
  description:
    "Get one run of an agent-authored workflow: its summary (input, output, status, error, duration) and event log. Events are slimmed by default (no payloads) to stay token-cheap; set fullEvents:true for each step's input/output. Use to debug why a candidate run failed. Runs of workflows the agent surface did not author are refused.",
  input: z.object({
    name: z.string().describe("Workflow name (must be agent-authored)"),
    runId: z.string().describe("Run id (from meta/list-runs or a meta/run-workflow result)"),
    fullEvents: z
      .boolean()
      .default(false)
      .describe("Include full per-step input/output payloads in events (default false: slimmed)."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    return requireAuthoring(ctx.services).getRun(cfg.name, cfg.runId, cfg.fullEvents);
  },
});
