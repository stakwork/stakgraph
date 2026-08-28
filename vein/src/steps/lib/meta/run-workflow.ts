import { z } from "zod";
import { defineStep } from "../../../core.js";
import { requireAuthoring } from "./_shared.js";

export default defineStep({
  type: "meta/run-workflow",
  description:
    "Run an agent-authored workflow (publisher 'ai' — i.e. published via meta/publish-workflow) with a given input, awaiting the result: { runId, status, output?, error? }. The candidate runs as its OWN persisted run (inspect it with meta/get-run). Sees steps and workflows published earlier in this same run (the registry is re-read fresh). Workflows the agent surface did not author are refused.",
  input: z.object({
    name: z.string().describe("Workflow name to run"),
    input: z
      .any()
      .optional()
      .describe(
        "Input passed to the workflow as a JSON OBJECT (not a string), referenced in its steps via {{ input.* }}. Use {} if none.",
      ),
    params: z
      .record(z.any())
      .optional()
      .describe(
        "Optional overrides for the workflow's `params` knobs (prompts, thresholds), shallow-merged over its defaults — those are runs, not versions.",
      ),
    version: z.string().optional().describe("Optional specific version. Defaults to the active version."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    // parentRunId links the child run's controller under this run's — so
    // cancelling/pausing this run reaches the children it launched
    // (RUN_CONTROL_SPEC §2.2 tree linkage).
    return requireAuthoring(ctx.services).runWorkflow(cfg.name, cfg.input, cfg.params, cfg.version, {
      parentRunId: ctx.runId,
    });
  },
});
