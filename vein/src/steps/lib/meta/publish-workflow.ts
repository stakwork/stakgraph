import { z } from "zod";
import { defineStep } from "../../../core.js";
import { requireAuthoring } from "./_shared.js";

export default defineStep({
  type: "meta/publish-workflow",
  description:
    "Publish a workflow from YAML — an explicit UPSERT: a new name creates v1, an existing agent-authored name gets the next version (identical content is a no-op; prior versions are kept for rollback). Everything published here is stamped publisher 'ai' — that stamp is what later allows meta/run-workflow and meta/get-run on it. Publishing over a workflow the agent surface did NOT author is refused: author candidates under new names.",
  input: z.object({
    name: z.string().describe("Workflow name (kebab-case)"),
    yaml: z.string().describe("Full workflow YAML"),
    description: z.string().optional(),
    category: z
      .string()
      .optional()
      .describe("Optional sidebar grouping label (e.g. an experiment name). Omit to leave unchanged."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    return requireAuthoring(ctx.services).publishWorkflow(
      cfg.name,
      cfg.yaml,
      cfg.description,
      cfg.category,
    );
  },
});
