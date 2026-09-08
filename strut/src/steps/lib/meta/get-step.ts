import { z } from "zod";
import { defineStep } from "../../../core.js";
import { requireAuthoring } from "./_shared.js";

export default defineStep({
  type: "meta/get-step",
  description:
    "Get a step type's details: input schema fields, description, and (for lib/custom steps) source code. Read before editing a step or wiring it into a workflow.",
  input: z.object({
    type: z.string().describe("Step type, e.g. 'http' or 'github/fetch-pr'"),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    return requireAuthoring(ctx.services).getStep(cfg.type);
  },
});
