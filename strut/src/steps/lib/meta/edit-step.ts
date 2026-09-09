import { z } from "zod";
import { defineStep } from "../../../core.js";
import { requireAuthoring } from "./_shared.js";

export default defineStep({
  type: "meta/edit-step",
  description:
    "Publish a NEW VERSION of an EXISTING agent-authored custom step. Same self-contained source rules as meta/create-step; call meta/get-step first to read the current source. Identical content is a no-op; a change increments the version (v1 → v2 → …) with prior versions kept for rollback. Only steps the agent surface authored can be edited — built-ins and seeded steps are refused.",
  input: z.object({
    type: z.string().describe("Existing custom step type to edit, e.g. 'candidates/my-fetcher'."),
    code: z.string().describe("Full updated TypeScript source (same self-contained shape as meta/create-step)."),
    description: z.string().optional(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    return requireAuthoring(ctx.services).editStep(cfg.type, cfg.code, cfg.description);
  },
});
