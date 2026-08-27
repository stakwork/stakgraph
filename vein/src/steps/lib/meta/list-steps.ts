import { z } from "zod";
import { defineStep } from "../../../core.js";
import { requireAuthoring } from "./_shared.js";

export default defineStep({
  type: "meta/list-steps",
  description:
    "List available step types like a filesystem. Valid paths: 'steps' (shows core/, lib/, custom/), 'steps/core', 'steps/lib', 'steps/lib/<namespace>', 'steps/custom'. Use before authoring to see what already exists.",
  input: z.object({
    path: z
      .string()
      .default("steps")
      .describe(
        "Path to list. Defaults to 'steps' (the root). Use 'steps/lib' to see lib namespaces, 'steps/custom' for workspace custom steps.",
      ),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    return requireAuthoring(ctx.services).listSteps(cfg.path);
  },
});
