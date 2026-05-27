import { z } from "zod";
import { defineStep } from "../../core.js";

export default defineStep({
  type: "wait",
  input: z.object({
    durationMs: z.number().int().min(0).default(1000),
    message: z.string().optional(),
  }),
  output: z.object({
    waited: z.number(),
    message: z.string().optional(),
  }),
  async run(cfg) {
    await new Promise((resolve) => setTimeout(resolve, cfg.durationMs));
    return { waited: cfg.durationMs, message: cfg.message };
  },
});
