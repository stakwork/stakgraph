import { z } from "zod";
import { defineStep } from "../../core.js";

const EXAMPLE = `- id: pause
  type: wait
  config:
    durationMs: 3000
    message: "Waiting for cooldown"`;

export default defineStep({
  type: "wait",
  description: `Pause for a duration. Output: { waited, message }.\n\n${EXAMPLE}`,
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
