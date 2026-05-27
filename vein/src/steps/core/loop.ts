import { z } from "zod";
import { defineStep } from "../../core.js";

/**
 * The `loop` step is handled as a control flow primitive by the runner.
 * This definition exists for registry/type-checking purposes.
 * The runner intercepts `type: "loop"` and handles iteration directly.
 */
export default defineStep({
  type: "loop",
  input: z.object({
    until: z.string(), // template expression
    maxIterations: z.number().int().positive(),
    delayMs: z.number().int().nonnegative().optional(),
    body: z.any(), // Step object
  }),
  output: z.any(),
  async run() {
    // Control flow handled by runner — this should never be called directly
    throw new Error("loop step must be executed by the runner");
  },
});
