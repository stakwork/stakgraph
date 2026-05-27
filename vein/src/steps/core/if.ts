import { z } from "zod";
import { defineStep } from "../../core.js";

/**
 * The `if` step is handled as a control flow primitive by the runner.
 * This definition exists for registry/type-checking purposes.
 * The runner intercepts `type: "if"` and handles branching directly.
 */
export default defineStep({
  type: "if",
  input: z.object({
    cond: z.any(),
    then: z.any(), // Step object
    else: z.any().optional(), // Step object
  }),
  output: z.any(),
  async run() {
    // Control flow handled by runner — this should never be called directly
    throw new Error("if step must be executed by the runner");
  },
});
