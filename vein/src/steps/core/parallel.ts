import { z } from "zod";
import { defineStep } from "../../core.js";

/**
 * The `parallel` step is handled as a control flow primitive by the runner.
 * This definition exists for registry/type-checking purposes.
 * The runner intercepts `type: "parallel"` and runs branches concurrently.
 */
export default defineStep({
  type: "parallel",
  input: z.object({
    branches: z.record(z.any()), // Record<string, Flow>
  }),
  output: z.any(),
  async run() {
    // Control flow handled by runner — this should never be called directly
    throw new Error("parallel step must be executed by the runner");
  },
});
