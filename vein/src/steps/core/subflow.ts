import { z } from "zod";
import { defineStep } from "../../core.js";

/**
 * The `subflow` step is handled as a control flow primitive by the runner.
 * This definition exists for registry/type-checking purposes.
 * The runner intercepts `type: "subflow"` and executes the child flow.
 */
export default defineStep({
  type: "subflow",
  input: z.object({
    flow: z.any(), // Flow object
    input: z.any(), // passed to the child flow
  }),
  output: z.any(),
  async run() {
    // Control flow handled by runner — this should never be called directly
    throw new Error("subflow step must be executed by the runner");
  },
});
