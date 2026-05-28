import { z } from "zod";
import { defineStep } from "../../core.js";

const EXAMPLE = `- id: child
  type: subflow
  config:
    input: "{{ input.data }}"
    flow:
      name: child-flow
      steps:
        - id: process
          type: log
          config:
            message: "Processing {{ input }}"`;

export default defineStep({
  type: "subflow",
  description: `Run a nested workflow. Config: "flow" (inline Flow with name + steps), "input" (passed as the child's input).\n\n${EXAMPLE}`,
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
