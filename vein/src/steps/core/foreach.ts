import { z } from "zod";
import { defineStep } from "../../core.js";

const EXAMPLE = `- id: process_changes
  type: foreach
  config:
    items: "{{ fetch.changes }}"
    body:
      id: handle
      type: subflow
      config:
        workflow: process-pr
        input:
          change: "{{ $current }}"
          index: "{{ $index }}"`;

export default defineStep({
  type: "foreach",
  description: `Iterate over a list, running "body" once per item. Config: "items" (template expression evaluating to an array), "body" (single Step). Inside body: "$current" is the current item, "$index" is the zero-based position. Output is the array of body results, one per item, in order.\n\n${EXAMPLE}`,
  input: z.object({
    items: z.any(), // template expression → array (resolved per-iteration scope)
    body: z.any(), // Step object
    maxIterations: z.number().int().positive().optional(), // optional safety cap
  }),
  output: z.any(),
  async run() {
    // Control flow handled by runner — this should never be called directly
    throw new Error("foreach step must be executed by the runner");
  },
});
