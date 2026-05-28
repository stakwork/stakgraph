import { z } from "zod";
import { defineStep } from "../../core.js";

const EXAMPLE = `- id: child
  type: subflow
  config:
    workflow: notify-flow
    version: v2
    input:
      message: "{{ deploy.result }}"`;

export default defineStep({
  type: "subflow",
  description: `Run a published workflow. Config: "workflow" (name of a published workflow), optional "version" (defaults to active version), "input" (passed as the child's input).\n\n${EXAMPLE}`,
  input: z.object({
    workflow: z.string(),
    version: z.string().optional(),
    input: z.any(),
  }),
  output: z.any(),
  async run() {
    throw new Error("subflow step must be executed by the runner");
  },
});
