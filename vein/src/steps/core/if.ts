import { z } from "zod";
import { defineStep } from "../../core.js";

const EXAMPLE = `- id: check
  type: if
  config:
    cond: "{{ fetch.body.status === 'active' }}"
    then:
      id: yes
      type: log
      config:
        message: "Active!"
    else:
      id: no
      type: log
      config:
        message: "Inactive"`;

export default defineStep({
  type: "if",
  description: `Conditional branch. Config: "cond" (template expression), "then" (single Step), optional "else" (single Step). NOT arrays — one step each.\n\n${EXAMPLE}`,
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
