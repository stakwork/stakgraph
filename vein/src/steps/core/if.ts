import { z } from "zod";
import { defineStep } from "../../core.js";

const EXAMPLE = `- id: check
  type: if
  config:
    cond: "{{ fetch.body.status === 'active' }}"
  - id: yes
    type: log
    config:
      message: "Active!"
    depends: check
    when: true
  - id: no
    type: log
    config:
      message: "Inactive"
    depends: check
    when: false`;

export default defineStep({
  type: "if",
  description: `Conditional gate. Config: "cond" (template expression). Evaluates the condition and returns true or false. Downstream steps use "when: true" or "when: false" to branch.\n\n${EXAMPLE}`,
  input: z.object({
    cond: z.any(),
  }),
  output: z.boolean(),
  async run(cfg) {
    return Boolean(cfg.cond);
  },
});
