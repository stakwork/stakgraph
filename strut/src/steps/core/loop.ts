import { z } from "zod";
import { defineStep } from "../../core.js";

const EXAMPLE = `- id: poll
  type: loop
  config:
    maxIterations: 10
    delayMs: 2000
    until: "{{ $current.body.status === 'complete' }}"
    body:
      id: check
      type: http
      config:
        url: "https://api.example.com/status"`;

export default defineStep({
  type: "loop",
  description: `Repeat a step until a condition is met. Config: "body" (single Step), "until" (template expression, use $current for last output), "maxIterations", optional "delayMs".\n\n${EXAMPLE}`,
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
