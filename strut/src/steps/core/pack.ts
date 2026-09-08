import { z } from "zod";
import { defineStep } from "../../core.js";

const EXAMPLE = `- id: result
  type: pack
  depends: [fetch, score]
  config:
    repo: "{{ input.repo }}"
    stars: "{{ fetch.body.stargazers_count }}"
    verdict: "{{ score.all_pass ? 'pass' : 'fail' }}"`;

/**
 * Assemble an object from other steps' outputs. A workflow's output is its
 * LAST step's output, so every workflow that must return more than one
 * step's result needs a step whose only job is to pack fields together —
 * and an `onError` fallback that packs an explicit failure shape instead
 * of killing the run wants the same primitive. The config IS the output:
 * every field is template-resolved by the runner before `run` sees it.
 */
export default defineStep({
  type: "pack",
  description: `Assemble an object from templates: the resolved config object IS the output. Use it as a workflow's last step to return fields from several earlier steps, or as an onError fallback that packs an explicit failure shape. Output: the config object.\n\n${EXAMPLE}`,
  input: z.record(z.string(), z.any()),
  output: z.any(),
  async run(cfg) {
    return cfg;
  },
});
