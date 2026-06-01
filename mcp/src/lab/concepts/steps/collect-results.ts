import { z, defineStep } from "vein";
import { addUsage, normalizeUsage } from "../../../aieo/src/usage.js";

/**
 * Merge the per-change results from a foreach into a single rollup:
 * union of modified concept ids + summed token usage. Output:
 * { modifiedConceptIds, usage, count }.
 */
export default defineStep({
  type: "concepts/collect-results",
  description: "Merge per-change results: union modifiedConceptIds + sum usage. Output: { modifiedConceptIds, usage, count }.",
  input: z.object({
    results: z.array(z.any()),
  }),
  output: z.any(),
  async run(cfg) {
    const ids = new Set<string>();
    let usage = normalizeUsage();
    for (const r of cfg.results ?? []) {
      for (const id of r?.modifiedConceptIds ?? []) ids.add(id);
      if (r?.usage) usage = normalizeUsage(addUsage(usage, r.usage));
    }
    return {
      modifiedConceptIds: [...ids],
      usage,
      count: (cfg.results ?? []).length,
    };
  },
});
