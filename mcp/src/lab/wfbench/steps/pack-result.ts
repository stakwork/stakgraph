import { z, defineStep } from "vein";

/**
 * Trivial combiner: echoes its (already-template-resolved) config back as
 * the step output — the final "assemble one object from many steps" step and
 * the `onError` fallback that packs an explicit failure instead of killing
 * the run (same as harvey/pack-result).
 */
export default defineStep({
  type: "wfbench/pack-result",
  description:
    "Echo the resolved config object back as output — a combiner for assembling fields from earlier steps into one object, or an onError fallback. Config: any JSON object. Output: the same object.",
  input: z.record(z.string(), z.any()),
  output: z.any(),
  async run(cfg) {
    return cfg;
  },
});
