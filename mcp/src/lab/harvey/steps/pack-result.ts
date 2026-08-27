import { z, defineStep } from "vein";

/**
 * Trivial combiner: echoes its (already-template-resolved) config back as
 * the step output. Used as the final step of a workflow that needs to
 * assemble fields from several earlier steps into one object (workflow
 * output = the last step's output), and as an `onError` fallback that packs
 * an explicit failure result instead of killing a batch run.
 */
export default defineStep({
  type: "harvey/pack-result",
  description:
    "Echo the resolved config object back as output — a combiner for assembling fields from earlier steps into one object (workflow output = last step's output). Config: any JSON object. Output: the same object.",
  input: z.record(z.any()),
  output: z.any(),
  async run(cfg) {
    return cfg;
  },
});
