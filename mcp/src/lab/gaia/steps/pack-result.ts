import { z, defineStep } from "vein";

/**
 * Trivial combiner: echoes its (already-template-resolved) config back as
 * the step output. Used as the final step of a small workflow that needs
 * to assemble fields from several earlier steps (task + agent output) into
 * one object, since workflow output = the last step's output.
 */
export default defineStep({
  type: "gaia/pack-result",
  description:
    "Echo the resolved config object back as output — a combiner for assembling fields from earlier steps into one object (workflow output = last step's output). Config: any JSON object. Output: the same object.",
  input: z.record(z.any()),
  output: z.any(),
  async run(cfg) {
    return cfg;
  },
});
