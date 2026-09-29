import { z, defineStep } from "strut";
import { type Ctx, browserOf, ok, oneOf, ONE_TARGET, target, timeoutMs } from "./_shared.js";

export default defineStep({
  type: "browser/select",
  description:
    "Choose an option in a <select> by its value or its label. Target the <select> itself by `ref` (from browser/snapshot) or `selector`. Custom dropdowns that are not a real <select> need click on the trigger, then click on the option.",
  input: z.object({ ...target, value: z.string().describe("option value or visible label"), timeoutMs }).refine(oneOf("ref", "selector"), ONE_TARGET),
  output: ok,
  async run(cfg, ctx: Ctx) {
    return browserOf(ctx).select(ctx.runId, cfg);
  },
});
