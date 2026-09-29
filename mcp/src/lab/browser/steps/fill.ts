import { z, defineStep } from "strut";
import { type Ctx, browserOf, ok, oneOf, ONE_TARGET, target, timeoutMs } from "./_shared.js";

export default defineStep({
  type: "browser/fill",
  description:
    "Set an input, textarea or contenteditable's value: clears it, then types `value`. Target by `ref` (from browser/snapshot) or `selector`. For a plain keystroke sequence into whatever is focused use browser/type; to submit, browser/press Enter. Fails when the target matches nothing or is not editable.",
  input: z.object({ ...target, value: z.string(), timeoutMs }).refine(oneOf("ref", "selector"), ONE_TARGET),
  output: ok,
  async run(cfg, ctx: Ctx) {
    return browserOf(ctx).fill(ctx.runId, cfg);
  },
});
