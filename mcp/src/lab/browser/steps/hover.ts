import { z, defineStep } from "strut";
import { type Ctx, browserOf, ok, oneOf, ONE_TARGET, target, timeoutMs } from "./_shared.js";

export default defineStep({
  type: "browser/hover",
  description: "Move the mouse over an element (tooltips, hover menus, revealed controls). Target by `ref` or `selector`; snapshot afterwards to see what appeared.",
  input: z.object({ ...target, timeoutMs }).refine(oneOf("ref", "selector"), ONE_TARGET),
  output: ok,
  async run(cfg, ctx: Ctx) {
    return browserOf(ctx).hover(ctx.runId, cfg);
  },
});
