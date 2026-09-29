import { z, defineStep } from "strut";
import { type Ctx, browserOf, oneOf, ONE_TARGET, target, timeoutMs } from "./_shared.js";

export default defineStep({
  type: "browser/click",
  description:
    "Click an element — by `ref` from the last browser/snapshot, or by `selector` when you know the page. Waits for the element, clicks, then settles. If the click navigated, refs are reset: snapshot again. Returns { url } (the page's URL afterwards — compare it to know whether you navigated). Fails when the target matches nothing.",
  input: z
    .object({
      ...target,
      button: z.enum(["left", "right", "middle"]).optional(),
      double: z.boolean().optional().describe("double-click"),
      timeoutMs,
    })
    .refine(oneOf("ref", "selector"), ONE_TARGET),
  output: z.object({ url: z.string() }),
  async run(cfg, ctx: Ctx) {
    return browserOf(ctx).click(ctx.runId, cfg);
  },
});
