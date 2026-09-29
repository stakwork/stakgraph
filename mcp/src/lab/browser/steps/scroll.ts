import { z, defineStep } from "strut";
import { type Ctx, browserOf, ok, oneOf, target, timeoutMs } from "./_shared.js";

export default defineStep({
  type: "browser/scroll",
  description:
    "Scroll an element into view (by `ref` or `selector`) OR the page by `y` pixels (negative scrolls up). Use before a screenshot of a section below the fold, or to trigger lazy-loaded content; snapshot afterwards for what loaded.",
  input: z
    .object({ ...target, y: z.number().int().optional().describe("pixels to scroll the window by"), timeoutMs })
    .refine(oneOf("ref", "selector", "y"), { message: "give exactly one of ref, selector, or y" }),
  output: ok,
  async run(cfg, ctx: Ctx) {
    return browserOf(ctx).scroll(ctx.runId, cfg);
  },
});
