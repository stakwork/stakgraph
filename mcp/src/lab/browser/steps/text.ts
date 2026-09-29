import { z, defineStep } from "strut";
import { type Ctx, atMostOneOf, browserOf, target, timeoutMs } from "./_shared.js";

export default defineStep({
  type: "browser/text",
  description:
    "The visible text (innerText) of the whole page, or of one element by `ref` or `selector` — capped at 50k characters. The cheap way to read content or compare a region between runs; browser/snapshot when you need the structure and the controls too.",
  input: z.object({ ...target, timeoutMs }).refine(atMostOneOf("ref", "selector"), { message: "give ref or selector, not both" }),
  output: z.object({ text: z.string() }),
  async run(cfg, ctx: Ctx) {
    return browserOf(ctx).text(ctx.runId, cfg);
  },
});
