import { z, defineStep } from "strut";
import { type Ctx, browserOf, ok, timeoutMs } from "./_shared.js";

export default defineStep({
  type: "browser/type",
  description:
    "Type `text` with the keyboard into the FOCUSED element (click or fill a field first). Use browser/fill to replace a field's value; this appends keystrokes, which is what autocomplete and editors that ignore programmatic value changes need. `delay` is ms between keys.",
  input: z.object({ text: z.string(), delay: z.number().int().nonnegative().optional(), timeoutMs }),
  output: ok,
  async run(cfg, ctx: Ctx) {
    return browserOf(ctx).type(ctx.runId, cfg);
  },
});
