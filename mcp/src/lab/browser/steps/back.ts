import { z, defineStep } from "strut";
import { type Ctx, browserOf, timeoutMs } from "./_shared.js";

export default defineStep({
  type: "browser/back",
  description: "History back, then settle. Resets refs — snapshot again. Returns { url, title }.",
  input: z.object({ timeoutMs }),
  output: z.object({ url: z.string(), title: z.string() }),
  async run(cfg, ctx: Ctx) {
    return browserOf(ctx).back(ctx.runId, cfg);
  },
});
