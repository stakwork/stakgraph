import { z, defineStep } from "strut";
import { type Ctx, browserOf, ok } from "./_shared.js";

export default defineStep({
  type: "browser/close",
  description: "Close the run's browser page and context now. Optional — every run's context is closed when the run ends anyway. A later browser/open starts a fresh one.",
  input: z.object({}),
  output: ok,
  async run(_cfg, ctx: Ctx) {
    return browserOf(ctx).closeRun(ctx.runId);
  },
});
