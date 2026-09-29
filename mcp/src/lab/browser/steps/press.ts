import { z, defineStep } from "strut";
import { type Ctx, browserOf, timeoutMs } from "./_shared.js";

export default defineStep({
  type: "browser/press",
  description:
    "Press a key on the page: Enter (submit the focused form), Escape (close a dialog), Tab, ArrowDown, Control+a, Shift+Enter, … Settles afterwards; if it navigated (Enter on a form), refs are reset — snapshot again. Returns { url }.",
  input: z.object({ key: z.string().describe("a Playwright key name, e.g. Enter, Escape, Control+a"), timeoutMs }),
  output: z.object({ url: z.string() }),
  async run(cfg, ctx: Ctx) {
    return browserOf(ctx).press(ctx.runId, cfg);
  },
});
