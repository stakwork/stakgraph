import { z, defineStep } from "vein";
import type { GitseeServices } from "../services/index.js";

/** Press a keyboard key on the page (e.g. Enter to submit a focused form).
 *  Per-run browser session keyed by `ctx.runId`. Workflow step OR agent tool. */
export default defineStep({
  type: "gitsee/browser-press",
  description: "Press a keyboard key on the page (e.g. Enter to submit a focused form).",
  input: z.object({ key: z.string().describe("a key name like Enter, Tab, Escape") }),
  output: z.any(),
  async run(cfg, ctx) {
    const { gitsee } = ctx.services as { gitsee: GitseeServices };
    const browser = gitsee.browser.get(ctx.runId);
    if (!browser) return "no page open — call gitsee/browser-open first";
    return browser.press(cfg.key);
  },
});
