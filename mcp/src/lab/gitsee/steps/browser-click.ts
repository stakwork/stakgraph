import { z, defineStep } from "vein";
import type { GitseeServices } from "../services/index.js";

/** Click an element by its `@eN` ref from the latest snapshot. Per-run browser
 *  session keyed by `ctx.runId`. Usable as a workflow step OR an agent tool. */
export default defineStep({
  type: "gitsee/browser-click",
  description: "Click an element by its @eN ref from the latest gitsee/browser-snapshot.",
  input: z.object({ ref: z.string().describe("an @eN ref, e.g. e3") }),
  output: z.any(),
  async run(cfg, ctx) {
    const { gitsee } = ctx.services as { gitsee: GitseeServices };
    const browser = gitsee.browser.get(ctx.runId);
    if (!browser) return "no page open — call gitsee/browser-open first";
    return browser.click(cfg.ref.replace(/^@/, ""));
  },
});
