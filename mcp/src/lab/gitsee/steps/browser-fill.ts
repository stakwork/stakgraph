import { z, defineStep } from "vein";
import type { GitseeServices } from "../services/index.js";

/** Type text into an input/textarea by its `@eN` ref (clears first). Per-run
 *  browser session keyed by `ctx.runId`. Usable as a workflow step OR agent tool. */
export default defineStep({
  type: "gitsee/browser-fill",
  description: "Type text into an input/textarea by its @eN ref (clears first). Use to fill login forms etc.",
  input: z.object({ ref: z.string(), text: z.string() }),
  output: z.any(),
  async run(cfg, ctx) {
    const { gitsee } = ctx.services as { gitsee: GitseeServices };
    const browser = gitsee.browser.get(ctx.runId);
    if (!browser) return "no page open — call gitsee/browser-open first";
    return browser.fill(cfg.ref.replace(/^@/, ""), cfg.text);
  },
});
