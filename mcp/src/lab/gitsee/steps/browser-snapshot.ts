import { z, defineStep } from "vein";
import type { GitseeServices } from "../services/index.js";

/**
 * List the currently-visible interactive elements (links, buttons, inputs…) with
 * `@eN` refs to click/fill. Refs reset on every navigation. Per-run browser
 * session keyed by `ctx.runId`. Usable as a workflow step OR an agent tool.
 */
export default defineStep({
  type: "gitsee/browser-snapshot",
  description:
    "List the currently visible interactive elements (links, buttons, inputs…) with @eN refs to click/fill. Refs reset on every navigation — re-snapshot after anything that changes the page. Requires gitsee/browser-open first.",
  input: z.object({}),
  output: z.any(),
  async run(_cfg, ctx) {
    const { gitsee } = ctx.services as { gitsee: GitseeServices };
    const browser = gitsee.browser.get(ctx.runId);
    if (!browser) return "no page open — call gitsee/browser-open first";
    return browser.snapshot();
  },
});
