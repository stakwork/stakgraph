import { z, defineStep } from "vein";
import type { GitseeServices } from "../services/index.js";

/**
 * Drain the browser console errors, failed requests, and 4xx/5xx API responses
 * accumulated since the last call — THE key "renders but is broken" signal a
 * screenshot can't show. Per-run session keyed by `ctx.runId`. Workflow step OR
 * agent tool.
 */
export default defineStep({
  type: "gitsee/browser-observe",
  description:
    "Drain the browser console errors, failed requests, and 4xx/5xx API responses accumulated since the last call. THE key signal for 'renders but is broken' — a page can look fine while every API call 500s.",
  input: z.object({}),
  output: z.any(),
  async run(_cfg, ctx) {
    const { gitsee } = ctx.services as { gitsee: GitseeServices };
    const browser = gitsee.browser.get(ctx.runId);
    if (!browser) return "no page open — call gitsee/browser-open first";
    return browser.drainSummary();
  },
});
