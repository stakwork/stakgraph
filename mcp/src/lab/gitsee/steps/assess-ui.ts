import { z, defineStep } from "vein";
import type { GitseeServices } from "../services/index.js";

/**
 * Take a fresh full-page screenshot and have a vision model judge whether the
 * intended UI rendered and looks functional (vs blank/error/404/500),
 * considering the recent browser + server errors. The structured "did it work"
 * signal. Per-run browser + stack keyed by `ctx.runId`. Workflow step OR agent
 * tool.
 *
 * Output: { working, reason, screenshotPath, cost, usage }.
 */
export default defineStep({
  type: "gitsee/assess-ui",
  description:
    "Take a fresh full-page screenshot and have a vision model judge whether the intended UI rendered and looks functional (vs blank/error/404/500), considering recent browser + server errors. Use to confirm progress after fixes. Config: checkPath? (default /), model?. Output: { working, reason, screenshotPath, cost, usage }.",
  input: z.object({
    checkPath: z.string().default("/"),
    model: z.string().optional(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { gitsee } = ctx.services as { gitsee: GitseeServices };
    const browser = gitsee.browser.get(ctx.runId);
    const stack = gitsee.stack.get(ctx.runId);
    if (!browser) return { working: null, reason: "no page open — call gitsee/browser-open first" };
    const shot = await browser.screenshot();
    if (!shot) return { working: null, reason: `no screenshot (browser unavailable: ${browser.note})` };
    const logs = stack ? await stack.readLogs() : "";
    const port = stack?.port ?? 3000;
    try {
      const v = await gitsee.vision.assess(shot, `http://localhost:${port}${cfg.checkPath}`, browser.obs, logs, cfg.model);
      return { working: v.working, reason: v.reason, screenshotPath: shot, cost: v.cost, usage: v.usage };
    } catch (e) {
      return { working: null, reason: `vision assessment failed: ${(e as Error).message}`, screenshotPath: shot };
    }
  },
});
