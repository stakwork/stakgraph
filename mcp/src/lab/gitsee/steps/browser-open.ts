import { z, defineStep } from "vein";
import { join } from "node:path";
import type { GitseeServices } from "../services/index.js";

/**
 * Open a path of the booted app in a real headless browser (the per-run session,
 * keyed by `ctx.runId`; created here on first use, pointed at the booted port).
 * Usable as a workflow step OR an agent tool. Output: a status/title string.
 */
export default defineStep({
  type: "gitsee/browser-open",
  description:
    "Open a path of the booted app in a real headless browser (default the app root). Returns the HTTP status + page title. Call after gitsee/boot, and after actions that should navigate. Requires a staged+booted stack.",
  input: z.object({ path: z.string().default("/").describe("path like /, /login, /dashboard") }),
  output: z.any(),
  async run(cfg, ctx) {
    const { gitsee } = ctx.services as { gitsee: GitseeServices };
    const stack = gitsee.stack.get(ctx.runId);
    if (!stack) return "no staged stack — run gitsee/stage-setup + gitsee/boot first";
    const base = `http://localhost:${stack.port}`;
    const browser = gitsee.browser.session(ctx.runId, base, join(stack.workspacePath, ".exercise"));
    browser.setBaseUrl(base);
    return browser.open(cfg.path ?? "/");
  },
});
