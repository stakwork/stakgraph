/**
 * Gitsee lab services bag — the HARNESS that the (future) QA tool-steps reach via
 * `ctx.services.gitsee.*`. Holds per-run BROWSER + STACK session managers and a
 * stateless VISION judge. The matching `onRunEnd(runId)` disposes a run's live
 * browser + booted stack — wired into vein's generic `services.onRunEnd` so
 * teardown is guaranteed on success AND error (the optimize-loop teardown fix).
 *
 * In-code only (constructed in `createLabVein`), NOT a seeded step — so the
 * tool-steps stay self-contained and swappable (real Playwright ↔ cassette;
 * staklink ↔ inline boot; swap the vision model) without touching the steps.
 */
import { BrowserManager } from "./browser.js";
import { StackManager } from "./stack.js";
import { buildVisionService, type VisionService } from "./vision.js";

export { BrowserManager, BrowserSession, summarizeObs } from "./browser.js";
export type { Observations } from "./browser.js";
export { StackManager, StackSession } from "./stack.js";
export type { StackOptions, BootResult } from "./stack.js";
export { buildVisionService } from "./vision.js";
export type { VisionService, VisionVerdict } from "./vision.js";

export interface GitseeServices {
  browser: BrowserManager;
  stack: StackManager;
  vision: VisionService;
}

/** Build the gitsee harness services + a disposer for a run's per-run sessions.
 *  The disposer is invoked by the lab's `onRunEnd(runId)`; stack teardown honors
 *  the session's own `keepUp` (set when the stack session was created). */
export function buildGitseeServices(): {
  gitsee: GitseeServices;
  disposeRun: (runId: string) => Promise<void>;
} {
  const browser = new BrowserManager();
  const stack = new StackManager();
  const vision = buildVisionService();

  return {
    gitsee: { browser, stack, vision },
    async disposeRun(runId) {
      // Close the browser first (cheap), then tear the stack down.
      await browser.dispose(runId).catch(() => {});
      await stack.dispose(runId).catch(() => {});
    },
  };
}
