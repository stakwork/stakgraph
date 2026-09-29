import { z, defineStep } from "strut";
import { type Ctx, browserOf, navigated, timeoutMs, viewport } from "./_shared.js";

export default defineStep({
  type: "browser/open",
  description:
    "Start the run's browser page — the FIRST call in every browser workflow. Idempotent: a second call returns the page the run already has (viewport and session apply only the first time). Give `url` to navigate straight away; otherwise call browser/goto next. `storageStateSecret` names a secret whose value is a Playwright storage state (cookies + localStorage JSON, see browser/state) for a logged-in session — the value never enters the model. Returns { url, title, status }.",
  input: z.object({
    url: z.string().optional().describe("navigate here after opening (http/https)"),
    viewport,
    storageStateSecret: z.string().optional().describe("NAME of a secret holding a Playwright storageState JSON"),
    timeoutMs,
  }),
  output: navigated,
  async run(cfg, ctx: Ctx) {
    const browser = browserOf(ctx);
    let storageState: unknown;
    if (cfg.storageStateSecret) {
      const raw = await ctx.services.secrets?.get(cfg.storageStateSecret);
      if (!raw) throw new Error(`browser/open: secret ${cfg.storageStateSecret} is not set`);
      try {
        storageState = JSON.parse(raw);
      } catch {
        throw new Error(`browser/open: secret ${cfg.storageStateSecret} is not JSON (expected a Playwright storageState)`);
      }
    }
    return browser.open(ctx.runId, { url: cfg.url, viewport: cfg.viewport, storageState, timeoutMs: cfg.timeoutMs });
  },
});
