import { z, defineStep } from "strut";
import { type Ctx, browserOf, navigated, timeoutMs } from "./_shared.js";

export default defineStep({
  type: "browser/goto",
  description:
    "Navigate the run's page to `url` and settle (network idle, fonts, two frames). RESETS every ref — call browser/snapshot again before clicking by ref. Refused for file: and link-local/metadata addresses. Returns { url, title, status }; a 404/500 document is still a successful navigation (check `status`). Needs browser/open first.",
  input: z.object({
    url: z.string().describe("http(s) URL"),
    wait: z.number().int().nonnegative().optional().describe("extra ms to wait after settling, for slow pages"),
    timeoutMs,
  }),
  output: navigated,
  async run(cfg, ctx: Ctx) {
    return browserOf(ctx).goto(ctx.runId, cfg);
  },
});
