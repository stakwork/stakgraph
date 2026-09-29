import { z, defineStep } from "strut";
import { type Ctx, browserOf } from "./_shared.js";

export default defineStep({
  type: "browser/observe",
  description:
    "Drain the console errors, page (uncaught) errors, failed requests and 4xx/5xx responses accumulated since the last call. THE signal for 'renders but is broken' — a page can look fine while every API call 500s, and a blank shot usually explains itself here. Prefer it over another screenshot when something looks off. Empty lists when nothing happened.",
  input: z.object({}),
  output: z.object({
    console: z.array(z.string()),
    pageErrors: z.array(z.string()),
    failedRequests: z.array(z.string()),
    httpErrors: z.array(z.string()),
  }),
  async run(_cfg, ctx: Ctx) {
    return browserOf(ctx).observe(ctx.runId);
  },
});
