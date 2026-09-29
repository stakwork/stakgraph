import { z, defineStep } from "strut";
import { type Ctx, browserOf, oneOf, target, timeoutMs } from "./_shared.js";

export default defineStep({
  type: "browser/wait",
  description:
    "Hold until dynamic content is there: an element is visible (`ref` or `selector`), a `text` appears on the page, the `url` matches (a substring, or a glob with *), or `ms` pass. The verb for data that loads after the page does — call it before reading or shooting a region that fetches. Unlike other steps its cap is the run's remaining browser budget (BROWSER_RUN_BUDGET_MS), so it can outlast a slow fetch; set `timeoutMs` to fail sooner. Returns { ok, waitedMs }.",
  input: z
    .object({
      ...target,
      text: z.string().optional().describe("wait until this text is visible on the page"),
      url: z.string().optional().describe("wait until the URL contains this, or matches this glob"),
      ms: z.number().int().nonnegative().optional().describe("just wait this many ms"),
      timeoutMs,
    })
    .refine(oneOf("ref", "selector", "text", "url", "ms"), { message: "give exactly one of ref, selector, text, url, or ms" }),
  output: z.object({ ok: z.literal(true), waitedMs: z.number() }),
  async run(cfg, ctx: Ctx) {
    return browserOf(ctx).wait(ctx.runId, cfg);
  },
});
