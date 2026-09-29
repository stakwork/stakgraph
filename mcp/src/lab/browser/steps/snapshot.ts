import { z, defineStep } from "strut";
import { type Ctx, browserOf } from "./_shared.js";

export default defineStep({
  type: "browser/snapshot",
  description:
    "The page's accessibility tree as text, every interactive element tagged [ref=eN]. This is how you SEE the page between screenshots — read it, pick a ref, act on it with click/fill/select/hover by `ref`. Refs are valid until the next navigation (goto, a click that navigates, back, press Enter on a form) — re-snapshot after any of those. Prefer this over a screenshot for reading text and finding controls; screenshot only when layout or visuals matter.",
  input: z.object({}),
  output: z.object({ url: z.string(), title: z.string(), tree: z.string() }),
  async run(_cfg, ctx: Ctx) {
    return browserOf(ctx).snapshot(ctx.runId);
  },
});
