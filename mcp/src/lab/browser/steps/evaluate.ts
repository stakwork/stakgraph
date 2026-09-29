import { z, defineStep } from "strut";
import { type Ctx, browserOf, timeoutMs } from "./_shared.js";

export default defineStep({
  type: "browser/evaluate",
  description:
    "Run a JavaScript expression IN THE PAGE and return its JSON-serializable result (capped at 50k characters; `truncated: true` when cut). For what the tree and text cannot tell you: document.title, location.href, a computed style, an element count, window state. Not for actions the other steps do. Returns { result }.",
  input: z.object({ expression: z.string().describe("e.g. document.querySelectorAll('tr').length"), timeoutMs }),
  output: z.object({ result: z.any(), truncated: z.literal(true).optional() }),
  async run(cfg, ctx: Ctx) {
    return browserOf(ctx).evaluate(ctx.runId, cfg);
  },
});
