import { z, defineStep } from "strut";
import { type Ctx, artifactsOf, browserOf } from "./_shared.js";

export default defineStep({
  type: "browser/state",
  description:
    "Write the page's current storage state (cookies + localStorage) to state.json under the run's artifacts, for a human to paste under Secrets once and load in later runs with browser/open { storageStateSecret }. Returns { path, url }. The contents are credentials: never read them into the conversation.",
  input: z.object({}),
  output: z.object({ path: z.string(), url: z.string() }),
  async run(_cfg, ctx: Ctx) {
    return browserOf(ctx).state(ctx.runId, artifactsOf(ctx));
  },
});
