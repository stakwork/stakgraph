import { z, defineStep, withMedia } from "strut";
import { type Ctx, artifactsOf, atMostOneOf, browserOf, target, timeoutMs } from "./_shared.js";

export default defineStep({
  type: "browser/screenshot",
  description:
    "Photograph the page (the viewport; `fullPage` for the whole scroll height) or one element (`ref` or `selector`). The PNG is saved under the run's artifacts as shots/NNN.png (or shots/<name>.png) and { path, url, width, height, bytes } is returned — `path` for a later step, `url` as a link — and, when you are an agent, you SEE the frame with the result. Costs the most of any step to look at: use it when layout or visuals matter, never to re-check a page you have not changed — browser/snapshot and browser/text are cheaper reads.",
  input: z
    .object({
      ...target,
      fullPage: z.boolean().optional().describe("the whole scrollable page, not just the viewport"),
      name: z.string().optional().describe("file name under shots/ (without .png); default a counter 001, 002, …"),
      timeoutMs,
    })
    .refine(atMostOneOf("ref", "selector"), { message: "give ref or selector, not both" }),
  output: z.object({ path: z.string(), url: z.string(), width: z.number(), height: z.number(), bytes: z.number() }),
  async run(cfg, ctx: Ctx) {
    const { small, ...shot } = await browserOf(ctx).screenshot(ctx.runId, cfg, artifactsOf(ctx));
    // The model's copy: as a tool the 0.6× frame rides beside the JSON
    // as a file part; templates, the event log and run.json see only `shot`.
    return withMedia(shot, [{ mediaType: "image/png", data: small, filename: shot.path }]);
  },
});
