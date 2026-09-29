import { z, defineStep, withMedia } from "strut";
import { type Ctx, artifactsOf, browserOf, CAPTURE_VERBS, timeoutMs, viewport } from "./_shared.js";

export default defineStep({
  type: "browser/capture",
  description:
    "The declared shot in one call: open `url`, run `steps` in order, settle (+ `wait` ms), screenshot, and drain the page's errors. Each step is { do: <verb>, ...that verb's input } — do: goto | click | fill | type | press | select | hover | scroll | wait | text | evaluate | back, e.g. { do: \"click\", selector: \"text=Annual\" } or { do: \"wait\", text: \"Total\" }. A step whose target matches nothing fails the whole capture with the step named. Returns { path, url, width, height, bytes, errors, pageUrl, title } — the PNG under the run's artifacts, the console/network errors observed (empty when the page was clean), and where the page ended up after the steps. For a page you can script; an agent that must look and decide uses the atomic steps instead.",
  input: z.object({
    url: z.string().describe("http(s) URL to open"),
    steps: z
      .array(z.looseObject({ do: z.enum(CAPTURE_VERBS) }))
      .optional()
      .describe("ordered actions before the shot, each the input of the step it names plus `do`"),
    fullPage: z.boolean().optional().describe("shoot the whole scrollable page"),
    wait: z.number().int().nonnegative().optional().describe("extra ms to settle before the shot"),
    viewport,
    timeoutMs,
  }),
  output: z.object({
    path: z.string(),
    url: z.string(),
    width: z.number(),
    height: z.number(),
    bytes: z.number(),
    errors: z.array(z.string()),
    pageUrl: z.string().describe("the page's URL after the steps — not the artifact's"),
    title: z.string(),
  }),
  async run(cfg, ctx: Ctx) {
    const { small, ...out } = await browserOf(ctx).capture(ctx.runId, cfg, artifactsOf(ctx));
    // As an agent tool the model sees the 0.6× frame beside the JSON.
    return withMedia(out, [{ mediaType: "image/png", data: small, filename: out.path }]);
  },
});
