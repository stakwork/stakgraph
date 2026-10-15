import { z, defineStep, withMedia } from "strut";
import { type Ctx, browserOf, CAPTURE_VERBS, shotSinkOf, timeoutMs, viewport } from "./_shared.js";

export default defineStep({
  type: "browser/walkthrough",
  description:
    "browser/verify with a FRAME PER STEP — a recording a person scrubs through, for a feature that is a flow (sign up → confirm → land on the dashboard), not one page. Most changes are one page: use browser/verify. " +
    "Open `url` (frame 1, captioned `caption`), run `steps` (browser/capture's list: { do: <verb>, ...that verb's input }) and after each step that has a `caption` take a frame; then test `expect` exactly as browser/verify does and answer in a claim's check contract { supports, content, locator, failed }. " +
    "Caption only the steps a person should see — the page after a click, the form filled — never a wait. A recording is a few frames; a longer flow is several walkthroughs, one per claim. Explore with the atomic browser/* steps first, then write the path that works here: the recording replays only that. " +
    "Like browser/verify it runs in a browser of its OWN, signed out with empty storage — nothing an earlier recording signed into — so a flow behind a sign-in signs in in `steps`, and the list is checked before anything opens (a wrong one starts no recording). " +
    "A step that fails, or an expectation that does not hold, is a result: supports: false, the recording ends on a frame of where the page got to, marked with why. " +
    "The recording is a directory, walkthroughs/NNN/ (frames 01.png, 02.png, … and walkthrough.json), under the job's directory on a job's run; `locator` is its manifest. To hand it to the person, list it in `artifacts` by that path with kind \"walkthrough\". As a claim's check its evidence is the recording. You see the last frame; open any other with job/read.",
  input: z.object({
    url: z.string().describe("http(s) URL to open — on a job's check usually {{ input.urls.pod }} plus a path"),
    title: z.string().optional().describe("what the recording shows, for the person: \"Switch billing to annual\""),
    caption: z.string().optional().describe("the opening frame's caption (default: the page's title)"),
    steps: z
      .array(z.looseObject({ do: z.enum(CAPTURE_VERBS), caption: z.string().optional().describe("take a frame after this step, captioned so") }))
      .optional()
      .describe("ordered actions, each the input of the step it names plus `do`, and `caption` on the ones a person should see"),
    wait: z.number().int().nonnegative().optional().describe("extra ms to settle before each frame"),
    expect: z
      .object({
        text: z.array(z.string()).optional().describe("each visible on the page (case-insensitive, a substring of an element's text)"),
        url: z.string().optional().describe("where the page ended up: a glob (** any, * within a path segment), else a substring"),
        selector: z.array(z.string()).optional().describe("each matches at least one element (a Playwright selector: CSS, text=…, role=…)"),
        noErrors: z.boolean().optional().describe("no console error, page error, failed request or HTTP error since the page opened"),
        evaluate: z.string().optional().describe("JavaScript in the page; holds when the result is truthy"),
      })
      .optional()
      .describe("what must hold once the steps ran — browser/verify's; none = the page opened without an HTTP error and every step ran"),
    viewport,
    timeoutMs,
  }),
  output: z.object({
    supports: z.boolean(),
    content: z.string(),
    locator: z.object({ path: z.string(), url: z.string() }).describe("the recording's manifest: walkthroughs/NNN/walkthrough.json, and its served link"),
    kind: z.literal("walkthrough").describe("the artifact kind to list it under"),
    frames: z.array(z.object({ file: z.string(), caption: z.string(), url: z.string() })),
    errors: z.array(z.string()),
    pageUrl: z.string().describe("the page's URL after the steps"),
    title: z.string(),
    failed: z.array(z.string()),
  }),
  async run(cfg, ctx: Ctx) {
    const { small, path, url, frames, supports, content, failed, errors, pageUrl, title } = await browserOf(ctx).walkthrough(ctx.runId, cfg, shotSinkOf(ctx));
    // As an agent tool the model sees the last frame, 0.6×, beside the verdict.
    const last = frames[frames.length - 1]!;
    return withMedia(
      { supports, content, locator: { path, url }, kind: "walkthrough" as const, frames, errors, pageUrl, title, failed },
      [{ mediaType: "image/png", data: small, filename: last.file }],
    );
  },
});
