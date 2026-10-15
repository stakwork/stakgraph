import { z, defineStep, withMedia } from "strut";
import { type Ctx, browserOf, CAPTURE_VERBS, shotSinkOf, timeoutMs, viewport } from "./_shared.js";

export default defineStep({
  type: "browser/verify",
  description:
    "A CHECK on a page, answering in a claim's check contract { supports, content, locator }: open `url`, run `steps` (browser/capture's list: { do: <verb>, ...that verb's input }), settle (+ `wait` ms), screenshot where the page got to, then test `expect` — `text` each visible, the `url` the page ended at (a glob: ** any, * within a segment; else a substring), each `selector` present, `noErrors` (no console error, page error, failed request or HTTP error since the page opened), `evaluate` truthy. " +
    "The page failing is a RESULT: a document that answered HTTP 400 or more, a step whose target matched nothing, an expectation that does not hold — supports: false, `failed` naming each, `content` one line saying where the page got to and what failed. Only the instrument failing throws: no browser, a url that does not open at all (DNS, refused, its own timeout), a selector or script that does not parse. " +
    "A `wait` step with no `timeoutMs` gives up after 30 s here. The shot is shots/NNN.png under the job's directory on a job's run (locator.url = /jobs/<job>/files/shots/NNN.png), else under the run's artifacts. " +
    "As a claim's check on a job: `url: \"{{ input.urls.pod }}/settings\"` reaches the pod the job holds. As a tool mid-turn it shows you what the check will see.",
  input: z.object({
    url: z.string().describe("http(s) URL to open — on a job's check usually {{ input.urls.pod }} plus a path"),
    steps: z
      .array(z.looseObject({ do: z.enum(CAPTURE_VERBS) }))
      .optional()
      .describe("ordered actions before the shot, each the input of the step it names plus `do` — browser/capture's list"),
    wait: z.number().int().nonnegative().optional().describe("extra ms to settle before the shot"),
    expect: z
      .object({
        text: z.array(z.string()).optional().describe("each visible on the page (case-insensitive, a substring of an element's text)"),
        url: z.string().optional().describe("where the page ended up: a glob (** any, * within a path segment), else a substring"),
        selector: z.array(z.string()).optional().describe("each matches at least one element (a Playwright selector: CSS, text=…, role=…)"),
        noErrors: z.boolean().optional().describe("no console error, page error, failed request or HTTP error since the page opened"),
        evaluate: z.string().optional().describe("JavaScript in the page; holds when the result is truthy"),
      })
      .optional()
      .describe("what must hold once the steps ran; none = the page opened without an HTTP error and every step ran"),
    viewport,
    timeoutMs,
  }),
  output: z.object({
    supports: z.boolean(),
    content: z.string(),
    locator: z.object({ path: z.string(), url: z.string() }).describe("the screenshot: path under the directory it was written to, and its served link"),
    errors: z.array(z.string()),
    pageUrl: z.string().describe("the page's URL after the steps — not the screenshot's"),
    title: z.string(),
    failed: z.array(z.string()),
  }),
  async run(cfg, ctx: Ctx) {
    const { small, path, url, supports, content, failed, errors, pageUrl, title } = await browserOf(ctx).verify(ctx.runId, cfg, shotSinkOf(ctx));
    // As an agent tool the model sees the 0.6× frame beside the verdict.
    return withMedia({ supports, content, locator: { path, url }, errors, pageUrl, title, failed }, [{ mediaType: "image/png", data: small, filename: path }]);
  },
});
