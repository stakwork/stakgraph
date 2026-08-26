import { z } from "zod";
import { defineStep, type StepContext } from "../../../core.js";
import type { VeinCapabilities } from "../../../capabilities.js";

const EXAMPLE = `- id: page
  type: html/extract
  config:
    url: "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
    headers:
      User-Agent: "vein research-agent contact@example.com"
    maxChars: 120000`;

/**
 * HTML → readable-text extraction over `html-to-text` (tolerant of malformed
 * markup; renders real text tables — financial filings live in tables).
 * LLM-authored adapter steps should compose this instead of hand-rolling
 * regex stripping (see AGENTS.md "step vs service").
 */

/** Grab <title> — html-to-text skips <head>, so pull it out separately. */
export function extractTitle(html: string): string | null {
  const m = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(html);
  if (!m) return null;
  const t = m[1]
    .replace(/&#x([0-9a-fA-F]+);/g, (_, h: string) => String.fromCodePoint(parseInt(h, 16)))
    .replace(/&#(\d+);/g, (_, d: string) => String.fromCodePoint(parseInt(d, 10)))
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&(apos|#39);/g, "'")
    .replace(/&nbsp;/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return t || null;
}

export default defineStep({
  type: "html/extract",
  description: `Fetch a web page (or take raw HTML) and extract readable text for LLM consumption — tables rendered as aligned text tables, scripts/styles/nav dropped, entities decoded, links kept as text. Give either url (fetched via the http capability, so runs are recordable) or html. Output: { text, title, length, truncated, status }. Set maxChars to bound output (default 100000); pass request headers when the site requires them (e.g. SEC EDGAR's User-Agent policy).\n\n${EXAMPLE}`,
  input: z
    .object({
      url: z.string().url().optional().describe("Page to fetch (via ctx.services.http)."),
      html: z.string().optional().describe("Raw HTML to extract from instead of fetching."),
      headers: z
        .record(z.string(), z.string())
        .optional()
        .describe("Extra request headers when fetching (e.g. a User-Agent for SEC EDGAR)."),
      maxChars: z
        .number()
        .int()
        .positive()
        .default(100_000)
        .describe("Truncate extracted text beyond this many characters (sets truncated: true)."),
      timeout: z.number().int().positive().optional().describe("Fetch timeout in ms."),
    })
    .refine((v) => Boolean(v.url) || typeof v.html === "string", {
      message: "html/extract needs either `url` or `html`",
    }),
  output: z.object({
    text: z.string(),
    title: z.string().nullable(),
    /** Full extracted length BEFORE truncation. */
    length: z.number(),
    truncated: z.boolean(),
    /** HTTP status when fetched; null when extracting from raw html. */
    status: z.number().nullable(),
  }),
  async run(cfg, ctx: StepContext<VeinCapabilities>) {
    let html = cfg.html;
    let status: number | null = null;

    if (cfg.url) {
      const http = ctx?.services?.http;
      if (!http) throw new Error("html/extract: http capability unavailable");
      const res = await http(cfg.url, {
        method: "GET",
        headers: cfg.headers,
        ...(cfg.timeout ? { timeout: cfg.timeout } : {}),
      });
      status = res.status;
      if (!res.ok) {
        throw new Error(`html/extract: GET ${cfg.url} failed with ${res.status}`);
      }
      // The http capability parses JSON bodies; anything else arrives as text.
      html = typeof res.body === "string" ? res.body : JSON.stringify(res.body);
    }

    // Lazy-load the dep inside run() so it's only pulled into memory when the
    // step executes — see AGENTS.md "Lib step dependency convention".
    const { convert } = await import("html-to-text");
    const text = convert(html ?? "", {
      wordwrap: false,
      selectors: [
        // Real text tables (filings/financials live in tables). rowSpan/colSpan
        // handled; cells padded into aligned columns.
        { selector: "table", format: "dataTable" },
        // Headings verbatim, not SHOUTED (the library default).
        ...[1, 2, 3, 4, 5, 6].map((n) => ({
          selector: `h${n}`,
          options: { uppercase: false },
        })),
        // Link/image noise off: keep anchor text, drop hrefs + images.
        { selector: "a", options: { ignoreHref: true } },
        { selector: "img", format: "skip" },
        { selector: "nav", format: "skip" },
      ],
    }).trim();

    const truncated = text.length > cfg.maxChars;
    return {
      text: truncated ? text.slice(0, cfg.maxChars) : text,
      title: extractTitle(html ?? ""),
      length: text.length,
      truncated,
      status,
    };
  },
});
