import { tool } from "ai";
import { z } from "zod";
import { createAnthropic } from "@ai-sdk/anthropic";
import { Provider, getGatewayBaseURL, normalizeApiKey } from "./provider.js";

/**
 * Web search, uniform across providers.
 *
 * Anthropic ships a server-executed `web_search` tool: the search loop
 * runs inside their API, results never round-trip through us, and the
 * model emits `<cite index="N-M">` tags referencing a flat, 1-based list
 * of every result returned across the whole turn. It's the best option
 * when it's available — no extra model hop, no per-search bill on us.
 *
 * Nobody else has an equivalent we can drop in. OpenAI, Google,
 * OpenRouter and xAI each expose *some* server-side search, but every
 * one has a different tool name, a different result shape, and a
 * different citation mechanism (Google's sources arrive as
 * `groundingMetadata`, OpenRouter's as message `annotations` — neither
 * is a tool-result at all). Adapting four of those into one pipeline is
 * four adapters and four citation normalizers.
 *
 * So: keep Anthropic on its native tool, and give every other provider a
 * client-executed tool of the same name, backed by Exa, that returns the
 * same shape. Consumers see one `web_search` tool, one result type, one
 * citation convention, regardless of which model is driving.
 *
 * Usage:
 *
 *   const ws = createWebSearch({ provider, apiKey });
 *   const tools = { ...(ws.tool ? { [WEB_SEARCH_TOOL_NAME]: ws.tool } : {}) };
 *   const system = basePrompt + ws.promptSnippet;
 *   // in onStepFinish:  ws.capture(step.content)
 *   // when writing up:  linkifyCitations(markdown, ws.results)
 *
 * `ws.results` ends the run holding every result in citation order on
 * both paths, so downstream code never branches on the backend.
 */

/** Tool name registered with the model. Load-bearing: consumers key UI
 *  and step-walking off this exact string. */
export const WEB_SEARCH_TOOL_NAME = "web_search";

/** Which implementation backs the tool. */
export type SearchBackend = "anthropic" | "exa";

/**
 * One search hit. Field-compatible with Anthropic's
 * `web_search_result` output (`url` / `title` / `pageAge` / `type`), so
 * code that walks Anthropic tool-results parses Exa results unchanged.
 */
export interface WebSearchResult {
  url: string;
  title: string | null;
  /** Publish date when the backend reports one. */
  pageAge: string | null;
  /**
   * Extracted page text. Exa path only — Anthropic returns
   * `encryptedContent` that only their model can read, so on the native
   * path the text is never visible to us (or to this process).
   */
  text?: string;
  /**
   * 1-based citation index, flat across every `web_search` call in the
   * run — the number the model is told to cite. Exa path only; on the
   * Anthropic path the model derives indices itself.
   */
  index?: number;
  type: "web_search_result";
}

export interface WebSearchOptions {
  /** Max `web_search` calls per run. Default 3, matching the previous
   *  Anthropic-only default. Enforced in-process on the Exa path. */
  maxUses?: number;
  /** Results per call on the Exa path. Default 5. */
  numResults?: number;
  /** Per-result text budget on the Exa path. Default 4000. Raise for
   *  research writeups, lower for quick factual lookups. */
  maxCharacters?: number;
  allowedDomains?: string[];
  blockedDomains?: string[];
}

export interface CreateWebSearchOptions extends WebSearchOptions {
  /** LLM provider driving the run — decides the backend. */
  provider: Provider;
  /** LLM API key. Only used on the Anthropic path. Falls back to env. */
  apiKey?: string;
  /** Exa key. Falls back to `EXA_API_KEY`. */
  searchApiKey?: string;
  /**
   * Force a backend regardless of provider. `"exa"` is how you A/B the
   * shim against Anthropic's native tool on identical prompts.
   */
  backend?: SearchBackend;
  /**
   * Ask the model to cite sources as `<cite index="N">` tags, so
   * `formatOutput` can turn them into markdown links. Default `false`:
   * no citation instructions go out, and any tag the model emits anyway
   * is stripped to plain prose.
   *
   * Only turn this on for surfaces that actually render source links
   * (a research writeup). Claude honors the instruction unreliably —
   * measured 0/3 runs against claude-sonnet-5, which writes bare
   * parentheticals like `(Bitcoin Magazine)` instead — so a chat reply
   * asking for citations tends to get prose with no links either way.
   * See `npm run cite-rate`.
   */
  citations?: boolean;
  abortSignal?: AbortSignal;
}

export interface WebSearchHandle {
  /** Register under {@link WEB_SEARCH_TOOL_NAME}. `undefined` when no
   *  key is configured for the chosen backend — drop the tool rather
   *  than failing the request. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  tool: any | undefined;
  backend: SearchBackend | undefined;
  /** True when the model runs the search server-side (Anthropic). */
  native: boolean;
  /** Every result from the run, in citation order. */
  results: WebSearchResult[];
  /**
   * Feed each step's content here (AI SDK `onStepFinish`). Walks
   * Anthropic tool-results into `results`; a no-op on the Exa path,
   * where `execute` already appended them. Safe to call either way.
   */
  capture(stepContent: unknown): void;
  /** Citation instructions to append to the system prompt. Empty unless
   *  `citations: true` was requested. */
  promptSnippet: string;
  /**
   * Run the model's final text through the right citation treatment for
   * this handle: markdown links when `citations` is on, plain prose when
   * it isn't. Either way no raw `<cite>` markup survives — a leftover
   * tag renders as literal text in any GFM viewer.
   */
  formatOutput(markdown: string): { content: string; converted: number; skipped: number };
}

const EXA_SEARCH_URL = "https://api.exa.ai/search";
const DEFAULT_MAX_USES = 3;
const DEFAULT_NUM_RESULTS = 5;
const DEFAULT_MAX_CHARACTERS = 4000;

export function getSearchApiKey(): string | undefined {
  return normalizeApiKey(process.env.EXA_API_KEY);
}

export function hasSearchApiKey(): boolean {
  return !!getSearchApiKey();
}

/**
 * Which backend a provider gets. Anthropic keeps its native tool;
 * everything else falls to Exa.
 */
export function resolveSearchBackend(provider: Provider): SearchBackend {
  return provider === "anthropic" ? "anthropic" : "exa";
}

interface ExaResult {
  url?: string;
  title?: string | null;
  publishedDate?: string | null;
  text?: string;
}

/**
 * Raw Exa search. Exposed for callers that want results without an LLM
 * in the loop (a research pre-fetch, a URL enrichment pass).
 *
 * Throws on transport/HTTP failure — {@link createWebSearch} catches and
 * hands the model a readable error instead of failing the whole turn.
 */
export async function searchWeb(
  query: string,
  opts: WebSearchOptions & { apiKey?: string; abortSignal?: AbortSignal } = {},
): Promise<WebSearchResult[]> {
  const apiKey = normalizeApiKey(opts.apiKey) || getSearchApiKey();
  if (!apiKey) throw new Error("EXA_API_KEY not configured");

  const res = await fetch(EXA_SEARCH_URL, {
    method: "POST",
    headers: { "content-type": "application/json", "x-api-key": apiKey },
    signal: opts.abortSignal,
    body: JSON.stringify({
      query,
      type: "auto",
      numResults: opts.numResults ?? DEFAULT_NUM_RESULTS,
      contents: {
        text: { maxCharacters: opts.maxCharacters ?? DEFAULT_MAX_CHARACTERS },
      },
      ...(opts.allowedDomains?.length
        ? { includeDomains: opts.allowedDomains }
        : {}),
      ...(opts.blockedDomains?.length
        ? { excludeDomains: opts.blockedDomains }
        : {}),
    }),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Exa search failed (${res.status}): ${body.slice(0, 300)}`);
  }

  const json = (await res.json()) as { results?: ExaResult[] };
  const results = Array.isArray(json.results) ? json.results : [];
  return results
    .filter((r): r is ExaResult & { url: string } => typeof r.url === "string")
    .map((r) => ({
      url: r.url,
      title: r.title ?? null,
      pageAge: r.publishedDate ?? null,
      ...(r.text ? { text: r.text } : {}),
      type: "web_search_result" as const,
    }));
}

/**
 * Citation instructions.
 *
 * Both backends need these. Claude does NOT reliably emit `<cite>` tags
 * for `web_search` results unprompted — a live run against
 * claude-sonnet-5 produced ten captured results and zero citation tags,
 * writing bare parentheticals instead. So the native path gets the
 * instructions too; it just describes Anthropic's own `N-M` index form
 * rather than the shim's flat `N`.
 *
 * On the Exa path the model also can't *compute* `N` — it's a global
 * offset across every search call in the turn, which the model can't
 * see. So the tool hands the number back with each result and the
 * prompt says to reuse it verbatim. One format across both backends is
 * what lets {@link linkifyCitations} stay backend-agnostic.
 */
function citationSnippet(backend: SearchBackend): string {
  const indexRule =
    backend === "exa"
      ? "Every result carries an `index` field. Use that number exactly as returned — never invent or renumber one. Indices are global across all searches in this conversation."
      : "Use the index Anthropic assigns to the search result you are citing.";
  return `

## Citing web search results
When you use information from a \`${WEB_SEARCH_TOOL_NAME}\` result, cite it inline as \`<cite index="N">anchor text</cite>\`. ${indexRule}

The anchor text is REQUIRED and becomes the link text — put the words being sourced inside the tag, e.g. \`<cite index="3">runs on the Lightning Network</cite>\`. Never emit an empty tag like \`<cite index="3"></cite>\`; a trailing marker with no anchor produces a broken link.

Do NOT name sources parenthetically (e.g. \`(Bitcoin Magazine)\`) or as bare URLs — observed models fall back to that style and it produces no link. Every source reference must be a cite tag.`;
}

/**
 * Build the `web_search` tool for a run, plus the citation bookkeeping
 * that goes with it. See the module header for the usage shape.
 */
export function createWebSearch(
  opts: CreateWebSearchOptions,
): WebSearchHandle {
  const results: WebSearchResult[] = [];
  const backend = opts.backend ?? resolveSearchBackend(opts.provider);
  const maxUses = opts.maxUses ?? DEFAULT_MAX_USES;

  if (backend === "anthropic") {
    const apiKey =
      normalizeApiKey(opts.apiKey) || normalizeApiKey(process.env.ANTHROPIC_API_KEY);
    if (!apiKey) {
      return emptyHandle(results);
    }
    const baseURL = getGatewayBaseURL("anthropic");
    const anthropic = createAnthropic({ apiKey, ...(baseURL && { baseURL }) });
    return {
      tool: anthropic.tools.webSearch_20250305({
        maxUses,
        ...(opts.allowedDomains?.length
          ? { allowedDomains: opts.allowedDomains }
          : {}),
        ...(opts.blockedDomains?.length
          ? { blockedDomains: opts.blockedDomains }
          : {}),
      }),
      backend,
      native: true,
      results,
      capture: (stepContent) => captureNativeResults(stepContent, results),
      promptSnippet: opts.citations ? citationSnippet("anthropic") : "",
      formatOutput: (markdown) => formatOutput(markdown, results, !!opts.citations),
    };
  }

  const searchApiKey = normalizeApiKey(opts.searchApiKey) || getSearchApiKey();
  if (!searchApiKey) {
    return emptyHandle(results);
  }

  let uses = 0;
  return {
    tool: tool({
      description:
        "Search the web for current information. Returns numbered results with page text. " +
        "Prefer one specific query per call over broad multi-topic queries.",
      inputSchema: z.object({
        query: z.string().describe("The search query."),
        num_results: z
          .number()
          .int()
          .min(1)
          .max(10)
          .optional()
          .describe("How many results to return. Default 5."),
      }),
      execute: async ({
        query,
        num_results,
      }: {
        query: string;
        num_results?: number;
      }) => {
        if (uses >= maxUses) {
          return {
            error: `web_search budget exhausted (${maxUses} calls). Answer with what you have.`,
          };
        }
        uses++;
        try {
          const hits = await searchWeb(query, {
            numResults: num_results ?? opts.numResults,
            maxCharacters: opts.maxCharacters,
            allowedDomains: opts.allowedDomains,
            blockedDomains: opts.blockedDomains,
            apiKey: searchApiKey,
            abortSignal: opts.abortSignal,
          });
          // Indices are assigned HERE, at append time, so they're the
          // true global offset even with concurrent tool calls in one
          // step. The model gets the number it must cite; it never has
          // to compute one.
          return hits.map((hit) => {
            const indexed: WebSearchResult = { ...hit, index: results.length + 1 };
            results.push(indexed);
            return indexed;
          });
        } catch (err) {
          return {
            error: `Search failed: ${err instanceof Error ? err.message : String(err)}`,
          };
        }
      },
    }),
    backend,
    native: false,
    results,
    // Exa results are appended by `execute` above; walking the step
    // would double-count them.
    capture: () => {},
    promptSnippet: opts.citations ? citationSnippet("exa") : "",
    formatOutput: (markdown) => formatOutput(markdown, results, !!opts.citations),
  };
}

function emptyHandle(results: WebSearchResult[]): WebSearchHandle {
  return {
    tool: undefined,
    backend: undefined,
    native: false,
    results,
    capture: () => {},
    promptSnippet: "",
    formatOutput: (markdown) => formatOutput(markdown, results, false),
  };
}

function formatOutput(
  markdown: string,
  results: WebSearchResult[],
  citations: boolean,
): { content: string; converted: number; skipped: number } {
  return citations
    ? linkifyCitations(markdown, results)
    : { content: stripCitations(markdown), converted: 0, skipped: 0 };
}

/**
 * Walk one AI SDK step's content for `web_search` tool-results and
 * append each hit to `target`, in order.
 *
 * Order is load-bearing: Anthropic's `<cite index="N-M">` tags index
 * this flat list 1-based across the entire turn.
 *
 * Tolerates both result shapes (`output` and `result`) and skips any
 * non-array body — adapters vary across AI SDK versions, and a shape we
 * don't recognize should cost us citations, not the run.
 */
export function captureNativeResults(
  stepContent: unknown,
  target: WebSearchResult[],
): void {
  if (!Array.isArray(stepContent)) return;
  for (const content of stepContent) {
    if (content?.type !== "tool-result") continue;
    if (content?.toolName !== WEB_SEARCH_TOOL_NAME) continue;
    const body = content.output ?? content.result ?? null;
    if (!Array.isArray(body)) continue;
    for (const r of body) {
      if (r && typeof r === "object" && typeof r.url === "string") {
        target.push({
          url: r.url,
          title: r.title ?? null,
          pageAge: r.pageAge ?? null,
          // Assigned here too, so `results` is shape-identical to the
          // Exa path. Anthropic's model derives its own indices from
          // the same ordering, so these agree by construction.
          index: target.length + 1,
          type: "web_search_result",
        });
      }
    }
  }
}


/**
 * Remove citation markup, leaving readable prose.
 *
 * The default treatment when `citations` is off. Models emit `<cite`
 * tags unprompted often enough that we can't just hope: Claude produces
 * them spontaneously with server-side search, and a raw tag renders as
 * literal text in any GFM viewer, which looks broken.
 *
 * Three shapes, in order:
 *   1. Empty markers — `a claim. <cite index="4"></cite>` — are
 *      trailing footnotes with nothing to say. They take their leading
 *      whitespace with them, so the sentence closes up cleanly.
 *   2. Anchored tags collapse to their anchor text, which is the words
 *      the model was sourcing and reads as normal prose.
 *   3. Any orphan `<cite ...>` or `</cite>` left by a truncated stream
 *      is swept, so no half-tag survives a cut-off response.
 */
export function stripCitations(markdown: string): string {
  return markdown
    .replace(/[ \t]*<cite\b[^>]*>\s*<\/cite>/g, "")
    .replace(/<cite\b[^>]*>(.*?)<\/cite>/g, "$1")
    .replace(/<\/?cite\b[^>]*>/g, "");
}

/**
 * Replace `<cite index="N">anchor</cite>` tags with markdown links into
 * `results`. Handles Anthropic's multi-part form (`index="2-1"`,
 * `index="2-1,3-4"`) by keying off the leading number, which is the
 * flat result index.
 *
 * Out-of-range or unresolvable indices collapse to the bare anchor
 * text. That span loses its link, but no raw `<cite>` markup survives
 * into the output — persisted markdown is usually rendered as plain
 * GFM, where a leftover tag shows up as literal text.
 *
 * Models regularly ignore the "anchor text is required" instruction and
 * emit a trailing `<cite index="4"></cite>` marker instead (grok-4 does
 * it consistently). An empty anchor would linkify to `[](url)` — a
 * broken, invisible link — so it falls back to a `[N]` footnote marker.
 */
export function linkifyCitations(
  markdown: string,
  results: WebSearchResult[],
): { content: string; converted: number; skipped: number } {
  let converted = 0;
  let skipped = 0;
  const content = markdown.replace(
    /<cite index="(\d+)(?:-\d+(?:,\d+-\d+)*)?">(.*?)<\/cite>/g,
    (_match, indexStr: string, anchor: string) => {
      const index = parseInt(indexStr, 10);
      const hit = results[index - 1];
      if (!hit || typeof hit.url !== "string") {
        skipped++;
        return anchor;
      }
      converted++;
      // Anchor text is model-generated; a stray `]` would break the
      // markdown link, and an empty one would render as an invisible
      // link, so degrade to a numbered footnote marker.
      const text = anchor.trim() ? anchor.replace(/\]/g, "\\]") : `[${index}]`;
      return `[${text}](${hit.url})`;
    },
  );
  return { content, converted, skipped };
}
