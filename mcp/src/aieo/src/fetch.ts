import { tool } from "ai";
import { z } from "zod";
import { createAnthropic } from "@ai-sdk/anthropic";
import { BlockList, isIP } from "node:net";
import { lookup as dnsLookup } from "node:dns/promises";
import { Provider, getGatewayBaseURL, normalizeApiKey } from "./provider.js";

/**
 * Web fetch, uniform across providers. Sibling of `./search.js`.
 *
 * Anthropic ships a server-executed `web_fetch` tool: given a URL, their
 * API retrieves the page (HTML or PDF), turns it into text and hands it
 * to the model with no round-trip through us. Same reasons that make it
 * the best option when available: no extra hop, nothing to host,
 * nothing to secure.
 *
 * Nobody else has one. So, as with search: keep Anthropic native, and
 * give every other provider a client-executed tool of the same name and
 * result shape, backed by a plain HTTP GET plus an HTML-to-text pass.
 *
 * Two things differ from search that consumers should know:
 *
 *   1. On the Anthropic path the model can only fetch URLs that already
 *      appeared in the conversation — pasted by the user, or returned by
 *      an earlier `web_search` / `web_fetch`. It refuses URLs it made
 *      up. The HTTP path has no such memory and fetches whatever the
 *      model asks for; what it WON'T do is reach anything private.
 *
 *   2. The HTTP path runs in *our* process, so the model can point it at
 *      our network. Every URL — and every redirect hop — is checked
 *      before connecting: http(s) only, no credentials, and the host
 *      must resolve exclusively to public unicast addresses (loopback,
 *      RFC 1918, link-local incl. cloud metadata, CGNAT, ULA and the
 *      v4-in-v6 forms are all refused). There is a DNS-rebinding window
 *      between our lookup and the socket's own; pin `allowedDomains` if
 *      the deployment cares.
 *
 * Usage:
 *
 *   const wf = createWebFetch({ provider, apiKey });
 *   const tools = { ...(wf.tool ? { [WEB_FETCH_TOOL_NAME]: wf.tool } : {}) };
 *   // in onStepFinish:  wf.capture(step.content)
 *   // afterwards:       wf.results — every page fetched, in order
 */

/** Tool name registered with the model. Matches Anthropic's native name
 *  so consumers key UI and step-walking off one string on both paths. */
export const WEB_FETCH_TOOL_NAME = "web_fetch";

/** Which implementation backs the tool. */
export type FetchBackend = "anthropic" | "http";

/**
 * One fetched page. Same fields on both paths; the notes say where the
 * backends differ in what they put in them.
 */
export interface WebFetchResult {
  /** Where the content came from. The final URL after redirects on the
   *  HTTP path; the URL Anthropic reports on the native path. */
  url: string;
  title: string | null;
  /**
   * Extracted text. Absent for a PDF on the Anthropic path — their API
   * returns it base64-encoded, which is only useful to their model.
   */
  text?: string;
  /** Original `content-type` on the HTTP path; Anthropic's normalized
   *  `text/plain` / `application/pdf` on the native path. */
  mediaType: string | null;
  retrievedAt: string | null;
  /** True when the HTTP path cut the text at `maxCharacters`. */
  truncated?: boolean;
  type: "web_fetch_result";
}

export interface WebFetchOptions {
  /** Max `web_fetch` calls per run. Default 5. Enforced in-process on
   *  the HTTP path, passed as `max_uses` to Anthropic. */
  maxUses?: number;
  /**
   * Text budget per page on the HTTP path, in characters. Default
   * 40000 — about 10k tokens. When only `maxContentTokens` is given,
   * derived from it at 4 chars/token so one option covers both paths.
   */
  maxCharacters?: number;
  /**
   * Content budget per page on the Anthropic path, in tokens (their
   * `max_content_tokens`). Unset means Anthropic's own default. When
   * only `maxCharacters` is given, derived from it.
   */
  maxContentTokens?: number;
  /** Only fetch from these domains. Subdomains are included, so
   *  `example.com` admits `docs.example.com`. */
  allowedDomains?: string[];
  blockedDomains?: string[];
}

/** Resolve a hostname to every address it answers with. */
export type HostLookup = (hostname: string) => Promise<string[]>;

export interface CreateWebFetchOptions extends WebFetchOptions {
  /** LLM provider driving the run — decides the backend. */
  provider: Provider;
  /** LLM API key. Only used on the Anthropic path. Falls back to env. */
  apiKey?: string;
  /** Force a backend regardless of provider. `"http"` is how you A/B
   *  the shim against Anthropic's native tool on identical prompts. */
  backend?: FetchBackend;
  abortSignal?: AbortSignal;
  /**
   * Override hostname resolution on the HTTP path. Tests inject a stub
   * here; a deployment with its own resolver policy can too. Must return
   * every address the host resolves to — the guard refuses a host if ANY
   * of them is private.
   */
  lookup?: HostLookup;
}

export interface WebFetchHandle {
  /** Register under {@link WEB_FETCH_TOOL_NAME}. `undefined` only when
   *  the Anthropic path has no API key — drop the tool rather than
   *  failing the request. The HTTP path needs no key. */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  tool: any | undefined;
  backend: FetchBackend | undefined;
  /** True when the fetch runs server-side (Anthropic). */
  native: boolean;
  /** Every page fetched during the run, in order. */
  results: WebFetchResult[];
  /**
   * Feed each step's content here (AI SDK `onStepFinish`). Walks
   * Anthropic tool-results into `results`; a no-op on the HTTP path,
   * where `execute` already appended them. Safe to call either way.
   */
  capture(stepContent: unknown): void;
}

const DEFAULT_MAX_USES = 5;
const DEFAULT_MAX_CHARACTERS = 40_000;
const CHARS_PER_TOKEN = 4;
const DEFAULT_TIMEOUT_MS = 20_000;
const MAX_REDIRECTS = 5;
/** Raw bytes read from a response before giving up on it. HTML runs
 *  several times its text; 4 MiB covers any page whose text fits the
 *  default budget with room to spare. */
const MAX_BODY_BYTES = 4 * 1024 * 1024;
const USER_AGENT = "aieo-web-fetch/1 (+https://github.com/stakwork/stakgraph)";
const ACCEPT =
  "text/html, application/xhtml+xml, text/plain;q=0.9, application/json;q=0.9, text/*;q=0.8, */*;q=0.5";

/**
 * Which backend a provider gets. Anthropic keeps its native tool;
 * everything else falls to the HTTP shim.
 */
export function resolveFetchBackend(provider: Provider): FetchBackend {
  return provider === "anthropic" ? "anthropic" : "http";
}

// ── Address guard ──────────────────────────────────────────────────────

/**
 * Addresses the HTTP path never connects to. Built once.
 *
 * `BlockList` checks an IPv4-mapped IPv6 address (`::ffff:a.b.c.d`)
 * against the v4 rules by itself. NAT64 and 6to4 embed a v4 address in
 * a way it doesn't unpack, so those prefixes are refused whole — neither
 * is a plausible target for a page fetch.
 */
const PRIVATE_ADDRESSES = buildBlockList();

function buildBlockList(): BlockList {
  const b = new BlockList();
  const v4: Array<[string, number]> = [
    ["0.0.0.0", 8], // "this" network
    ["10.0.0.0", 8], // RFC 1918
    ["100.64.0.0", 10], // carrier-grade NAT
    ["127.0.0.0", 8], // loopback
    ["169.254.0.0", 16], // link-local, incl. cloud metadata at 169.254.169.254
    ["172.16.0.0", 12], // RFC 1918
    ["192.0.0.0", 24], // IETF protocol assignments
    ["192.168.0.0", 16], // RFC 1918
    ["198.18.0.0", 15], // benchmarking
    ["224.0.0.0", 4], // multicast
    ["240.0.0.0", 4], // reserved, incl. broadcast
  ];
  for (const [net, prefix] of v4) b.addSubnet(net, prefix, "ipv4");
  b.addAddress("::", "ipv6"); // unspecified
  b.addAddress("::1", "ipv6"); // loopback
  const v6: Array<[string, number]> = [
    ["64:ff9b::", 96], // NAT64
    ["2002::", 16], // 6to4
    ["fc00::", 7], // unique local
    ["fe80::", 10], // link-local
    ["fec0::", 10], // site-local (deprecated, still routed on old gear)
    ["ff00::", 8], // multicast
  ];
  for (const [net, prefix] of v6) b.addSubnet(net, prefix, "ipv6");
  return b;
}

/**
 * True for an IP literal (v4 or v6, brackets tolerated) the HTTP path
 * refuses to connect to. Anything that isn't an IP literal is `true`
 * too: an address we can't classify isn't one we connect to.
 */
export function isPrivateAddress(ip: string): boolean {
  const bare = ip.startsWith("[") && ip.endsWith("]") ? ip.slice(1, -1) : ip;
  const family = isIP(bare);
  try {
    if (family === 4) return PRIVATE_ADDRESSES.check(bare, "ipv4");
    if (family === 6) return PRIVATE_ADDRESSES.check(bare, "ipv6");
  } catch {
    // Zone ids and other oddities BlockList can't parse.
  }
  return true;
}

const defaultLookup: HostLookup = async (hostname) => {
  const addresses = await dnsLookup(hostname, { all: true });
  return addresses.map((a) => a.address);
};

function matchesDomain(host: string, domains: string[]): boolean {
  return domains.some((d) => {
    const dom = d
      .trim()
      .toLowerCase()
      .replace(/^https?:\/\//, "")
      .replace(/\/.*$/, "")
      .replace(/^\*\./, "")
      .replace(/\.$/, "");
    return !!dom && (host === dom || host.endsWith("." + dom));
  });
}

/**
 * Validate a URL before the HTTP path connects to it. Throws a readable
 * error on any rejection — the model gets the message back as the
 * tool's `error` and can pick a different URL.
 *
 * A hostname is resolved and refused if ANY of its addresses is
 * private: a host that answers with a mix is misconfigured or hostile,
 * and we'd have no say in which address the socket picks.
 */
export async function validateFetchUrl(
  raw: string,
  opts: { allowedDomains?: string[]; blockedDomains?: string[]; lookup?: HostLookup } = {},
): Promise<URL> {
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    throw new Error(`Not a valid absolute URL: ${raw}`);
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error(`Only http(s) URLs can be fetched, got ${url.protocol}`);
  }
  if (url.username || url.password) {
    throw new Error("URLs with embedded credentials are not fetched");
  }
  // WHATWG parsing already normalized the numeric IPv4 forms
  // (`http://2130706433/`, `http://0x7f.1/`) to dotted quads.
  const host = url.hostname.replace(/\.$/, "").toLowerCase();
  if (!host) throw new Error("URL has no host");
  if (host === "localhost" || host.endsWith(".localhost")) {
    throw new Error("Refusing to fetch a local address");
  }
  if (opts.allowedDomains?.length && !matchesDomain(host, opts.allowedDomains)) {
    throw new Error(`${host} is not in the allowed domains`);
  }
  if (opts.blockedDomains?.length && matchesDomain(host, opts.blockedDomains)) {
    throw new Error(`${host} is a blocked domain`);
  }

  const literal = host.startsWith("[") ? host.slice(1, -1) : host;
  if (isIP(literal)) {
    if (isPrivateAddress(literal)) {
      throw new Error("Refusing to fetch a private or reserved address");
    }
    return url;
  }

  let addresses: string[];
  try {
    addresses = await (opts.lookup ?? defaultLookup)(host);
  } catch (err) {
    throw new Error(
      `Could not resolve ${host}: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
  if (!addresses.length) throw new Error(`Could not resolve ${host}`);
  if (addresses.some(isPrivateAddress)) {
    throw new Error(`Refusing to fetch ${host}: it resolves to a private or reserved address`);
  }
  return url;
}

// ── HTTP path ──────────────────────────────────────────────────────────

export interface FetchUrlOptions {
  maxCharacters?: number;
  allowedDomains?: string[];
  blockedDomains?: string[];
  abortSignal?: AbortSignal;
  timeoutMs?: number;
  lookup?: HostLookup;
}

/**
 * Raw fetch-and-extract. Exposed for callers that want a page without
 * an LLM in the loop (a URL enrichment pass, a link preview).
 *
 * Throws on any rejection or failure — {@link createWebFetch} catches
 * and hands the model a readable error instead of failing the turn.
 */
export async function fetchUrl(
  rawUrl: string,
  opts: FetchUrlOptions = {},
): Promise<WebFetchResult> {
  const maxCharacters = opts.maxCharacters ?? DEFAULT_MAX_CHARACTERS;
  const signals = [AbortSignal.timeout(opts.timeoutMs ?? DEFAULT_TIMEOUT_MS)];
  if (opts.abortSignal) signals.push(opts.abortSignal);
  const signal = AbortSignal.any(signals);

  let url = await validateFetchUrl(rawUrl, opts);
  let res: Response;
  for (let hop = 0; ; hop++) {
    try {
      res = await fetch(url, {
        method: "GET",
        redirect: "manual",
        signal,
        headers: { "user-agent": USER_AGENT, accept: ACCEPT },
      });
    } catch (err) {
      throw new Error(`Could not connect to ${url.hostname}: ${describeTransportError(err)}`);
    }
    if (!isRedirect(res.status)) break;
    // Every hop is re-validated: a public host that 302s to
    // 169.254.169.254 is the classic SSRF bypass.
    const location = res.headers.get("location");
    await discard(res);
    if (!location) throw new Error(`Redirect (${res.status}) without a Location header`);
    if (hop >= MAX_REDIRECTS) throw new Error(`Too many redirects (more than ${MAX_REDIRECTS})`);
    let next: string;
    try {
      next = new URL(location, url).toString();
    } catch {
      throw new Error(`Redirect to an invalid URL: ${location}`);
    }
    url = await validateFetchUrl(next, opts);
  }

  if (!res.ok) {
    await discard(res);
    throw new Error(`HTTP ${res.status} fetching ${url.hostname}`);
  }
  const contentType = res.headers.get("content-type") ?? "";
  const mediaType = contentType.split(";")[0].trim().toLowerCase() || null;
  if (mediaType === "application/pdf") {
    await discard(res);
    throw new Error(
      "PDF documents are not supported by this fetch backend (only Anthropic's native web_fetch reads PDFs)",
    );
  }
  if (!isTextual(mediaType)) {
    await discard(res);
    throw new Error(`Unsupported content type: ${mediaType}`);
  }

  const { bytes, capped } = await readCapped(res, MAX_BODY_BYTES);
  const raw = decode(bytes, contentType);
  const isHtml =
    mediaType === "text/html" || mediaType === "application/xhtml+xml" || looksLikeHtml(raw);
  const { title, text } = isHtml ? htmlToText(raw) : { title: null, text: raw.trim() };
  const truncated = capped || text.length > maxCharacters;
  return {
    url: url.toString(),
    title,
    text: truncated ? text.slice(0, maxCharacters) : text,
    mediaType,
    retrievedAt: new Date().toISOString(),
    truncated,
    type: "web_fetch_result",
  };
}

/**
 * undici wraps every transport failure as `TypeError: fetch failed` and
 * puts the real reason on `cause` — an expired certificate, ECONNREFUSED,
 * a reset. Surface that: "fetch failed" tells the model (and whoever
 * reads the logs) nothing.
 */
function describeTransportError(err: unknown): string {
  const e = err as {
    name?: string;
    message?: string;
    cause?: { code?: string; message?: string };
  };
  if (e?.name === "TimeoutError") return "timed out";
  if (e?.name === "AbortError") return "aborted";
  const cause = e?.cause;
  if (cause?.message) return cause.code ? `${cause.message} (${cause.code})` : cause.message;
  return e?.message ?? String(err);
}

function isRedirect(status: number): boolean {
  return status === 301 || status === 302 || status === 303 || status === 307 || status === 308;
}

/** Release a response we won't read, so its connection goes back to the pool. */
async function discard(res: Response): Promise<void> {
  await res.body?.cancel().catch(() => {});
}

function isTextual(mediaType: string | null): boolean {
  // No header at all: read it and sniff.
  if (!mediaType) return true;
  if (mediaType.startsWith("text/")) return true;
  if (mediaType.endsWith("+json") || mediaType.endsWith("+xml")) return true;
  return (
    mediaType === "application/json" ||
    mediaType === "application/xml" ||
    mediaType === "application/xhtml+xml" ||
    mediaType === "application/javascript"
  );
}

async function readCapped(
  res: Response,
  maxBytes: number,
): Promise<{ bytes: Uint8Array; capped: boolean }> {
  const body = res.body;
  if (!body) return { bytes: new Uint8Array(await res.arrayBuffer()), capped: false };
  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  let capped = false;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;
    if (total + value.byteLength > maxBytes) {
      chunks.push(value.subarray(0, maxBytes - total));
      total = maxBytes;
      capped = true;
      await reader.cancel().catch(() => {});
      break;
    }
    chunks.push(value);
    total += value.byteLength;
  }
  const out = new Uint8Array(total);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.byteLength;
  }
  return { bytes: out, capped };
}

function decode(bytes: Uint8Array, contentType: string): string {
  const charset = /charset=["']?([\w.:-]+)/i.exec(contentType)?.[1] ?? "utf-8";
  try {
    return new TextDecoder(charset).decode(bytes);
  } catch {
    return new TextDecoder("utf-8").decode(bytes);
  }
}

function looksLikeHtml(s: string): boolean {
  return /^\s*(?:<!doctype\s+html|<html[\s>]|<head[\s>]|<body[\s>])/i.test(s.slice(0, 1024));
}

// ── HTML → text ────────────────────────────────────────────────────────

const NAMED_ENTITIES: Record<string, string> = {
  nbsp: " ",
  lt: "<",
  gt: ">",
  quot: '"',
  apos: "'",
  mdash: "—",
  ndash: "–",
  hellip: "…",
  copy: "©",
  reg: "®",
  trade: "™",
  laquo: "«",
  raquo: "»",
  ldquo: "“",
  rdquo: "”",
  lsquo: "‘",
  rsquo: "’",
  bull: "•",
  middot: "·",
  times: "×",
  deg: "°",
};

function codePoint(cp: number, fallback: string): string {
  if (!Number.isFinite(cp) || cp <= 0 || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) {
    return fallback;
  }
  return String.fromCodePoint(cp);
}

/** Numeric and the common named entities. `&amp;` goes last so
 *  `&amp;lt;` decodes to the literal `&lt;` the author wrote. */
function decodeEntities(s: string): string {
  return s
    .replace(/&#x([0-9a-f]{1,6});/gi, (m, hex: string) => codePoint(parseInt(hex, 16), m))
    .replace(/&#(\d{1,7});/g, (m, dec: string) => codePoint(parseInt(dec, 10), m))
    .replace(/&([a-z]+);/gi, (m, name: string) => NAMED_ENTITIES[name.toLowerCase()] ?? m)
    .replace(/&amp;/g, "&");
}

const BLOCK_TAGS =
  "p|div|section|article|header|footer|main|aside|nav|h[1-6]|ul|ol|tr|table|thead|tbody|tfoot|blockquote|pre|hr|dd|dt|dl|figure|figcaption|form|fieldset|address|details|summary";

/**
 * Dependency-free HTML to text. Good enough for a model to read a page;
 * not a renderer. Scripts, styles and SVG go away entirely; block-level
 * boundaries become line breaks; list items get a leading dash; the
 * rest of the markup is dropped and entities decoded. Whitespace is
 * collapsed except for line breaks, so `<pre>` keeps its lines but not
 * its indentation.
 */
export function htmlToText(html: string): { title: string | null; text: string } {
  const titleMatch = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(html);
  const title = titleMatch
    ? decodeEntities(titleMatch[1]).replace(/\s+/g, " ").trim() || null
    : null;

  const stripped = html
    .replace(/<!--[\s\S]*?-->/g, " ")
    .replace(/<(script|style|noscript|svg|template|head)\b[\s\S]*?<\/\1\s*>/gi, " ")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<li\b[^>]*>/gi, "\n- ")
    .replace(new RegExp(`<\\/?(?:${BLOCK_TAGS})\\b[^>]*>`, "gi"), "\n")
    .replace(/<[^>]+>/g, " ");

  const text = decodeEntities(stripped)
    .replace(/[ \t\r\f\v ]+/g, " ")
    .replace(/ ?\n ?/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  return { title, text };
}

// ── Handle ─────────────────────────────────────────────────────────────

/**
 * Build the `web_fetch` tool for a run. See the module header for the
 * usage shape.
 */
export function createWebFetch(opts: CreateWebFetchOptions): WebFetchHandle {
  const results: WebFetchResult[] = [];
  const backend = opts.backend ?? resolveFetchBackend(opts.provider);
  const maxUses = opts.maxUses ?? DEFAULT_MAX_USES;
  const maxCharacters =
    opts.maxCharacters ??
    (opts.maxContentTokens ? opts.maxContentTokens * CHARS_PER_TOKEN : DEFAULT_MAX_CHARACTERS);
  const maxContentTokens =
    opts.maxContentTokens ??
    (opts.maxCharacters ? Math.ceil(opts.maxCharacters / CHARS_PER_TOKEN) : undefined);

  if (backend === "anthropic") {
    const apiKey =
      normalizeApiKey(opts.apiKey) || normalizeApiKey(process.env.ANTHROPIC_API_KEY);
    if (!apiKey) {
      return emptyFetchHandle(results);
    }
    const baseURL = getGatewayBaseURL("anthropic");
    const anthropic = createAnthropic({ apiKey, ...(baseURL && { baseURL }) });
    return {
      tool: anthropic.tools.webFetch_20250910({
        maxUses,
        ...(maxContentTokens ? { maxContentTokens } : {}),
        ...(opts.allowedDomains?.length ? { allowedDomains: opts.allowedDomains } : {}),
        ...(opts.blockedDomains?.length ? { blockedDomains: opts.blockedDomains } : {}),
      }),
      backend,
      native: true,
      results,
      capture: (stepContent) => captureNativeFetchResults(stepContent, results),
    };
  }

  let uses = 0;
  return {
    tool: tool({
      description:
        "Fetch a specific web page by URL and return its text. HTML is converted to plain text; " +
        "JSON and plain text come back as-is. Use this to read a page whose address you already " +
        "have — one the user gave you, or one returned by web_search. It is not a search engine: " +
        "it needs a full http(s) URL. PDFs and other binary content are not supported.",
      inputSchema: z.object({
        url: z.string().describe("The absolute http(s) URL to fetch."),
      }),
      execute: async ({ url }: { url: string }, ctx?: { abortSignal?: AbortSignal }) => {
        if (uses >= maxUses) {
          return {
            error: `web_fetch budget exhausted (${maxUses} calls). Work with what you have.`,
          };
        }
        uses++;
        try {
          const page = await fetchUrl(url, {
            maxCharacters,
            allowedDomains: opts.allowedDomains,
            blockedDomains: opts.blockedDomains,
            abortSignal: ctx?.abortSignal ?? opts.abortSignal,
            lookup: opts.lookup,
          });
          results.push(page);
          return page;
        } catch (err) {
          return {
            error: `Fetch failed: ${err instanceof Error ? err.message : String(err)}`,
          };
        }
      },
    }),
    backend,
    native: false,
    results,
    // HTTP results are appended by `execute` above; walking the step
    // would double-count them.
    capture: () => {},
  };
}

function emptyFetchHandle(results: WebFetchResult[]): WebFetchHandle {
  return {
    tool: undefined,
    backend: undefined,
    native: false,
    results,
    capture: () => {},
  };
}

/**
 * Walk one AI SDK step's content for `web_fetch` tool-results and
 * append each page to `target`, in order.
 *
 * Tolerates both result shapes (`output` and `result`) and both key
 * casings for the nested fields — adapters vary across AI SDK versions,
 * and a shape we don't recognize should cost us a bookkeeping entry,
 * not the run. Anthropic's error results (`web_fetch_tool_result_error`)
 * are skipped: there's no page to record.
 */
export function captureNativeFetchResults(
  stepContent: unknown,
  target: WebFetchResult[],
): void {
  if (!Array.isArray(stepContent)) return;
  for (const content of stepContent) {
    if (content?.type !== "tool-result") continue;
    if (content?.toolName !== WEB_FETCH_TOOL_NAME) continue;
    const body = content.output ?? content.result ?? null;
    if (!body || typeof body !== "object") continue;
    if (body.type !== "web_fetch_result" || typeof body.url !== "string") continue;
    const doc = body.content ?? {};
    const source = doc.source ?? {};
    target.push({
      url: body.url,
      title: doc.title ?? null,
      ...(source.type === "text" && typeof source.data === "string"
        ? { text: source.data }
        : {}),
      mediaType: source.mediaType ?? source.media_type ?? null,
      retrievedAt: body.retrievedAt ?? body.retrieved_at ?? null,
      type: "web_fetch_result",
    });
  }
}
