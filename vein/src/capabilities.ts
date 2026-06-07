/**
 * Standard "capabilities" — the small, generic, host-owned services that
 * LLM-authored adapter STEPS build on (see AGENTS.md "step vs service").
 *
 * An adapter step reaches the outside world through `ctx.services.http` (a
 * fetch-like transport) and `ctx.services.secrets` (credential access), never
 * the global `fetch` / `process.env` directly. Routing I/O through these two
 * capabilities is what makes an adapter:
 *   - **recordable** — `http` returns a PLAIN serializable object (a real
 *     `fetch` Response can't be written to a cassette), so the record/replay
 *     wrapper (`cassette.ts`) can capture and replay it; and
 *   - **leak-free** — secrets flow through one boundary, so the recorder knows
 *     exactly which values to scrub out of the cassette.
 *
 * These are the DEFAULT implementations the standard server injects. Consumers
 * using vein as a library can spread them into — or override them within —
 * their own typed services bag.
 */

// ── secrets ────────────────────────────────────────────────────────────────

/** Credential access. The single boundary through which adapters read secrets
 *  (API keys, tokens). Backed by `process.env` by default; swap the `source`
 *  for a real secret store later without touching any adapter. */
export interface SecretsCapability {
  get(name: string): Promise<string | undefined>;
}

/** Build a secrets capability over a flat key→value source (defaults to
 *  `process.env`). */
export function secretsCapability(
  source: Record<string, string | undefined> = process.env,
): SecretsCapability {
  return {
    async get(name: string) {
      return source[name];
    },
  };
}

// ── http ─────────────────────────────────────────────────────────────────

export interface HttpRequestOptions {
  method?: string;
  headers?: Record<string, string>;
  /** Request body. Objects are JSON-encoded (with a default
   *  `content-type: application/json`); strings are sent as-is. */
  body?: unknown;
  /** Query params appended to the URL. */
  query?: Record<string, string | number | boolean>;
  /** Abort the request after this many milliseconds. */
  timeout?: number;
}

/** A plain, fully-serializable HTTP response — deliberately NOT a `fetch`
 *  Response, so it can be written to / replayed from a cassette. `body` is the
 *  parsed JSON when the response is JSON, otherwise the raw text. */
export interface HttpResponse {
  status: number;
  ok: boolean;
  headers: Record<string, string>;
  body: unknown;
}

/** Fetch-like transport. The blessed path for adapter network I/O. */
export type HttpCapability = (
  url: string,
  opts?: HttpRequestOptions,
) => Promise<HttpResponse>;

/** Minimal shape of the global `fetch` we depend on — kept tiny so a fake can
 *  be injected in tests without pulling in DOM lib types. */
export type FetchLike = (
  url: string,
  init?: {
    method?: string;
    headers?: Record<string, string>;
    body?: string;
    signal?: AbortSignal;
  },
) => Promise<{
  status: number;
  ok: boolean;
  headers: { forEach(cb: (value: string, key: string) => void): void };
  text(): Promise<string>;
}>;

function appendQuery(
  url: string,
  query?: Record<string, string | number | boolean>,
): string {
  if (!query) return url;
  const pairs = Object.entries(query).map(
    ([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`,
  );
  if (!pairs.length) return url;
  return url + (url.includes("?") ? "&" : "?") + pairs.join("&");
}

/** Build an http capability over a `fetch` implementation (defaults to the
 *  global `fetch`). Encodes object bodies as JSON, parses JSON responses, and
 *  returns a plain serializable {@link HttpResponse}. */
export function httpCapability(
  fetchImpl: FetchLike = globalThis.fetch as unknown as FetchLike,
): HttpCapability {
  if (typeof fetchImpl !== "function") {
    throw new Error(
      "httpCapability: no fetch available — pass a fetch implementation",
    );
  }
  return async (url, opts = {}) => {
    const headers: Record<string, string> = { ...(opts.headers ?? {}) };
    let body: string | undefined;
    if (opts.body !== undefined) {
      if (typeof opts.body === "string") {
        body = opts.body;
      } else {
        body = JSON.stringify(opts.body);
        if (!hasHeader(headers, "content-type")) {
          headers["content-type"] = "application/json";
        }
      }
    }

    const res = await fetchImpl(appendQuery(url, opts.query), {
      method: opts.method ?? (opts.body !== undefined ? "POST" : "GET"),
      headers,
      body,
      ...(opts.timeout ? { signal: AbortSignal.timeout(opts.timeout) } : {}),
    });

    const outHeaders: Record<string, string> = {};
    res.headers.forEach((value, key) => {
      outHeaders[key.toLowerCase()] = value;
    });

    const text = await res.text();
    const isJson = (outHeaders["content-type"] ?? "").includes("application/json");
    let parsed: unknown = text;
    if (isJson || looksLikeJson(text)) {
      try {
        parsed = JSON.parse(text);
      } catch {
        parsed = text;
      }
    }

    return { status: res.status, ok: res.ok, headers: outHeaders, body: parsed };
  };
}

function hasHeader(headers: Record<string, string>, name: string): boolean {
  return Object.keys(headers).some((k) => k.toLowerCase() === name);
}

function looksLikeJson(text: string): boolean {
  const t = text.trim();
  return t.startsWith("{") || t.startsWith("[");
}

// ── standard bag ───────────────────────────────────────────────────────────

/** The standard capability shape adapters rely on. Consumers extend this with
 *  their own typed services (graph store, llm client, …). */
export interface VeinCapabilities {
  http: HttpCapability;
  secrets: SecretsCapability;
}

/** The default standard services bag: env-backed secrets + global-fetch http.
 *  Injected by the standard server; override per environment as needed. */
export function standardServices(
  opts: { fetchImpl?: FetchLike; secretsSource?: Record<string, string | undefined> } = {},
): VeinCapabilities {
  return {
    http: httpCapability(opts.fetchImpl),
    secrets: secretsCapability(opts.secretsSource),
  };
}
