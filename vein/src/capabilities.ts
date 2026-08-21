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

import { mkdir, readFile, writeFile, readdir } from "node:fs/promises";
import {
  dirname as pathDirname,
  join as pathJoin,
  relative as pathRelative,
  resolve as pathResolve,
  sep as pathSep,
} from "node:path";

// ── secrets ────────────────────────────────────────────────────────────────

/** Credential access. The single boundary through which adapters read secrets
 *  (API keys, tokens). Backed by `process.env` by default; back it with a
 *  persisted {@link SecretReadable} (e.g. a `SecretStore`) for UI-managed
 *  secrets — without touching any adapter. */
export interface SecretsCapability {
  get(name: string): Promise<string | undefined>;
}

/** The narrow read surface a secrets capability needs from a backing store —
 *  satisfied by `SecretStore` (secret-store.ts) without importing it here. */
export interface SecretReadable {
  get(name: string): Promise<string | undefined>;
}

/**
 * Build a secrets capability. Pass either:
 *   - a flat key→value source (defaults to `process.env`), or
 *   - a `SecretReadable` store (e.g. a `SecretStore`), optionally with an
 *     `envFallback` so unset secrets still resolve from `process.env`.
 *
 * The store path is async-aware: a managed secret wins, then the env fallback.
 */
export function secretsCapability(
  source: Record<string, string | undefined> | SecretReadable = process.env,
  opts: { envFallback?: Record<string, string | undefined> } = {},
): SecretsCapability {
  if (typeof (source as SecretReadable).get === "function") {
    const store = source as SecretReadable;
    const fallback = opts.envFallback;
    return {
      async get(name: string) {
        const v = await store.get(name);
        if (v !== undefined) return v;
        return fallback?.[name];
      },
    };
  }
  const flat = source as Record<string, string | undefined>;
  return {
    async get(name: string) {
      return flat[name];
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

// ── artifacts ──────────────────────────────────────────────────────────────

/**
 * Per-run artifact storage — files a run produces that later steps (and
 * humans, via `GET /artifacts/:runId/…`) reference. The convention: a step
 * writes a file and puts its RELATIVE path in its output; downstream steps
 * resolve it through this capability (or point an `agent` step's `cwd` at
 * `dir(ctx.runId)` so the built-in file tools see the same files).
 *
 * Artifacts are retained after the run ends — they're part of the run's
 * record, not scratch space (`onRunEnd` does not touch them).
 */
export interface ArtifactsCapability {
  /** Absolute path of the run's artifact directory, created on demand. */
  dir(runId: string): Promise<string>;
  /** Write `content` at `relPath` under the run's dir (subdirectories are
   *  created). Returns the absolute path of the written file. */
  write(runId: string, relPath: string, content: string | Uint8Array): Promise<string>;
  /** Read the file at `relPath` under the run's dir, as bytes.
   *  (`Buffer.from(bytes).toString()` for text.) */
  read(runId: string, relPath: string): Promise<Uint8Array>;
  /** Relative paths of every file under the run's dir (recursive, sorted).
   *  `[]` when the run has no artifacts. */
  list(runId: string): Promise<string[]>;
}

/** Reject run ids / relative paths that could escape the RUN's directory
 *  (one run must not reach another run's files). Returns the resolved
 *  absolute path when safe. */
function artifactPath(root: string, runId: string, relPath = ""): string {
  if (!runId || /[/\\]|\.\./.test(runId)) {
    throw new Error(`artifacts: invalid runId "${runId}"`);
  }
  const runDir = pathResolve(root, runId);
  const abs = pathResolve(runDir, relPath);
  if (abs !== runDir && !abs.startsWith(runDir + pathSep)) {
    throw new Error(`artifacts: path escapes the artifact root: ${relPath}`);
  }
  return abs;
}

/** Filesystem-backed artifacts capability rooted at `root`
 *  (`<root>/<runId>/<relPath>`). The default the standard server injects,
 *  rooted at `<workspace>/artifacts`. */
export function fileArtifactsCapability(root: string): ArtifactsCapability {
  return {
    async dir(runId) {
      const d = artifactPath(root, runId);
      await mkdir(d, { recursive: true });
      return d;
    },
    async write(runId, relPath, content) {
      const abs = artifactPath(root, runId, relPath);
      await mkdir(pathDirname(abs), { recursive: true });
      await writeFile(abs, content);
      return abs;
    },
    async read(runId, relPath) {
      const abs = artifactPath(root, runId, relPath);
      return new Uint8Array(await readFile(abs));
    },
    async list(runId) {
      const d = artifactPath(root, runId);
      let entries;
      try {
        entries = await readdir(d, { recursive: true, withFileTypes: true });
      } catch (err: any) {
        if (err?.code === "ENOENT") return [];
        throw err;
      }
      return entries
        .filter((e) => e.isFile())
        .map((e) => pathRelative(d, pathJoin(e.parentPath, e.name)))
        .sort();
    },
  };
}

// ── standard bag ───────────────────────────────────────────────────────────

/** The standard capability shape adapters rely on. Consumers extend this with
 *  their own typed services (graph store, llm client, …). */
export interface VeinCapabilities {
  http: HttpCapability;
  secrets: SecretsCapability;
  /** Per-run artifact files. Present on the standard server (rooted in the
   *  workspace); optional because a bare in-code bag may not carry one. */
  artifacts?: ArtifactsCapability;
}

/** The default standard services bag: global-fetch http + secrets. Secrets are
 *  env-backed by default; pass `secretStore` for a persisted (UI-managed) store
 *  with `process.env` as fallback. Injected by the standard server; override
 *  per environment as needed. */
export function standardServices(
  opts: {
    fetchImpl?: FetchLike;
    secretsSource?: Record<string, string | undefined>;
    secretStore?: SecretReadable;
  } = {},
): VeinCapabilities {
  const secrets = opts.secretStore
    ? secretsCapability(opts.secretStore, { envFallback: process.env })
    : secretsCapability(opts.secretsSource);
  return {
    http: httpCapability(opts.fetchImpl),
    secrets,
  };
}
