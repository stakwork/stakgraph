// Tiny typed fetch wrapper. Every API call in the SPA goes through
// `apiFetch` so:
//
//   - Cookies are always sent (`credentials: "include"`).
//   - 401 throws a sentinel error that the top-level QueryClient
//     handler converts into a /login redirect (see app.tsx).
//   - Non-2xx responses are surfaced with the phase-7 error code so
//     UI banners can present something useful.
//   - The `/_plugin` prefix is centralised — pages just pass routes.
//
// This file is < 100 lines on purpose. Adding fancy retry, caching,
// or auth-refresh logic here lights the trap of having two HTTP
// stacks; Tanstack Query gives us all of that one layer up.

import type { ApiError } from "./types";

const PLUGIN_PREFIX = "/_plugin";

/** Sentinel error thrown on 401. The QueryClient's `onError` catches
 *  it and triggers a redirect to /login?next=<current>. Importable
 *  from app.tsx via `instanceof UnauthorizedError`. */
export class UnauthorizedError extends Error {
  constructor() {
    super("unauthorized");
    this.name = "UnauthorizedError";
  }
}

/** Generic ApiError-bearing failure for everything that isn't a 401.
 *  Pages render `err.code` next to a banner via getErrorMessage(). */
export class ApiCallError extends Error {
  readonly code: string;
  readonly status: number;
  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = "ApiCallError";
    this.code = code;
    this.status = status;
  }
}

export interface ApiFetchOptions {
  method?: "GET" | "POST" | "PATCH" | "DELETE";
  body?: unknown;
  /** Extra headers (e.g. Authorization: Basic on the login call). */
  headers?: Record<string, string>;
  /** Timeout in ms. Default 5s — matches the plugin's own ~5s
   *  upstream timeouts so the UI doesn't outwait its backend. */
  timeoutMs?: number;
}

/** Issues a request against /_plugin/<path> and returns the parsed
 *  JSON body (typed at the call site). On non-2xx, throws either
 *  UnauthorizedError (401) or ApiCallError (anything else). */
export async function apiFetch<T>(
  path: string,
  opts: ApiFetchOptions = {}
): Promise<T> {
  const { method = "GET", body, headers = {}, timeoutMs = 5000 } = opts;

  const ctl = new AbortController();
  const tid = setTimeout(() => ctl.abort(), timeoutMs);

  // CSRF: the gateway requires `X-Bifrost-CSRF` on every cookie-
  // authed mutation. Browsers won't auto-send a custom header on
  // cross-origin requests, so its presence is the signal that "this
  // request was made by our own SPA, not a forged form post." Any
  // value works — we use "1" for brevity. GET / HEAD are exempt
  // server-side, but adding the header unconditionally keeps the
  // request shape uniform and is harmless.
  let resp: Response;
  try {
    resp = await fetch(PLUGIN_PREFIX + path, {
      method,
      credentials: "include",
      headers: {
        ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
        Accept: "application/json",
        "X-Bifrost-CSRF": "1",
        ...headers,
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: ctl.signal,
    });
  } catch (e) {
    clearTimeout(tid);
    if ((e as Error).name === "AbortError") {
      throw new ApiCallError(0, "timeout", "request timed out");
    }
    throw new ApiCallError(0, "network", (e as Error).message);
  }
  clearTimeout(tid);

  if (resp.status === 401) {
    throw new UnauthorizedError();
  }

  if (resp.status === 204) {
    // No body — return undefined as the typed result. Callers that
    // expect T must use `void` / `undefined` at the call site.
    return undefined as T;
  }

  if (!resp.ok) {
    let env: ApiError | undefined;
    try {
      env = (await resp.json()) as ApiError;
    } catch {
      /* body wasn't JSON — fall through with a synthetic code */
    }
    throw new ApiCallError(
      resp.status,
      env?.error?.code ?? "http_error",
      env?.error?.message ?? `HTTP ${resp.status}`
    );
  }

  return (await resp.json()) as T;
}

/** Convenience for one-off POSTs that don't need an envelope. */
export function apiPost<T>(path: string, body?: unknown, headers?: Record<string, string>) {
  return apiFetch<T>(path, { method: "POST", body, headers });
}

/** Read a useful human string from any error thrown by apiFetch. */
export function getErrorMessage(err: unknown): string {
  if (err instanceof ApiCallError) return `${err.code}: ${err.message}`;
  if (err instanceof Error) return err.message;
  return String(err);
}
