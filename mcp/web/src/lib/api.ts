// Auth-aware fetch wrapper. Reads the JWT injected into index.html by the
// Express server (see mcp/src/index.ts) and attaches it as a Bearer token.
//
// The token is short-lived (1h). If a request 401s, we reload the page to
// re-trigger the Basic Auth flow and pick up a fresh token.

declare global {
  interface Window {
    __AUTH_TOKEN__?: string;
  }
}

export function getAuthToken(): string | undefined {
  if (typeof window === "undefined") return undefined;
  return window.__AUTH_TOKEN__ || undefined;
}

export function authHeaders(): Record<string, string> {
  const token = getAuthToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/**
 * Drop-in replacement for fetch() that injects the auth header and
 * reloads the page on 401 to re-trigger Basic Auth.
 */
export async function apiFetch(
  input: string,
  init: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(init.headers || {});
  const token = getAuthToken();
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  const res = await fetch(input, { ...init, headers });
  if (res.status === 401 && typeof window !== "undefined") {
    // Token expired or missing — reload to get a fresh one via Basic Auth
    window.location.reload();
  }
  return res;
}
