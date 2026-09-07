import type { Context, Next } from "hono";

/**
 * Deployment-scoped shared-secret auth for step-registration mutations.
 *
 * If `VEIN_API_KEY` is set in the environment, every gated request must
 * present `Authorization: Bearer <key>` matching that value. If unset, the
 * middleware is permissive (dev mode) — it logs a one-time warning at boot
 * so the lax posture is visible.
 *
 * The same secret authenticates first-party services in both directions
 * within a deployment: mcp uses it to register steps with vein, and vein's
 * uploaded step files use it to call back to mcp. See AGENTS.md.
 */

const ENV_VAR = "VEIN_API_KEY";

let warned = false;

/** Read the configured key at request time. Returns undefined if unset. */
function configuredKey(): string | undefined {
  const v = process.env[ENV_VAR];
  return v && v.length > 0 ? v : undefined;
}

/** Emit a one-time stderr warning if running without a configured key. */
export function warnIfUnconfigured(): void {
  if (warned) return;
  warned = true;
  if (!configuredKey()) {
    console.warn(
      `[vein] ${ENV_VAR} is not set — step registration is unauthenticated (dev mode).`,
    );
  }
}

/**
 * Hono middleware that gates step-registration mutations on a bearer
 * token matching `VEIN_API_KEY`. Permissive when the env var is unset.
 */
export async function requireApiKey(c: Context, next: Next) {
  if (!apiKeyMatches(c.req.header("authorization"))) {
    return c.json(
      { error: "unauthorized: valid Authorization: Bearer <VEIN_API_KEY> required" },
      401,
    );
  }

  return next();
}

/**
 * Does a request carry the deployment key? Accepts `Authorization: Bearer`
 * and, when the caller passes it, a `?key=` query value — the WebSocket
 * dictation route needs the latter because a browser's WebSocket cannot set
 * headers. Permissive (true) when `VEIN_API_KEY` is unset.
 */
export function apiKeyMatches(authorization: string | undefined, queryKey?: string | null): boolean {
  const expected = configuredKey();
  if (!expected) return true;
  const match = (authorization ?? "").match(/^Bearer\s+(.+)$/i);
  const got = match?.[1]?.trim() || queryKey?.trim();
  return !!got && got === expected;
}

/** Test-only: reset the one-time-warning state so tests stay deterministic. */
export function _resetAuthState(): void {
  warned = false;
}
