/**
 * NOT a seeded step — shared TYPES + the canonical copy of the per-step
 * preamble for the jarvis/* steps.
 *
 * Seeded steps must be SELF-CONTAINED (value-imports from "vein" only), so
 * each step file inlines its own copy of the small `jarvisCtx` helper below.
 * If you change the contract (env names, auth header), update every step in
 * this directory — they are deliberately duplicated, not shared.
 *
 * Contract:
 *   - `JARVIS_URL`  — base URL of the Jarvis backend (same-compose container).
 *   - `API_TOKEN`   — sent as `X-Api-Token` (the same shared secret mcp uses).
 *   - `JARVIS_HTTP_TIMEOUT_MS` — optional request timeout (default 180s).
 * All three resolve through `ctx.services.secrets` (secret store → env
 * fallback), and every request goes through `ctx.services.http`, so the steps
 * are cassette-recordable and secret values are scrubbed from fixtures.
 */
export {};
