/**
 * NOT a seeded step — the canonical description of the per-step preamble for
 * the sheets/* steps.
 *
 * Seeded steps must be SELF-CONTAINED (value-imports from "vein" and node
 * builtins only), so each step file inlines its own copy of the auth/request
 * preamble (`sheetsCtx` + helpers below). If you change the contract, update
 * every step in this directory — they are deliberately duplicated, not shared.
 *
 * Contract (ported from `mcp/src/repo/toolsGoogleSheets.ts` — do not modify
 * that file; it stays the production repo-agent implementation):
 *   - Credentials: `cfg.serviceAccount` (explicit step config — parsed JSON
 *     object, JSON string, or base64 JSON) wins, else
 *     `ctx.services.secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON")` (secret store
 *     → env fallback). Missing credentials THROW a loud per-run error — the
 *     steps are always seeded, never silently absent.
 *   - `cfg.driveFolderId` wins, else the `GOOGLE_DRIVE_FOLDER_ID` secret.
 *     Spreadsheets are created inside that folder (share it with the service
 *     account's client_email so humans can see agent-created sheets).
 *   - Auth: plain service-account JWT flow — RS256 assertion built with
 *     `node:crypto` (`createSign("RSA-SHA256")`, base64url header+claims; no
 *     jsonwebtoken dep), exchanged at the SA's `token_uri` (default
 *     `https://oauth2.googleapis.com/token`) for a bearer token with the
 *     `spreadsheets` + `drive` scopes. The access token is cached at module
 *     level per step file, keyed by client_email, with 60s expiry slack.
 *   - Every request (token exchange included) goes through
 *     `ctx.services.http` and credentials resolve through
 *     `ctx.services.secrets`, so runs are cassette-recordable with secret
 *     values scrubbed from fixtures.
 *   - API errors are returned as readable "teaching" strings (`errorResult`),
 *     never thrown at the LLM; only missing/invalid credentials throw.
 */
export {};
