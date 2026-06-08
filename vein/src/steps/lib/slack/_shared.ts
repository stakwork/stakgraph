// Shared helpers for the slack/* lib steps. Leading-underscore file → imported
// by siblings, skipped by registry discovery (see AGENTS.md). Slack adapters go
// through ctx.services.http (raw REST — no SDK needed, fully recordable) rather
// than @slack/web-api, per the "prefer raw REST" authoring convention.
import type { StepContext } from "../../../core.js";
import type { VeinCapabilities } from "../../../capabilities.js";

const SLACK_API = "https://slack.com/api";

/** Resolve a Slack bot token: explicit config wins, else the SLACK_BOT_TOKEN
 *  secret (UI-managed store → env). Throws an actionable error if neither. */
export async function slackToken(
  explicit: string | undefined,
  ctx: StepContext<VeinCapabilities>,
): Promise<string> {
  const token =
    explicit ?? (await ctx?.services?.secrets?.get("SLACK_BOT_TOKEN"));
  if (!token) {
    throw new Error(
      "No Slack token. Pass `token` in config or add a SLACK_BOT_TOKEN secret (a bot token, starts with xoxb-).",
    );
  }
  return token;
}

/** Call a Slack Web API method and return its (ok:true) payload.
 *
 *  Slack is unusual: it returns HTTP 200 even on logical failures, signalling
 *  the real outcome in the JSON body (`{ ok: false, error: "channel_not_found" }`).
 *  So we check `body.ok`, NOT the HTTP status, and map the common `error`
 *  codes to actionable messages. */
export async function slackCall(
  ctx: StepContext<VeinCapabilities>,
  method: string,
  token: string,
  opts: {
    body?: unknown;
    query?: Record<string, string | number | boolean>;
  } = {},
): Promise<Record<string, unknown>> {
  const http = ctx?.services?.http;
  if (!http) throw new Error("slack steps require ctx.services.http");

  const res = await http(`${SLACK_API}/${method}`, {
    method: opts.body !== undefined ? "POST" : "GET",
    headers: { authorization: `Bearer ${token}` },
    body: opts.body,
    query: opts.query,
  });

  const data = (res.body ?? {}) as Record<string, unknown>;
  // A non-2xx with no JSON body (proxy error, 5xx) won't have `ok` at all.
  if (data.ok !== true) {
    const code =
      typeof data.error === "string" ? data.error : `http_${res.status}`;
    throw describeSlackError(code, method);
  }
  return data;
}

/** Map a Slack `error` code to an actionable Error. Unknown codes pass through
 *  verbatim so nothing is swallowed. */
export function describeSlackError(error: string, method: string): Error {
  const hints: Record<string, string> = {
    channel_not_found:
      "channel not found — use the channel ID (e.g. C0123ABCD) and make sure the bot has been added to it",
    not_in_channel:
      "the bot isn't a member of that channel — /invite it, or post to a channel it has joined",
    is_archived: "the channel is archived",
    missing_scope: `the bot token is missing a required OAuth scope for ${method} (e.g. chat:write to post, channels:history to read, users:read to resolve names)`,
    invalid_auth:
      "invalid Slack token — check SLACK_BOT_TOKEN (a bot token starts with xoxb-)",
    not_authed: "no Slack token was sent — set SLACK_BOT_TOKEN",
    token_revoked: "the Slack token has been revoked — generate a new bot token",
    account_inactive: "the Slack token's bot/user has been deactivated",
    msg_too_long: "the message text is too long for Slack",
    rate_limited: "rate limited by Slack — retry after a short delay",
  };
  const hint = hints[error] ?? error;
  return new Error(`Slack ${method} failed: ${hint} (error: ${error})`);
}
