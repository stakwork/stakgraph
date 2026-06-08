import { z } from "zod";
import { defineStep, type StepContext } from "../../../core.js";
import type { VeinCapabilities } from "../../../capabilities.js";
import { slackToken, slackCall } from "./_shared.js";

const EXAMPLE = `- id: history
  type: slack/read-channel
  config:
    channel: "C0123ABCD"
    limit: 50
    token: "{{ input.slackToken }}"`;

export interface SlackMessage {
  ts: string;
  user: string;
  text: string;
  threadTs: string | null;
}

export default defineStep({
  type: "slack/read-channel",
  description: `Read recent messages from a Slack channel and format them as markdown for LLM consumption. Returns messages oldest→newest with author names resolved. Auth: a bot token (xoxb-…) via \`token\` or the SLACK_BOT_TOKEN secret; the bot needs channels:history (and users:read to resolve names) and must be a member of the channel. Output: { markdown, messages: [{ ts, user, text, threadTs }], channel, hasMore }.\n\n${EXAMPLE}`,
  input: z.object({
    /** Channel ID (e.g. C0123ABCD). */
    channel: z.string().min(1),
    /** Max messages to fetch (Slack caps a page at 1000). */
    limit: z.number().int().positive().max(1000).default(50),
    /** Only messages after this Unix ts (Slack `oldest` cursor). */
    oldest: z.string().optional(),
    /** Only messages before this Unix ts (Slack `latest` cursor). */
    latest: z.string().optional(),
    /** Resolve user IDs → display names via users.info (one call per unique
     *  participant, fail-soft to the ID). Set false to skip (no users:read). */
    resolveUsers: z.boolean().default(true),
    token: z.string().optional(),
  }),
  output: z.object({
    markdown: z.string(),
    channel: z.string(),
    hasMore: z.boolean(),
    messages: z.array(
      z.object({
        ts: z.string(),
        user: z.string(),
        text: z.string(),
        threadTs: z.string().nullable(),
      }),
    ),
  }),
  async run(cfg, ctx: StepContext<VeinCapabilities>) {
    const token = await slackToken(cfg.token, ctx);

    const data = await slackCall(ctx, "conversations.history", token, {
      query: {
        channel: cfg.channel,
        limit: cfg.limit,
        ...(cfg.oldest ? { oldest: cfg.oldest } : {}),
        ...(cfg.latest ? { latest: cfg.latest } : {}),
      },
    });

    const raw = (data.messages as Record<string, unknown>[]) ?? [];
    const userMap = cfg.resolveUsers
      ? await resolveUserNames(ctx, token, raw)
      : {};

    const messages: SlackMessage[] = raw.map((m) => ({
      ts: (m.ts as string) ?? "",
      user: authorOf(m, userMap),
      text: (m.text as string) ?? "",
      threadTs: (m.thread_ts as string) ?? null,
    }));
    // Slack returns newest-first; flip to chronological for natural reading.
    messages.reverse();

    return {
      markdown: formatChannel(cfg.channel, messages),
      channel: cfg.channel,
      hasMore: data.has_more === true,
      messages,
    };
  },
});

// ── helpers ────────────────────────────────────────────────────────────────

/** Best-effort author label: resolved display name, else bot username, else
 *  the raw id. */
function authorOf(
  m: Record<string, unknown>,
  userMap: Record<string, string>,
): string {
  const userId = m.user as string | undefined;
  if (userId) return userMap[userId] ?? userId;
  return (m.username as string) ?? (m.bot_id as string) ?? "unknown";
}

/** Resolve unique user IDs → display names. One users.info per id (in
 *  parallel); a failure (e.g. missing users:read scope) falls back to the id
 *  so reading never breaks on name resolution. */
async function resolveUserNames(
  ctx: StepContext<VeinCapabilities>,
  token: string,
  messages: Record<string, unknown>[],
): Promise<Record<string, string>> {
  const ids = [
    ...new Set(messages.map((m) => m.user as string).filter(Boolean)),
  ];
  const entries = await Promise.all(
    ids.map(async (id) => {
      try {
        const info = await slackCall(ctx, "users.info", token, {
          query: { user: id },
        });
        const user = info.user as
          | {
              profile?: { display_name?: string; real_name?: string };
              real_name?: string;
              name?: string;
            }
          | undefined;
        const name =
          user?.profile?.display_name ||
          user?.profile?.real_name ||
          user?.real_name ||
          user?.name ||
          id;
        return [id, name] as const;
      } catch {
        return [id, id] as const;
      }
    }),
  );
  return Object.fromEntries(entries);
}

function formatChannel(channel: string, messages: SlackMessage[]): string {
  const lines = [`# Slack channel ${channel} (${messages.length} messages)`];
  for (const m of messages) {
    const when = formatTs(m.ts);
    const threadNote = m.threadTs ? " (in thread)" : "";
    lines.push(`\n**@${m.user}** — ${when}${threadNote}`);
    lines.push(`> ${m.text.replace(/\n/g, "\n> ")}`);
  }
  return lines.join("\n");
}

/** Slack ts is "<seconds>.<microseconds>" — render as a readable date. */
function formatTs(ts: string): string {
  const seconds = Number(ts.split(".")[0]);
  if (!Number.isFinite(seconds)) return ts;
  return new Date(seconds * 1000).toISOString();
}
