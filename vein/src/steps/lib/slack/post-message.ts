import { z } from "zod";
import { defineStep, type StepContext } from "../../../core.js";
import type { VeinCapabilities } from "../../../capabilities.js";
import { slackToken, slackCall } from "./_shared.js";

const EXAMPLE = `- id: notify
  type: slack/post-message
  config:
    channel: "C0123ABCD"
    text: "Build {{ build.status }} for {{ input.repo }}"
    token: "{{ input.slackToken }}"`;

export default defineStep({
  type: "slack/post-message",
  description: `Post a message to a Slack channel (or thread). Auth: a bot token (xoxb-…) via \`token\` config or the SLACK_BOT_TOKEN secret; the bot needs the chat:write scope and must be a member of the channel. Output: { ts, channel } — pass \`ts\` back as a later step's \`thread_ts\` to reply in-thread.\n\n${EXAMPLE}`,
  input: z.object({
    /** Channel ID (e.g. C0123ABCD) — recommended — or a #channel name. */
    channel: z.string().min(1),
    /** Message text (markdown-ish "mrkdwn"). Required unless `blocks` is set. */
    text: z.string().optional(),
    /** Slack Block Kit blocks, for rich messages. Overrides `text` layout. */
    blocks: z.array(z.record(z.unknown())).optional(),
    /** Reply in a thread by passing the parent message's `ts`. */
    thread_ts: z.string().optional(),
    token: z.string().optional(),
  }),
  output: z.object({
    ts: z.string(),
    channel: z.string(),
  }),
  async run(cfg, ctx: StepContext<VeinCapabilities>) {
    if (!cfg.text && !cfg.blocks) {
      throw new Error("slack/post-message needs `text` or `blocks`.");
    }
    const token = await slackToken(cfg.token, ctx);

    const data = await slackCall(ctx, "chat.postMessage", token, {
      body: {
        channel: cfg.channel,
        ...(cfg.text ? { text: cfg.text } : {}),
        ...(cfg.blocks ? { blocks: cfg.blocks } : {}),
        ...(cfg.thread_ts ? { thread_ts: cfg.thread_ts } : {}),
      },
    });

    return {
      ts: (data.ts as string) ?? "",
      channel: (data.channel as string) ?? cfg.channel,
    };
  },
});
