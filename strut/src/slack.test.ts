import { describe, it } from "node:test";
import assert from "node:assert/strict";
import type { HttpResponse } from "./capabilities.js";
import postMessage from "./steps/lib/slack/post-message.js";
import readChannel from "./steps/lib/slack/read-channel.js";

// ── Fake ctx ────────────────────────────────────────────────────────────────
//
// Slack steps go through ctx.services.http (raw REST). We stub it with a
// per-method response map keyed by the Slack API method (last URL segment),
// and record the calls so we can assert what was sent.

interface Call {
  method: string;
  reqMethod: string;
  body?: unknown;
  query?: Record<string, unknown>;
  auth?: string;
}

function makeCtx(
  responses: Record<string, unknown>,
  opts: { secrets?: Record<string, string>; status?: number } = {},
) {
  const calls: Call[] = [];
  const http = async (
    url: string,
    o: {
      method?: string;
      headers?: Record<string, string>;
      body?: unknown;
      query?: Record<string, string | number | boolean>;
    } = {},
  ): Promise<HttpResponse> => {
    const apiMethod = url.split("/").pop()!;
    calls.push({
      method: apiMethod,
      reqMethod: o.method ?? "GET",
      body: o.body,
      query: o.query,
      auth: o.headers?.authorization,
    });
    const body = responses[apiMethod] ?? { ok: false, error: "unknown_method" };
    return { status: opts.status ?? 200, ok: true, headers: {}, body };
  };
  const ctx = {
    runId: "t",
    path: "t",
    scope: {},
    input: {},
    emit: async () => {},
    services: {
      http,
      secrets: { get: async (n: string) => opts.secrets?.[n] },
    },
  } as never;
  return { ctx, calls };
}

// ── post-message ─────────────────────────────────────────────────────────────

describe("slack/post-message", () => {
  it("posts text and returns ts + channel", async () => {
    const { ctx, calls } = makeCtx({
      "chat.postMessage": { ok: true, ts: "1700000000.000100", channel: "C1" },
    });
    const out = await postMessage.run(
      { channel: "C1", text: "hello", token: "xoxb-test" },
      ctx,
    );
    assert.deepEqual(out, { ts: "1700000000.000100", channel: "C1" });
    assert.equal(calls[0]!.method, "chat.postMessage");
    assert.equal(calls[0]!.reqMethod, "POST");
    assert.equal(calls[0]!.auth, "Bearer xoxb-test");
    assert.deepEqual(calls[0]!.body, { channel: "C1", text: "hello" });
  });

  it("threads via thread_ts and supports blocks-only", async () => {
    const { ctx, calls } = makeCtx({
      "chat.postMessage": { ok: true, ts: "2.0", channel: "C1" },
    });
    await postMessage.run(
      {
        channel: "C1",
        blocks: [{ type: "section" }],
        thread_ts: "1.0",
        token: "x",
      },
      ctx,
    );
    assert.deepEqual(calls[0]!.body, {
      channel: "C1",
      blocks: [{ type: "section" }],
      thread_ts: "1.0",
    });
  });

  it("requires text or blocks", async () => {
    const { ctx } = makeCtx({});
    await assert.rejects(
      () => postMessage.run({ channel: "C1", token: "x" }, ctx),
      /needs `text` or `blocks`/,
    );
  });

  it("maps ok:false to an actionable error", async () => {
    const { ctx } = makeCtx({
      "chat.postMessage": { ok: false, error: "not_in_channel" },
    });
    await assert.rejects(
      () => postMessage.run({ channel: "C1", text: "hi", token: "x" }, ctx),
      /isn't a member.*error: not_in_channel/s,
    );
  });

  it("reads the token from the SLACK_BOT_TOKEN secret", async () => {
    const { ctx, calls } = makeCtx(
      { "chat.postMessage": { ok: true, ts: "1.0", channel: "C1" } },
      { secrets: { SLACK_BOT_TOKEN: "xoxb-secret" } },
    );
    await postMessage.run({ channel: "C1", text: "hi" }, ctx);
    assert.equal(calls[0]!.auth, "Bearer xoxb-secret");
  });

  it("errors clearly when no token is available", async () => {
    const { ctx } = makeCtx({});
    await assert.rejects(
      () => postMessage.run({ channel: "C1", text: "hi" }, ctx),
      /No Slack token/,
    );
  });
});

// ── read-channel ─────────────────────────────────────────────────────────────

describe("slack/read-channel", () => {
  it("returns messages chronologically with resolved names + markdown", async () => {
    const { ctx, calls } = makeCtx({
      "conversations.history": {
        ok: true,
        has_more: false,
        // Slack returns newest-first:
        messages: [
          { ts: "1700000002.0", user: "U2", text: "second" },
          { ts: "1700000001.0", user: "U1", text: "first" },
        ],
      },
      "users.info": { ok: true, user: { profile: { display_name: "alice" } } },
    });
    const out = await readChannel.run(
      { channel: "C1", limit: 50, resolveUsers: true, token: "x" },
      ctx,
    );

    // chronological (oldest first)
    assert.deepEqual(
      out.messages.map((m) => m.text),
      ["first", "second"],
    );
    assert.equal(out.hasMore, false);
    assert.equal(out.messages[0]!.user, "alice");
    assert.match(out.markdown, /# Slack channel C1 \(2 messages\)/);
    assert.match(out.markdown, /\*\*@alice\*\*/);

    // history is a GET with channel+limit in the query
    const hist = calls.find((c) => c.method === "conversations.history")!;
    assert.equal(hist.reqMethod, "GET");
    assert.deepEqual(hist.query, { channel: "C1", limit: 50 });
  });

  it("falls back to the user id when name resolution fails", async () => {
    const { ctx } = makeCtx({
      "conversations.history": {
        ok: true,
        messages: [{ ts: "1.0", user: "U9", text: "hi" }],
      },
      "users.info": { ok: false, error: "missing_scope" },
    });
    const out = await readChannel.run(
      { channel: "C1", limit: 50, resolveUsers: true, token: "x" },
      ctx,
    );
    assert.equal(out.messages[0]!.user, "U9");
  });

  it("skips user resolution when resolveUsers is false", async () => {
    const { ctx, calls } = makeCtx({
      "conversations.history": {
        ok: true,
        messages: [{ ts: "1.0", user: "U9", text: "hi" }],
      },
    });
    const out = await readChannel.run(
      { channel: "C1", limit: 50, resolveUsers: false, token: "x" },
      ctx,
    );
    assert.equal(out.messages[0]!.user, "U9");
    assert.equal(
      calls.some((c) => c.method === "users.info"),
      false,
    );
  });

  it("labels bot messages without a user id", async () => {
    const { ctx } = makeCtx({
      "conversations.history": {
        ok: true,
        messages: [{ ts: "1.0", username: "deploybot", text: "shipped" }],
      },
    });
    const out = await readChannel.run(
      { channel: "C1", limit: 50, resolveUsers: false, token: "x" },
      ctx,
    );
    assert.equal(out.messages[0]!.user, "deploybot");
  });

  it("maps channel_not_found to an actionable error", async () => {
    const { ctx } = makeCtx({
      "conversations.history": { ok: false, error: "channel_not_found" },
    });
    await assert.rejects(
      () =>
        readChannel.run(
          { channel: "bad", limit: 50, resolveUsers: false, token: "x" },
          ctx,
        ),
      /channel not found.*error: channel_not_found/s,
    );
  });
});
