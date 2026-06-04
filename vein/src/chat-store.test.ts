import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { readFile, rm, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { randomUUID } from "node:crypto";

import {
  FileChatStore,
  MemoryChatStore,
  truncateToolMessages,
  isChatTerminal,
  generateChatId,
  type ChatEvent,
} from "./chat-store.js";

const ev = (
  chatId: string,
  turn: number,
  type: ChatEvent["type"],
  extra: Partial<ChatEvent> = {},
): ChatEvent => ({
  ts: new Date().toISOString(),
  chatId,
  turn,
  type,
  ...extra,
});

// ── MemoryChatStore ────────────────────────────────────────────────────────

describe("MemoryChatStore", () => {
  it("creates a chat with status live and currentTurn -1", async () => {
    const store = new MemoryChatStore();
    const meta = await store.createChat({ id: "c1", title: "hi" });
    assert.equal(meta.status, "live");
    assert.equal(meta.currentTurn, -1);
    assert.equal(meta.title, "hi");
    assert.deepEqual(await store.getMeta("c1"), meta);
  });

  it("appends + loads messages losslessly", async () => {
    const store = new MemoryChatStore();
    await store.createChat({ id: "c1" });
    await store.appendMessages("c1", [{ role: "user", content: "a" }]);
    await store.appendMessages("c1", [{ role: "assistant", content: "b" }]);
    assert.deepEqual(await store.loadMessages("c1"), [
      { role: "user", content: "a" },
      { role: "assistant", content: "b" },
    ]);
  });

  it("setMeta patches + bumps updatedAt; returns null for missing chat", async () => {
    const store = new MemoryChatStore();
    await store.createChat({ id: "c1" });
    const next = await store.setMeta("c1", { status: "done", currentTurn: 0 });
    assert.equal(next?.status, "done");
    assert.equal(next?.currentTurn, 0);
    assert.equal(await store.setMeta("missing", { status: "done" }), null);
  });

  it("tailEvents yields only the requested turn and stops at its terminal", async () => {
    const store = new MemoryChatStore();
    await store.createChat({ id: "c1" });
    await store.appendEvent("c1", ev("c1", 0, "text-delta", { delta: "x" }));
    await store.appendEvent("c1", ev("c1", 0, "chat.end"));
    await store.appendEvent("c1", ev("c1", 1, "text-delta", { delta: "y" }));
    await store.appendEvent("c1", ev("c1", 1, "chat.end"));

    const turn1: ChatEvent[] = [];
    for await (const e of store.tailEvents("c1", 1)) turn1.push(e);
    assert.deepEqual(turn1.map((e) => e.type), ["text-delta", "chat.end"]);
    assert.equal(turn1[0]!.delta, "y");
  });

  it("listChats sorts newest-updated first", async () => {
    const store = new MemoryChatStore();
    await store.createChat({ id: "a" });
    await new Promise((r) => setTimeout(r, 5));
    await store.createChat({ id: "b" });
    await store.setMeta("a", { status: "done" }); // touches updatedAt
    const list = await store.listChats();
    assert.equal(list[0]!.id, "a");
  });
});

// ── FileChatStore ──────────────────────────────────────────────────────────

describe("FileChatStore", () => {
  let tempDir: string;

  beforeEach(async () => {
    tempDir = join(tmpdir(), `vein-chat-${randomUUID()}`);
    await mkdir(tempDir, { recursive: true });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("persists meta.json + messages.jsonl + events.jsonl under chats/<id>/", async () => {
    const store = new FileChatStore(tempDir);
    await store.createChat({ id: "c1", title: "t", model: "m" });
    await store.appendMessages("c1", [{ role: "user", content: "hello" }]);
    await store.appendEvent("c1", ev("c1", 0, "text-delta", { delta: "hi" }));

    const meta = JSON.parse(
      await readFile(join(tempDir, "chats", "c1", "meta.json"), "utf-8"),
    );
    assert.equal(meta.id, "c1");
    assert.equal(meta.model, "m");

    const msgs = await readFile(join(tempDir, "chats", "c1", "messages.jsonl"), "utf-8");
    assert.equal(msgs.trim().split("\n").length, 1);

    const events = await readFile(join(tempDir, "chats", "c1", "events.jsonl"), "utf-8");
    assert.ok(events.includes('"text-delta"'));
  });

  it("loadMessages survives a fresh store instance (durable)", async () => {
    const a = new FileChatStore(tempDir);
    await a.createChat({ id: "c1" });
    await a.appendMessages("c1", [{ role: "user", content: "x" }]);

    const b = new FileChatStore(tempDir);
    assert.deepEqual(await b.loadMessages("c1"), [{ role: "user", content: "x" }]);
  });

  it("tailEvents drains a completed turn then returns", async () => {
    const store = new FileChatStore(tempDir);
    await store.createChat({ id: "c1" });
    await store.appendEvent("c1", ev("c1", 0, "text-delta", { delta: "a" }));
    await store.appendEvent("c1", ev("c1", 0, "tool-input", { toolName: "x" }));
    await store.appendEvent("c1", ev("c1", 0, "chat.end"));

    const seen: ChatEvent[] = [];
    for await (const e of store.tailEvents("c1", 0, { intervalMs: 10 })) seen.push(e);
    assert.deepEqual(seen.map((e) => e.type), ["text-delta", "tool-input", "chat.end"]);
  });

  it("tailEvents replays a multi-turn log but stops at the requested turn's terminal", async () => {
    const store = new FileChatStore(tempDir);
    await store.createChat({ id: "c1" });
    await store.appendEvent("c1", ev("c1", 0, "text-delta", { delta: "0" }));
    await store.appendEvent("c1", ev("c1", 0, "chat.end"));
    await store.appendEvent("c1", ev("c1", 1, "text-delta", { delta: "1" }));
    await store.appendEvent("c1", ev("c1", 1, "chat.end"));

    const seen: ChatEvent[] = [];
    for await (const e of store.tailEvents("c1", 1, { intervalMs: 10 })) seen.push(e);
    assert.deepEqual(seen.map((e) => e.delta), ["1", undefined]);
    assert.deepEqual(seen.map((e) => e.type), ["text-delta", "chat.end"]);
  });

  it("tailEvents follows a live turn until its terminal (detached reattach)", async () => {
    const store = new FileChatStore(tempDir);
    await store.createChat({ id: "c1" });
    await store.appendEvent("c1", ev("c1", 0, "text-delta", { delta: "start" }));

    const seen: ChatEvent[] = [];
    const consume = (async () => {
      for await (const e of store.tailEvents("c1", 0, { intervalMs: 10 })) seen.push(e);
    })();

    await new Promise((r) => setTimeout(r, 30));
    await store.appendEvent("c1", ev("c1", 0, "tool-input", { toolName: "run_workflow" }));
    await new Promise((r) => setTimeout(r, 30));
    await store.appendEvent("c1", ev("c1", 0, "chat.end"));

    await consume;
    assert.deepEqual(seen.map((e) => e.type), ["text-delta", "tool-input", "chat.end"]);
  });
});

// ── Helpers ────────────────────────────────────────────────────────────────

describe("truncateToolMessages", () => {
  it("truncates long strings inside tool messages, leaving others intact", () => {
    const big = "x".repeat(5000);
    const out = truncateToolMessages(
      [
        { role: "assistant", content: [{ type: "text", text: big }] },
        { role: "tool", content: [{ type: "tool-result", output: { value: big } }] },
      ],
      4000,
    );
    // Assistant message untouched (only role:"tool" is trimmed).
    assert.equal((out[0]!.content as any)[0].text.length, 5000);
    // Tool result truncated + marked.
    const toolText = (out[1]!.content as any)[0].output.value as string;
    assert.ok(toolText.length < 5000);
    assert.ok(toolText.includes("[TRUNCATED"));
  });

  it("leaves short tool content unchanged", () => {
    const msgs = [{ role: "tool", content: [{ type: "tool-result", output: "ok" }] }];
    assert.deepEqual(truncateToolMessages(msgs, 4000), msgs);
  });
});

describe("isChatTerminal / generateChatId", () => {
  it("flags chat.end and chat.error as terminal", () => {
    assert.equal(isChatTerminal(ev("c", 0, "chat.end")), true);
    assert.equal(isChatTerminal(ev("c", 0, "chat.error")), true);
    assert.equal(isChatTerminal(ev("c", 0, "text-delta")), false);
  });

  it("generates unique-ish sortable ids", () => {
    const a = generateChatId();
    const b = generateChatId();
    assert.notEqual(a, b);
    assert.match(a, /^[a-z0-9]+-[a-z0-9]+$/);
  });
});
