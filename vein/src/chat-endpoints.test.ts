import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";

import { createVein } from "./createVein.js";
import { WorkspaceManager } from "./workspace.js";
import { MemoryRunStore } from "./store.js";
import { MemoryChatStore, type ChatEvent } from "./chat-store.js";

/**
 * HTTP plumbing for the detached chat endpoints. We inject a `MemoryChatStore`
 * and pre-seed it so we exercise the routes WITHOUT launching the real LLM
 * agent (the turn runner is covered indirectly; the model itself needs a key
 * + network). `POST /chat`'s happy path persists synchronously before the
 * detached turn fires, so we can assert that bit deterministically.
 */

describe("chat endpoints", () => {
  let tempDir: string;
  let chatStore: MemoryChatStore;

  async function makeVein() {
    return createVein({
      workspace: new WorkspaceManager(tempDir),
      store: new MemoryRunStore(),
      chatStore,
      serveUi: false,
    });
  }

  beforeEach(async () => {
    tempDir = join(tmpdir(), `vein-chat-ep-${randomUUID()}`);
    await mkdir(tempDir, { recursive: true });
    chatStore = new MemoryChatStore();
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("POST /chat rejects a missing message", async () => {
    const vein = await makeVein();
    const res = await vein.app.request("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    assert.equal(res.status, 400);
  });

  it("POST /chat with an unknown chatId is a 404", async () => {
    const vein = await makeVein();
    const res = await vein.app.request("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ chatId: "nope", message: "hi" }),
    });
    assert.equal(res.status, 404);
  });

  it("POST /chat creates a session + persists the user message synchronously", async () => {
    const vein = await makeVein();
    const res = await vein.app.request("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "build me a workflow" }),
    });
    assert.equal(res.status, 202);
    const { chatId, turn } = (await res.json()) as { chatId: string; turn: number };
    assert.ok(chatId);
    assert.equal(turn, 0);

    // The subject message is appended before the (detached) turn launches.
    const messages = await chatStore.loadMessages(chatId);
    assert.equal(messages[0]!.role, "user");
    assert.equal(messages[0]!.content, "build me a workflow");

    // currentTurn is set synchronously and isn't moved by the background turn.
    const meta = await chatStore.getMeta(chatId);
    assert.equal(meta!.currentTurn, 0);
  });

  it("GET /chat/:id returns the transcript + meta; 404 when missing", async () => {
    const vein = await makeVein();
    await chatStore.createChat({ id: "c1", title: "t" });
    await chatStore.appendMessages("c1", [{ role: "user", content: "x" }]);

    const ok = await vein.app.request("/chat/c1");
    assert.equal(ok.status, 200);
    const body = (await ok.json()) as { meta: any; messages: any[] };
    assert.equal(body.meta.id, "c1");
    assert.equal(body.messages.length, 1);

    const missing = await vein.app.request("/chat/missing");
    assert.equal(missing.status, 404);
  });

  it("GET /chats lists sessions", async () => {
    const vein = await makeVein();
    await chatStore.createChat({ id: "a" });
    await chatStore.createChat({ id: "b" });
    const res = await vein.app.request("/chats");
    const list = (await res.json()) as { id: string }[];
    assert.equal(list.length, 2);
  });

  it("GET /chat/:id/stream replays a completed turn then sends done", async () => {
    const vein = await makeVein();
    await chatStore.createChat({ id: "c1" });
    await chatStore.setMeta("c1", { status: "done", currentTurn: 0 });
    const ev = (type: ChatEvent["type"], extra: Partial<ChatEvent> = {}): ChatEvent => ({
      ts: new Date().toISOString(),
      chatId: "c1",
      turn: 0,
      type,
      ...extra,
    });
    await chatStore.appendEvent("c1", ev("text-delta", { delta: "hello" }));
    await chatStore.appendEvent("c1", ev("chat.end"));

    const res = await vein.app.request("/chat/c1/stream?turn=0");
    assert.equal(res.status, 200);
    const text = await res.text();
    assert.ok(text.includes("hello"), text);
    assert.ok(text.includes("event: done"), text);
  });

  it("GET /chat/:id/stream for a not-yet-started turn sends done immediately", async () => {
    const vein = await makeVein();
    await chatStore.createChat({ id: "c1" }); // currentTurn -1
    const res = await vein.app.request("/chat/c1/stream?turn=3");
    assert.equal(res.status, 200);
    const text = await res.text();
    assert.ok(text.includes("event: done"), text);
  });

  it("GET /chat/:id/stream is a 404 for an unknown chat", async () => {
    const vein = await makeVein();
    const res = await vein.app.request("/chat/missing/stream");
    assert.equal(res.status, 404);
  });
});
