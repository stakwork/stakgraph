import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { MemoryChatStore } from "../chat-store.js";
import type { StoredMessage } from "../chat-store.js";
import {
  createChatNotifier,
  formatRunNotification,
  NOTIFICATION_PREFIX,
} from "./notifier.js";

// ── formatRunNotification ──────────────────────────────────────────────────

describe("formatRunNotification", () => {
  it("renders a success with duration, output, and the get_run pointer", () => {
    const msg = formatRunNotification({
      workflow: "harvey-run",
      runId: "123",
      status: "success",
      durationMs: 41 * 60_000 + 12_000,
      output: { score: 0.82 },
    });
    assert.ok(msg.startsWith(`${NOTIFICATION_PREFIX} Workflow "harvey-run" run 123 finished: success in 41m 12s.`));
    assert.ok(msg.includes(`Output: {"score":0.82}`));
    assert.ok(msg.includes(`get_run("harvey-run", "123")`));
  });

  it("truncates large outputs", () => {
    const msg = formatRunNotification(
      { workflow: "wf", runId: "1", status: "success", output: { blob: "x".repeat(5000) } },
      100,
    );
    assert.ok(msg.includes("Output (truncated,"));
    // The included slice is bounded.
    const outputLine = msg.split("\n").find((l) => l.startsWith("x"));
    assert.ok(outputLine === undefined || outputLine.length <= 100);
    assert.ok(!msg.includes("x".repeat(200)));
  });

  it("renders an error without output", () => {
    const msg = formatRunNotification({
      workflow: "wf",
      runId: "9",
      status: "error",
      durationMs: 3000,
      error: { message: "step produce failed" },
    });
    assert.ok(msg.includes("finished: error in 3s."));
    assert.ok(msg.includes("Error: step produce failed"));
    assert.ok(!msg.includes("Output"));
  });
});

// ── createChatNotifier ─────────────────────────────────────────────────────

function harness(maxAutoTurns = 10) {
  const chatStore = new MemoryChatStore();
  const launched: { chatId: string; turn: number; messages: StoredMessage[] }[] = [];
  const notifier = createChatNotifier({
    chatStore,
    maxAutoTurns,
    startTurn: (chatId, turn, messages) => launched.push({ chatId, turn, messages }),
  });
  return { chatStore, launched, notifier };
}

async function makeChat(chatStore: MemoryChatStore, id = "c1") {
  await chatStore.createChat({ id });
  // Simulate one finished human turn.
  await chatStore.appendMessages(id, [{ role: "user", content: "hi" }]);
  await chatStore.setMeta(id, { status: "done", currentTurn: 0 });
  return id;
}

describe("createChatNotifier", () => {
  it("delivers immediately when no turn is live: appends the message and launches the next turn", async () => {
    const { chatStore, launched, notifier } = harness();
    const id = await makeChat(chatStore);

    await notifier.deliver(id, `${NOTIFICATION_PREFIX} run done`);

    assert.equal(launched.length, 1);
    assert.equal(launched[0].chatId, id);
    assert.equal(launched[0].turn, 1);
    // The launched model messages include the notification as a user message.
    const last = launched[0].messages[launched[0].messages.length - 1];
    assert.equal(last.role, "user");
    assert.ok(String(last.content).startsWith(NOTIFICATION_PREFIX));

    const meta = await chatStore.getMeta(id);
    assert.equal(meta?.status, "live");
    assert.equal(meta?.currentTurn, 1);
    assert.equal(meta?.autoTurns, 1);
  });

  it("queues while a turn is live and drains ALL queued notifications into one wake-up turn", async () => {
    const { chatStore, launched, notifier } = harness();
    const id = await makeChat(chatStore);

    notifier.turnStarted(id);
    await notifier.deliver(id, "n1");
    await notifier.deliver(id, "n2");
    assert.equal(launched.length, 0, "must not launch while live");

    await notifier.turnEnded(id);
    assert.equal(launched.length, 1, "one wake-up turn for both notifications");

    const messages = await chatStore.loadMessages(id);
    const notes = messages.filter((m) => m.content === "n1" || m.content === "n2");
    assert.equal(notes.length, 2);
  });

  it("parks at the auto-turn cap: message appended, no turn launched", async () => {
    const { chatStore, launched, notifier } = harness(2);
    const id = await makeChat(chatStore);
    await chatStore.setMeta(id, { autoTurns: 2 });

    await notifier.deliver(id, "late notification");

    assert.equal(launched.length, 0);
    const messages = await chatStore.loadMessages(id);
    assert.equal(messages[messages.length - 1].content, "late notification");
    // Turn counter untouched — the chat is parked, not advanced.
    const meta = await chatStore.getMeta(id);
    assert.equal(meta?.currentTurn, 0);
  });

  it("a human reset of autoTurns un-parks delivery", async () => {
    const { chatStore, launched, notifier } = harness(1);
    const id = await makeChat(chatStore);
    await chatStore.setMeta(id, { autoTurns: 1 });

    await notifier.deliver(id, "parked");
    assert.equal(launched.length, 0);

    // POST /chat resets autoTurns to 0 on a human message.
    await chatStore.setMeta(id, { autoTurns: 0 });
    await notifier.deliver(id, "unparked");
    assert.equal(launched.length, 1);
  });

  it("chained wake-ups increment autoTurns toward the cap", async () => {
    const { chatStore, launched, notifier } = harness(2);
    const id = await makeChat(chatStore);

    await notifier.deliver(id, "w1"); // autoTurns 0 → launches, now 1
    await notifier.turnEnded(id); // wake turn finished, nothing queued
    await notifier.deliver(id, "w2"); // autoTurns 1 → launches, now 2
    await notifier.turnEnded(id);
    await notifier.deliver(id, "w3"); // autoTurns 2 = cap → parks

    assert.equal(launched.length, 2);
    const meta = await chatStore.getMeta(id);
    assert.equal(meta?.autoTurns, 2);
  });

  it("delivering to an unknown chat is a silent no-op", async () => {
    const { launched, notifier } = harness();
    await notifier.deliver("nope", "hello?");
    assert.equal(launched.length, 0);
  });

  it("a deliver racing the launch's awaits queues instead of double-launching", async () => {
    const { chatStore, launched, notifier } = harness();
    const id = await makeChat(chatStore);

    // Fire two delivers back-to-back without awaiting the first: the first
    // claims liveness synchronously, so the second must queue.
    const p1 = notifier.deliver(id, "a");
    const p2 = notifier.deliver(id, "b");
    await Promise.all([p1, p2]);

    assert.equal(launched.length, 1);
    // The queued one arrives when the launched turn ends.
    await notifier.turnEnded(id);
    assert.equal(launched.length, 2);
  });
});
