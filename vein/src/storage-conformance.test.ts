import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdir, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { randomUUID } from "node:crypto";
import type { RunEvent, RunSummary } from "./core.js";
import { FileRunStore, MemoryRunStore, summarizeFromEvents, type RunStore } from "./store.js";
import { FileWorkspaceStore } from "./workspace.js";
import { pathlessWorkspace } from "./test-util/pathless-workspace.js";
import { workspaceConformance } from "./test-util/workspace-conformance.js";
import { FileChatStore, MemoryChatStore, type ChatStore, type ChatEvent } from "./chat-store.js";
import { FileSecretStore, MemorySecretStore, type SecretStore } from "./secret-store.js";

/**
 * The storage boundary's spec, as tests: one behavioral suite per layer,
 * parameterized over implementations. Every backend — file, memory, and a
 * future graph-backed one — passes the same cases. A backend that needs
 * a different assertion here is a backend that changed the contract.
 */

workspaceConformance({ name: "FileWorkspaceStore", make: (dir) => new FileWorkspaceStore(dir) });
workspaceConformance({ name: "path-less WorkspaceStore", make: (dir) => pathlessWorkspace(new FileWorkspaceStore(dir)) });

// ── Run store ──────────────────────────────────────────────────────────────

const runImpls: Array<{ name: string; make: (dir: string) => RunStore }> = [
  { name: "FileRunStore", make: (dir) => new FileRunStore(dir) },
  { name: "MemoryRunStore", make: () => new MemoryRunStore() },
];

const WF = "wf";
const ev = (runId: string, type: RunEvent["type"], extra: Partial<RunEvent> = {}): RunEvent => ({
  ts: new Date().toISOString(),
  runId,
  path: WF,
  type,
  ...extra,
});
const summaryFor = (runId: string): RunSummary => ({
  runId,
  workflow: WF,
  startedAt: "s",
  finishedAt: "f",
  durationMs: 1,
  status: "success",
  input: { q: 1 },
  output: { answer: 42 },
});

for (const impl of runImpls) {
  describe(`RunStore conformance: ${impl.name}`, () => {
    let dir: string;
    let store: RunStore;
    beforeEach(async () => {
      dir = join(tmpdir(), `vein-conf-run-${randomUUID()}`);
      await mkdir(dir, { recursive: true });
      store = impl.make(dir);
    });
    afterEach(() => rm(dir, { recursive: true, force: true }));

    it("append → getRunEvents → finalize → getRunSummary; unknown runs are empty/null", async () => {
      await store.append(WF, "1000", ev("1000", "run.start", { input: { q: 1 } }));
      await store.append(WF, "1000", ev("1000", "run.end"));
      assert.deepEqual((await store.getRunEvents(WF, "1000")).map((e) => e.type), ["run.start", "run.end"]);
      assert.equal(await store.getRunSummary(WF, "1000"), null);
      await store.finalize(WF, "1000", summaryFor("1000"));
      assert.deepEqual(await store.getRunSummary(WF, "1000"), summaryFor("1000"));
      assert.deepEqual(await store.getRunEvents(WF, "nope"), []);
      assert.equal(await store.getRunSummary(WF, "nope"), null);
    });

    it("listRuns is per-workflow, newest first; lastRunAt is the newest run's start", async () => {
      await store.append(WF, "1000", ev("1000", "run.start"));
      await store.append(WF, "3000", ev("3000", "run.start"));
      await store.append(WF, "2000", ev("2000", "run.start"));
      await store.append("other", "9000", ev("9000", "run.start"));
      assert.deepEqual(await store.listRuns(WF), ["3000", "2000", "1000"]);
      assert.deepEqual(await store.listRuns("never"), []);
      assert.equal(await store.lastRunAt(WF), 3000);
      assert.equal(await store.lastRunAt("never"), null);
    });

    it("a finalize-less log yields a partial summary (crash / in-flight)", async () => {
      await store.append(WF, "1", ev("1", "run.start", { input: { q: 1 } }));
      await store.append(WF, "1", ev("1", "step.end", { path: `${WF}/a`, output: "A" }));
      await store.append(WF, "1", ev("1", "step.error", { path: `${WF}/b`, error: { message: "boom" } }));
      const partial = summarizeFromEvents(WF, "1", await store.getRunEvents(WF, "1"));
      assert.equal(partial?.partial, true);
      assert.deepEqual(partial?.steps, { a: "A" });
      assert.equal(partial?.lastError?.message, "boom");
      assert.equal(summarizeFromEvents(WF, "nope", await store.getRunEvents(WF, "nope")), null);
    });

    it("tailEvents: history, then live follow, closing at the terminal event", async () => {
      await store.append(WF, "1", ev("1", "run.start"));
      const seen: string[] = [];
      const tail = (async () => {
        for await (const e of store.tailEvents(WF, "1", { intervalMs: 5 })) seen.push(e.type);
      })();
      await new Promise((r) => setTimeout(r, 25));
      await store.append(WF, "1", ev("1", "step.start", { path: `${WF}/a` }));
      await store.append(WF, "1", ev("1", "run.end"));
      await tail;
      assert.deepEqual(seen, ["run.start", "step.start", "run.end"]);
    });

    it("tailEvents: a run.resumed past a terminal event reopens the log", async () => {
      for (const t of ["run.start", "run.cancelled", "run.resumed", "run.end"] as const) {
        await store.append(WF, "1", ev("1", t));
      }
      const seen: string[] = [];
      for await (const e of store.tailEvents(WF, "1", { intervalMs: 5 })) seen.push(e.type);
      assert.deepEqual(seen, ["run.start", "run.cancelled", "run.resumed", "run.end"]);
    });

    it("tailEvents: stillLive keeps following after a terminal event; abort stops it", async () => {
      await store.append(WF, "1", ev("1", "run.start"));
      await store.append(WF, "1", ev("1", "run.error", { error: { message: "x" } }));
      let live = true;
      const seen: string[] = [];
      const tail = (async () => {
        for await (const e of store.tailEvents(WF, "1", { intervalMs: 5, stillLive: () => live })) {
          seen.push(e.type);
        }
      })();
      await new Promise((r) => setTimeout(r, 25));
      await store.append(WF, "1", ev("1", "run.resumed"));
      await store.append(WF, "1", ev("1", "run.end"));
      live = false;
      await tail;
      assert.deepEqual(seen, ["run.start", "run.error", "run.resumed", "run.end"]);

      const ac = new AbortController();
      const aborted: string[] = [];
      const t2 = (async () => {
        for await (const e of store.tailEvents(WF, "2", { intervalMs: 5, signal: ac.signal })) {
          aborted.push(e.type);
        }
      })();
      await new Promise((r) => setTimeout(r, 15));
      ac.abort();
      await t2;
      assert.deepEqual(aborted, [], "no events yet and aborted → nothing, and it returns");
    });
  });
}

// ── Chat store ─────────────────────────────────────────────────────────────

const chatImpls: Array<{ name: string; make: (dir: string) => ChatStore }> = [
  { name: "FileChatStore", make: (dir) => new FileChatStore(dir) },
  { name: "MemoryChatStore", make: () => new MemoryChatStore() },
];

const cev = (chatId: string, turn: number, type: ChatEvent["type"], extra: Partial<ChatEvent> = {}): ChatEvent => ({
  ts: new Date().toISOString(),
  chatId,
  turn,
  type,
  ...extra,
});

for (const impl of chatImpls) {
  describe(`ChatStore conformance: ${impl.name}`, () => {
    let dir: string;
    let store: ChatStore;
    beforeEach(async () => {
      dir = join(tmpdir(), `vein-conf-chat-${randomUUID()}`);
      await mkdir(dir, { recursive: true });
      store = impl.make(dir);
    });
    afterEach(() => rm(dir, { recursive: true, force: true }));

    it("create → meta → list → messages → delete", async () => {
      const meta = await store.createChat({ id: "c1", title: "t" });
      assert.equal(meta.id, "c1");
      assert.equal((await store.getMeta("c1"))?.title, "t");
      assert.equal(await store.getMeta("nope"), null);
      await store.setMeta("c1", { status: "done" });
      assert.equal((await store.getMeta("c1"))?.status, "done");
      assert.deepEqual((await store.listChats()).map((c) => c.id), ["c1"]);
      await store.appendMessages("c1", [{ role: "user", content: "hi" } as never]);
      assert.equal((await store.loadMessages("c1")).length, 1);
      await store.deleteChat("c1");
      assert.equal(await store.getMeta("c1"), null);
    });

    it("tailEvents yields one turn's events (history → live) and stops at its terminal", async () => {
      await store.createChat({ id: "c1" });
      await store.appendEvent("c1", cev("c1", 0, "text-delta", { delta: "a" }));
      await store.appendEvent("c1", cev("c1", 0, "chat.end"));
      await store.appendEvent("c1", cev("c1", 1, "text-delta", { delta: "b" }));
      const seen: ChatEvent[] = [];
      const tail = (async () => {
        for await (const e of store.tailEvents("c1", 1, { intervalMs: 5 })) seen.push(e);
      })();
      await new Promise((r) => setTimeout(r, 25));
      await store.appendEvent("c1", cev("c1", 1, "chat.end"));
      await tail;
      assert.deepEqual(seen.map((e) => [e.turn, e.type]), [[1, "text-delta"], [1, "chat.end"]]);
    });
  });
}

// ── Secret store ───────────────────────────────────────────────────────────

const secretImpls: Array<{ name: string; make: (dir: string) => SecretStore }> = [
  { name: "FileSecretStore", make: (dir) => new FileSecretStore(dir) },
  { name: "MemorySecretStore", make: () => new MemorySecretStore() },
];

for (const impl of secretImpls) {
  describe(`SecretStore conformance: ${impl.name}`, () => {
    let dir: string;
    let store: SecretStore;
    beforeEach(async () => {
      dir = join(tmpdir(), `vein-conf-secret-${randomUUID()}`);
      await mkdir(dir, { recursive: true });
      store = impl.make(dir);
    });
    afterEach(() => rm(dir, { recursive: true, force: true }));

    it("set → get → list (names only, never values) → overwrite → delete", async () => {
      assert.equal(await store.get("API_KEY"), undefined);
      await store.set("API_KEY", "s3cret");
      assert.equal(await store.get("API_KEY"), "s3cret");
      const listed = await store.list();
      assert.deepEqual(listed.map((s) => s.name), ["API_KEY"]);
      assert.equal(JSON.stringify(listed).includes("s3cret"), false, "list never carries values");
      await store.set("API_KEY", "rotated");
      assert.equal(await store.get("API_KEY"), "rotated");
      assert.equal(await store.delete("API_KEY"), true);
      assert.equal(await store.delete("API_KEY"), false);
      assert.equal(await store.get("API_KEY"), undefined);
    });
  });
}
