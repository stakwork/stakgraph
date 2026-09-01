import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdir, rm, stat } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { randomUUID } from "node:crypto";
import type { RunEvent, RunSummary } from "./core.js";
import { FileRunStore, MemoryRunStore, summarizeFromEvents, type RunStore } from "./store.js";
import { FileWorkspaceStore, type WorkspaceStore } from "./workspace.js";
import { pathlessWorkspace } from "./test-util/pathless-workspace.js";
import { FileChatStore, MemoryChatStore, type ChatStore, type ChatEvent } from "./chat-store.js";
import { FileSecretStore, MemorySecretStore, type SecretStore } from "./secret-store.js";

/**
 * The storage boundary's spec, as tests: one behavioral suite per layer,
 * parameterized over implementations. Every backend — file, memory, and a
 * future graph-backed one — passes the same cases. A backend that needs
 * a different assertion here is a backend that changed the contract.
 */

const STEP_SRC = (type: string, desc: string) => `import { z } from "zod";
import { defineStep } from "vein";
export default defineStep({
  type: ${JSON.stringify(type)},
  description: ${JSON.stringify(desc)},
  input: z.object({}),
  output: z.any(),
  async run() { return ${JSON.stringify(desc)}; },
});
`;

// ── Workspace store ────────────────────────────────────────────────────────

const workspaceImpls: Array<{ name: string; make: (dir: string) => WorkspaceStore }> = [
  { name: "FileWorkspaceStore", make: (dir) => new FileWorkspaceStore(dir) },
  { name: "path-less WorkspaceStore", make: (dir) => pathlessWorkspace(new FileWorkspaceStore(dir)) },
];

for (const impl of workspaceImpls) {
  describe(`WorkspaceStore conformance: ${impl.name}`, () => {
    let dir: string;
    let ws: WorkspaceStore;
    beforeEach(async () => {
      dir = join(tmpdir(), `vein-conf-ws-${randomUUID()}`);
      await mkdir(dir, { recursive: true });
      ws = impl.make(dir);
    });
    afterEach(() => rm(dir, { recursive: true, force: true }));

    const steps = [{ id: "a", type: "log", config: { message: "hi" } }];

    it("workflow publish → list → metadata → source → hash round-trip", async () => {
      await ws.publishWorkflow("wf", "v1", { steps }, "first", "exp", "me");
      const list = await ws.listWorkflows();
      assert.deepEqual(
        list.map((w) => [w.name, w.activeVersion, w.versions, w.description, w.category, w.publisher]),
        [["wf", "v1", ["v1"], "first", "exp", "me"]],
      );
      assert.equal("lastRunAt" in list[0]!, false, "runs are the run store's — never listed here");
      const meta = await ws.getWorkflowMetadata("wf");
      assert.equal(meta?.active, "v1");
      assert.equal(meta?.publisher, "me");
      assert.equal(await ws.getWorkflowMetadata("nope"), null);
      const src = await ws.getWorkflowSource("wf", "v1");
      assert.ok(src.includes("type: log"));
      assert.equal(typeof (await ws.getWorkflowHash("wf")), "string");
      assert.equal(await ws.getWorkflowHash("nope"), null);
      assert.equal((await ws.getWorkflow("wf")).steps.length, 1);
      assert.equal((await ws.getWorkflowVersion("wf", "v1")).steps.length, 1);
      await assert.rejects(() => ws.getWorkflow("nope"), /not found/);
    });

    it("versions, active switching, content dedup, category, params", async () => {
      await ws.publishWorkflow("wf", "v1", { steps, params: { greeting: "old" } });
      const first = await ws.publishWorkflowByContent("wf", await ws.getWorkflowSource("wf", "v1"));
      assert.equal(first.changed, false, "same content → no new version");
      assert.equal(first.version, "v1");
      const second = await ws.publishWorkflowByContent(
        "wf",
        (await ws.getWorkflowSource("wf", "v1")).replace("old", "new"),
      );
      assert.equal(second.changed, true);
      assert.notEqual(second.version, "v1");
      assert.equal((await ws.getWorkflowMetadata("wf"))?.active, second.version);
      await ws.setActiveVersion("wf", "v1");
      assert.equal((await ws.getWorkflowMetadata("wf"))?.active, "v1");
      await assert.rejects(() => ws.setActiveVersion("wf", "v99"));
      await ws.setWorkflowCategory("wf", "cat");
      assert.equal((await ws.getWorkflowMetadata("wf"))?.category, "cat");
      const p = await ws.setParam("wf", "greeting", "newer");
      assert.deepEqual([p.before, p.after], ["old", "newer"]);
      assert.equal((await ws.getWorkflow("wf")).params?.["greeting"], "newer");
    });

    it("createWorkflow allocates a fresh name/version and returns it", async () => {
      const a = await ws.createWorkflow("made", { steps });
      const b = await ws.createWorkflow("made", { steps });
      assert.equal(a.name, "made");
      assert.notEqual(b.name, a.name, "a second create under the same name is renamed, not clobbered");
    });

    it("step publish → list → versions → source → active switching → delete", async () => {
      const v1 = await ws.publishStep("my-step", STEP_SRC("my-step", "one"), "one", "svc");
      const again = await ws.publishStep("my-step", STEP_SRC("my-step", "one"), "one", "svc");
      assert.equal(again.changed, false, "same source → no new version");
      const v2 = await ws.publishStep("my-step", STEP_SRC("my-step", "two"), "two", "svc");
      assert.equal(v2.changed, true);
      assert.deepEqual(
        (await ws.listSteps()).map((s) => [s.type, s.description, s.publisher]),
        [["my-step", "two", "svc"]],
      );
      assert.deepEqual(await ws.listSteps({ publisher: "other" }), []);
      const versions = await ws.listStepVersions("my-step");
      assert.equal(versions.active, v2.version);
      assert.deepEqual(new Set(versions.versions), new Set([v1.version, v2.version]));
      assert.ok((await ws.getStepVersionSource("my-step", v1.version)).includes('"one"'));
      await ws.setActiveStepVersion("my-step", v1.version);
      assert.equal((await ws.listStepVersions("my-step")).active, v1.version);
      assert.equal((await ws.getStepSource("my-step"))?.code.includes('"one"'), true);
      assert.equal(await ws.deleteStep("my-step"), true);
      assert.equal(await ws.deleteStep("my-step"), false);
      assert.deepEqual(await ws.listSteps(), []);
    });

    it("deleteStepsByPublisher removes exactly that publisher's steps", async () => {
      await ws.publishStep("a", STEP_SRC("a", "a"), "a", "svc-1");
      await ws.publishStep("ns/b", STEP_SRC("ns/b", "b"), "b", "svc-1");
      await ws.publishStep("c", STEP_SRC("c", "c"), "c", "svc-2");
      assert.deepEqual((await ws.deleteStepsByPublisher("svc-1")).sort(), ["a", "ns/b"]);
      assert.deepEqual((await ws.listSteps()).map((s) => s.type), ["c"]);
    });

    it("getStepSource spans tiers: custom from the store, lib + core from the engine, null otherwise", async () => {
      await ws.publishStep("ns/custom", STEP_SRC("ns/custom", "x"));
      assert.equal((await ws.getStepSource("ns/custom"))?.origin, "custom");
      assert.equal((await ws.getStepSource("log"))?.origin, "core");
      assert.equal((await ws.getStepSource("github/fetch-pr"))?.origin, "lib");
      assert.equal(await ws.getStepSource("no/such/step"), null);
    });

    it("materializeCustomSteps returns a directory holding every active custom step as a file", async () => {
      await ws.publishStep("flat", STEP_SRC("flat", "f"));
      await ws.publishStep("ns/nested", STEP_SRC("ns/nested", "n"));
      const root = await ws.materializeCustomSteps();
      assert.ok((await stat(join(root, "flat.ts"))).isFile());
      assert.ok((await stat(join(root, "ns", "nested.ts"))).isFile());
      assert.equal(await ws.materializeCustomSteps(), root, "idempotent");
    });
  });
}

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
