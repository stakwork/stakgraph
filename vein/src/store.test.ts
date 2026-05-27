import { describe, it, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { readFile, rm, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { randomUUID } from "node:crypto";
import type { RunEvent, RunSummary } from "./core.js";
import { FileRunStore, MemoryRunStore } from "./store.js";

const WF = "test-workflow";

// ── MemoryRunStore ─────────────────────────────────────────────────────────

describe("MemoryRunStore", () => {
  it("appends events", async () => {
    const store = new MemoryRunStore();
    const event: RunEvent = {
      ts: new Date().toISOString(),
      runId: "run-1",
      path: "test/step",
      type: "step.start",
    };
    await store.append(WF, "run-1", event);

    assert.equal(store.getEvents(WF, "run-1").length, 1);
    assert.deepEqual(store.getEvents(WF, "run-1")[0], event);
  });

  it("appends multiple events to same run", async () => {
    const store = new MemoryRunStore();
    const e1: RunEvent = {
      ts: new Date().toISOString(),
      runId: "run-1",
      path: "test/a",
      type: "step.start",
    };
    const e2: RunEvent = {
      ts: new Date().toISOString(),
      runId: "run-1",
      path: "test/a",
      type: "step.end",
    };
    await store.append(WF, "run-1", e1);
    await store.append(WF, "run-1", e2);

    assert.equal(store.getEvents(WF, "run-1").length, 2);
  });

  it("separates events by runId", async () => {
    const store = new MemoryRunStore();
    const e1: RunEvent = {
      ts: new Date().toISOString(),
      runId: "run-1",
      path: "test",
      type: "step.start",
    };
    const e2: RunEvent = {
      ts: new Date().toISOString(),
      runId: "run-2",
      path: "test",
      type: "step.start",
    };
    await store.append(WF, "run-1", e1);
    await store.append(WF, "run-2", e2);

    assert.equal(store.getEvents(WF, "run-1").length, 1);
    assert.equal(store.getEvents(WF, "run-2").length, 1);
  });

  it("stores summaries", async () => {
    const store = new MemoryRunStore();
    const summary: RunSummary = {
      runId: "run-1",
      workflow: "test",
      startedAt: new Date().toISOString(),
      finishedAt: new Date().toISOString(),
      durationMs: 100,
      status: "success",
      input: {},
      output: "done",
    };
    await store.finalize(WF, "run-1", summary);

    assert.deepEqual(store.getSummary(WF, "run-1"), summary);
  });
});

// ── FileRunStore ───────────────────────────────────────────────────────────

describe("FileRunStore", () => {
  let tempDir: string;

  beforeEach(async () => {
    tempDir = join(tmpdir(), `vein-test-${randomUUID()}`);
    await mkdir(tempDir, { recursive: true });
  });

  afterEach(async () => {
    await rm(tempDir, { recursive: true, force: true });
  });

  it("creates run directory and writes events.jsonl", async () => {
    const store = new FileRunStore(tempDir);
    const event: RunEvent = {
      ts: "2026-05-27T12:00:00.000Z",
      runId: "1748361600000",
      path: "test/step",
      type: "step.start",
      stepType: "http",
    };
    await store.append(WF, "1748361600000", event);

    const content = await readFile(
      join(tempDir, "workflows", WF, "runs", "1748361600000", "events.jsonl"),
      "utf-8",
    );
    const lines = content.trim().split("\n");
    assert.equal(lines.length, 1);
    assert.deepEqual(JSON.parse(lines[0]!), event);
  });

  it("appends multiple events to same file", async () => {
    const store = new FileRunStore(tempDir);
    const e1: RunEvent = {
      ts: "2026-05-27T12:00:00.000Z",
      runId: "100",
      path: "test/a",
      type: "step.start",
    };
    const e2: RunEvent = {
      ts: "2026-05-27T12:00:01.000Z",
      runId: "100",
      path: "test/a",
      type: "step.end",
      durationMs: 1000,
    };
    await store.append(WF, "100", e1);
    await store.append(WF, "100", e2);

    const content = await readFile(
      join(tempDir, "workflows", WF, "runs", "100", "events.jsonl"),
      "utf-8",
    );
    const lines = content.trim().split("\n");
    assert.equal(lines.length, 2);
    assert.equal(JSON.parse(lines[0]!).type, "step.start");
    assert.equal(JSON.parse(lines[1]!).type, "step.end");
  });

  it("writes run.json on finalize", async () => {
    const store = new FileRunStore(tempDir);
    const summary: RunSummary = {
      runId: "200",
      workflow: "deploy",
      startedAt: "2026-05-27T12:00:00.000Z",
      finishedAt: "2026-05-27T12:00:05.000Z",
      durationMs: 5000,
      status: "success",
      input: { service: "api" },
      output: "done",
    };
    await store.finalize("deploy", "200", summary);

    const content = await readFile(
      join(tempDir, "workflows", "deploy", "runs", "200", "run.json"),
      "utf-8",
    );
    assert.deepEqual(JSON.parse(content), summary);
  });

  it("handles separate run directories", async () => {
    const store = new FileRunStore(tempDir);
    await store.append(WF, "100", {
      ts: "2026-05-27T12:00:00.000Z",
      runId: "100",
      path: "test",
      type: "run.start",
    });
    await store.append(WF, "200", {
      ts: "2026-05-27T12:00:00.000Z",
      runId: "200",
      path: "test",
      type: "run.start",
    });

    const content1 = await readFile(
      join(tempDir, "workflows", WF, "runs", "100", "events.jsonl"),
      "utf-8",
    );
    const content2 = await readFile(
      join(tempDir, "workflows", WF, "runs", "200", "events.jsonl"),
      "utf-8",
    );

    assert.ok(content1.includes('"100"'));
    assert.ok(content2.includes('"200"'));
  });

  it("writes error summary", async () => {
    const store = new FileRunStore(tempDir);
    const summary: RunSummary = {
      runId: "300",
      workflow: "broken",
      startedAt: "2026-05-27T12:00:00.000Z",
      finishedAt: "2026-05-27T12:00:01.000Z",
      durationMs: 1000,
      status: "error",
      input: {},
      error: { message: "something failed", stack: "Error: something failed\n  at ..." },
    };
    await store.finalize("broken", "300", summary);

    const content = await readFile(
      join(tempDir, "workflows", "broken", "runs", "300", "run.json"),
      "utf-8",
    );
    const parsed = JSON.parse(content);
    assert.equal(parsed.status, "error");
    assert.equal(parsed.error.message, "something failed");
  });

  it("lists runs sorted newest first", async () => {
    const store = new FileRunStore(tempDir);
    await store.append(WF, "1000", { ts: "", runId: "1000", path: "", type: "run.start" });
    await store.append(WF, "3000", { ts: "", runId: "3000", path: "", type: "run.start" });
    await store.append(WF, "2000", { ts: "", runId: "2000", path: "", type: "run.start" });

    const ids = await store.listRuns(WF);
    assert.deepEqual(ids, ["3000", "2000", "1000"]);
  });

  it("getRunSummary returns summary", async () => {
    const store = new FileRunStore(tempDir);
    const summary: RunSummary = {
      runId: "500",
      workflow: WF,
      startedAt: "2026-05-27T12:00:00.000Z",
      finishedAt: "2026-05-27T12:00:01.000Z",
      durationMs: 1000,
      status: "success",
      input: {},
      output: "ok",
    };
    await store.finalize(WF, "500", summary);

    const result = await store.getRunSummary(WF, "500");
    assert.deepEqual(result, summary);
  });

  it("getRunSummary returns null for missing run", async () => {
    const store = new FileRunStore(tempDir);
    const result = await store.getRunSummary(WF, "missing");
    assert.equal(result, null);
  });

  it("getRunEvents returns events", async () => {
    const store = new FileRunStore(tempDir);
    const evt: RunEvent = { ts: "", runId: "600", path: "", type: "run.start" };
    await store.append(WF, "600", evt);

    const events = await store.getRunEvents(WF, "600");
    assert.equal(events.length, 1);
    assert.deepEqual(events[0], evt);
  });
});
