/**
 * Unit tests for graceful-shutdown helpers in reqs.ts:
 *
 *  1. failPendingReqs — flips only "pending" records, reads webhookUrl from
 *     disk, returns correct OrphanedReq descriptors.
 *  2. Cross-boot idempotency — a record flipped by failPendingReqs (now
 *     "failed" on disk) is NOT re-fired by the next boot's sweepOrphanedReqs.
 *
 * Each describe block uses a fresh dynamic import (cache-busting query param)
 * so the module-level META/REQ_ORDER state is clean and REQS_DIR is re-read.
 */
import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function makeTmpDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "reqs-shutdown-"));
  return dir;
}

// ---------------------------------------------------------------------------
// 1. failPendingReqs — flips only pending, reads webhookUrl from disk
// ---------------------------------------------------------------------------

describe("failPendingReqs — only pending records are flipped", () => {
  let dir: string;
  let mod: any;

  before(async () => {
    dir = makeTmpDir();
    // Remove pre-created tmp dir so ensureDir has to create it (proves it
    // resolves the absolute path correctly, same as the existing ensureDir tests).
    rmSync(dir, { recursive: true, force: true });
    process.env.REQS_DIR = dir;
    mod = await import(`./reqs.js?shutdown1=${Date.now()}`);
  });

  after(() => {
    delete process.env.REQS_DIR;
    rmSync(dir, { recursive: true, force: true });
  });

  it("returns an empty array when there are no pending requests", () => {
    const orphans = mod.failPendingReqs("shutdown");
    assert.deepStrictEqual(orphans, []);
  });

  it("flips a pending request and includes its webhookUrl", () => {
    const id = mod.startReq("https://example.com/hook");
    const orphans: any[] = mod.failPendingReqs("test shutdown");

    assert.strictEqual(orphans.length, 1);
    assert.strictEqual(orphans[0].request_id, id);
    assert.strictEqual(orphans[0].error, "test shutdown");
    assert.strictEqual(orphans[0].retryable, true);
    assert.strictEqual(orphans[0].webhookUrl, "https://example.com/hook");
  });

  it("writes status:failed to disk for the flipped request", () => {
    const id = mod.startReq("https://example.com/hook2");
    mod.failPendingReqs("test shutdown disk");

    const raw = JSON.parse(readFileSync(join(dir, `${id}.json`), "utf-8"));
    assert.strictEqual(raw.status, "failed");
    assert.strictEqual(raw.retryable, true);
  });

  it("omits webhookUrl from the descriptor when none was registered", () => {
    const id = mod.startReq(); // no webhookUrl
    const orphans: any[] = mod.failPendingReqs("no-webhook shutdown");

    const mine = orphans.find((o: any) => o.request_id === id);
    assert.ok(mine, "should have an entry for the request");
    assert.strictEqual(mine.webhookUrl, undefined);
  });

  it("does not flip already-completed requests", () => {
    const id = mod.startReq("https://example.com/done");
    mod.finishReq(id, { ok: true });

    // Only pending entries should appear; completed one must not be re-flipped.
    const before = mod.checkReq(id);
    assert.strictEqual(before.status, "completed");

    mod.failPendingReqs("shutdown should skip completed");

    const after = mod.checkReq(id);
    assert.strictEqual(after.status, "completed", "completed request must not be overwritten");
  });

  it("does not flip already-failed requests", () => {
    const id = mod.startReq("https://example.com/already-failed");
    mod.failReq(id, "pre-existing error", false);

    const beforeStatus = mod.checkReq(id).status;
    assert.strictEqual(beforeStatus, "failed");

    mod.failPendingReqs("shutdown should skip failed");

    // Record must still be failed with the original error, not overwritten
    const after = mod.checkReq(id);
    assert.strictEqual(after.status, "failed");
  });
});

// ---------------------------------------------------------------------------
// 2. failPendingReqs with multiple mixed-status requests
// ---------------------------------------------------------------------------

describe("failPendingReqs — mixed status requests", () => {
  let dir: string;
  let mod: any;

  before(async () => {
    dir = makeTmpDir();
    rmSync(dir, { recursive: true, force: true });
    process.env.REQS_DIR = dir;
    mod = await import(`./reqs.js?shutdown2=${Date.now()}`);
  });

  after(() => {
    delete process.env.REQS_DIR;
    rmSync(dir, { recursive: true, force: true });
  });

  it("flips only the pending ones and returns exactly those descriptors", () => {
    const pendingA = mod.startReq("https://example.com/a");
    const pendingB = mod.startReq("https://example.com/b");
    const completedId = mod.startReq("https://example.com/c");
    const failedId = mod.startReq("https://example.com/d");

    mod.finishReq(completedId, { done: true });
    mod.failReq(failedId, "pre-fail", false);

    const orphans: any[] = mod.failPendingReqs(mod.SHUTDOWN_ORPHAN_ERROR);

    // Only the two pending ones come back
    assert.strictEqual(orphans.length, 2);
    const ids = orphans.map((o: any) => o.request_id);
    assert.ok(ids.includes(pendingA), "pendingA should be in result");
    assert.ok(ids.includes(pendingB), "pendingB should be in result");

    // Completed and failed untouched
    assert.strictEqual(mod.checkReq(completedId).status, "completed");
    assert.strictEqual(mod.checkReq(failedId).status, "failed");
  });
});

// ---------------------------------------------------------------------------
// 3. Cross-boot idempotency — shutdown-flipped record skipped by sweep
// ---------------------------------------------------------------------------

describe("cross-boot idempotency — shutdown-flipped record not re-fired by sweepOrphanedReqs", () => {
  let dir: string;
  // Two separate module instances to simulate a process restart
  let bootA: any;
  let bootB: any;

  before(async () => {
    dir = makeTmpDir();
    rmSync(dir, { recursive: true, force: true });
    process.env.REQS_DIR = dir;
    // "Boot A" — the process that shuts down gracefully
    bootA = await import(`./reqs.js?bootA=${Date.now()}`);
  });

  after(() => {
    delete process.env.REQS_DIR;
    rmSync(dir, { recursive: true, force: true });
  });

  it("sweep on next boot returns empty (no re-fire) for a shutdown-flipped record", async () => {
    // Boot A: create a pending request then gracefully shut down
    bootA.startReq("https://example.com/webhook");
    const orphansFromShutdown = bootA.failPendingReqs(bootA.SHUTDOWN_ORPHAN_ERROR);
    assert.strictEqual(orphansFromShutdown.length, 1, "shutdown should have flipped 1 record");

    // Disk file is now status:failed — verify before simulating restart
    const id = orphansFromShutdown[0].request_id;
    const diskRecord = JSON.parse(readFileSync(join(dir, `${id}.json`), "utf-8"));
    assert.strictEqual(diskRecord.status, "failed");

    // "Boot B" — fresh module instance simulating a process restart
    process.env.REQS_DIR = dir; // same dir
    bootB = await import(`./reqs.js?bootB=${Date.now()}`);

    // The sweep on next boot must NOT flip this already-failed record again
    const orphansFromSweep = bootB.sweepOrphanedReqs();
    assert.strictEqual(
      orphansFromSweep.length,
      0,
      "sweepOrphanedReqs should skip the already-failed shutdown record",
    );

    // Disk record still reflects the shutdown failure, not a new restart error
    const after = JSON.parse(readFileSync(join(dir, `${id}.json`), "utf-8"));
    assert.strictEqual(after.status, "failed");
    assert.strictEqual(after.retryable, true);
  });
});
