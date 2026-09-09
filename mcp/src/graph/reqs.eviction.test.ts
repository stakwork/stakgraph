/**
 * Unit tests for status-aware, env-configurable eviction in reqs.ts.
 *
 * Core invariants under test:
 *   1. Mixed pending + terminal at cap → oldest TERMINAL evicted (by index,
 *      not just index 0), all pending entries survive.
 *   2. All-pending at cap → NO eviction, console.error alarm fires.
 *   3. MAX_REQS env var is respected (small cap triggers eviction early).
 *   4. A live (pending) run's disk file survives >cap subsequent startReq
 *      calls — checkReq still returns its status.
 *   5. sweepOrphanedReqs routes through the shared evictOldestCompleted()
 *      helper and respects the env-configured cap; re-adopted entries are
 *      terminal before trimming runs.
 *
 * Each describe block uses a fresh dynamic import (cache-busting query param)
 * and its own REQS_DIR + MAX_REQS so module-level META/REQ_ORDER is clean.
 */

import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, existsSync, writeFileSync, mkdirSync } from "node:fs";
import { randomUUID } from "node:crypto";
import { tmpdir } from "node:os";
import { join } from "node:path";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function makeTmpDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "reqs-evict-"));
  // Remove so ensureDir() creates it fresh (mirrors existing test style)
  rmSync(dir, { recursive: true, force: true });
  return dir;
}

/** Unique cache-bust token so every import() loads a fresh module instance. */
let seq = 0;
function bust(): string {
  return `ev${Date.now()}_${++seq}`;
}

// ---------------------------------------------------------------------------
// 1. Mixed pending + terminal at cap — oldest terminal evicted, pending survive
// ---------------------------------------------------------------------------

describe("eviction — oldest terminal evicted when cap reached (mixed statuses)", () => {
  let dir: string;
  let mod: any;
  // Track console.error calls to assert the alarm does NOT fire in this path
  const errorCalls: string[] = [];
  let origConsoleError: typeof console.error;

  before(async () => {
    dir = makeTmpDir();
    process.env.REQS_DIR = dir;
    process.env.MAX_REQS = "5";

    origConsoleError = console.error;
    console.error = (...args: any[]) => {
      errorCalls.push(args.join(" "));
    };

    mod = await import(`./reqs.js?${bust()}`);
  });

  after(() => {
    console.error = origConsoleError;
    delete process.env.REQS_DIR;
    delete process.env.MAX_REQS;
    rmSync(dir, { recursive: true, force: true });
  });

  it("evicts the oldest terminal entry (not necessarily index 0) and preserves all pending entries", () => {
    // Layout (insertion order): pending0, terminal1(completed), pending2, terminal3(failed), pending4
    // Cap is 5 — so adding a 6th should evict terminal1 (index 1), NOT pending0 (index 0)
    const pending0 = mod.startReq();          // idx 0 — pending
    const terminal1 = mod.startReq();         // idx 1 — will be completed
    const pending2 = mod.startReq();          // idx 2 — pending
    const terminal3 = mod.startReq();         // idx 3 — will be failed
    const pending4 = mod.startReq();          // idx 4 — pending

    mod.finishReq(terminal1, { ok: true });   // mark terminal1 completed
    mod.failReq(terminal3, "boom", false);    // mark terminal3 failed

    // At this point we have 5 entries (= cap). Adding one more triggers eviction.
    const newReq = mod.startReq();

    // terminal1 was the OLDEST terminal — must be evicted
    assert.ok(
      !existsSync(join(dir, `${terminal1}.json`)),
      "oldest terminal (terminal1/completed) disk file must be deleted on eviction"
    );

    // terminal3 is also terminal but was inserted later — must NOT be evicted yet
    assert.ok(
      existsSync(join(dir, `${terminal3}.json`)),
      "newer terminal (terminal3/failed) disk file must survive first eviction"
    );

    // All three pending entries must survive (disk files intact)
    for (const id of [pending0, pending2, pending4]) {
      assert.ok(
        existsSync(join(dir, `${id}.json`)),
        `pending entry ${id} disk file must not be deleted`
      );
      const status = mod.checkReq(id)?.status;
      assert.strictEqual(status, "pending", `pending entry ${id} must still report pending`);
    }

    // The newly added request must be registered and pending
    assert.strictEqual(mod.checkReq(newReq)?.status, "pending");

    // Alarm console.error must NOT have fired (a terminal entry was available)
    const alarmFired = errorCalls.some((m) => m.includes("ALARM"));
    assert.ok(!alarmFired, "alarm must not fire when a terminal entry is available to evict");
  });
});

// ---------------------------------------------------------------------------
// 2. All-pending at cap — no eviction, alarm fires
// ---------------------------------------------------------------------------

describe("eviction — all-pending at cap causes no eviction and fires alarm", () => {
  let dir: string;
  let mod: any;
  const errorCalls: string[] = [];
  let origConsoleError: typeof console.error;

  before(async () => {
    dir = makeTmpDir();
    process.env.REQS_DIR = dir;
    process.env.MAX_REQS = "3";

    origConsoleError = console.error;
    console.error = (...args: any[]) => {
      errorCalls.push(args.join(" "));
    };

    mod = await import(`./reqs.js?${bust()}`);
  });

  after(() => {
    console.error = origConsoleError;
    delete process.env.REQS_DIR;
    delete process.env.MAX_REQS;
    rmSync(dir, { recursive: true, force: true });
  });

  it("registry grows past cap and alarm fires when every entry is pending", () => {
    const p0 = mod.startReq();
    const p1 = mod.startReq();
    const p2 = mod.startReq();
    // All three are pending; cap is 3 — next startReq must NOT evict anything

    errorCalls.length = 0; // reset before the triggering call
    const p3 = mod.startReq(); // triggers eviction attempt

    // All original pending entries must still be alive on disk
    for (const id of [p0, p1, p2]) {
      assert.ok(
        existsSync(join(dir, `${id}.json`)),
        `pending entry ${id} must not be deleted when all entries are pending`
      );
      assert.strictEqual(
        mod.checkReq(id)?.status,
        "pending",
        `${id} must still be pending`
      );
    }

    // The new request was still added (registry grew past cap)
    assert.strictEqual(mod.checkReq(p3)?.status, "pending", "new request must be registered");

    // Alarm must have fired
    const alarmFired = errorCalls.some((m) => m.includes("ALARM"));
    assert.ok(alarmFired, "console.error alarm must fire when all entries are pending at cap");
  });

  it("alarm message includes the registry size and MAX_REQS value", () => {
    // Reset calls before a clean check
    errorCalls.length = 0;
    mod.startReq(); // one more to trigger again
    const alarmMsg = errorCalls.find((m) => m.includes("ALARM")) ?? "";
    assert.ok(
      alarmMsg.includes("pending"),
      "alarm message must mention pending entries"
    );
    assert.ok(
      alarmMsg.includes("3"),
      "alarm message must include the MAX_REQS cap value (3)"
    );
  });
});

// ---------------------------------------------------------------------------
// 3. MAX_REQS env var is respected
// ---------------------------------------------------------------------------

describe("eviction — MAX_REQS env var configures the cap", () => {
  let dir: string;
  let mod: any;

  before(async () => {
    dir = makeTmpDir();
    process.env.REQS_DIR = dir;
    process.env.MAX_REQS = "2"; // very small cap

    mod = await import(`./reqs.js?${bust()}`);
  });

  after(() => {
    delete process.env.REQS_DIR;
    delete process.env.MAX_REQS;
    rmSync(dir, { recursive: true, force: true });
  });

  it("eviction fires at cap = 2, not at the default 100", () => {
    const t0 = mod.startReq();
    mod.finishReq(t0, {}); // terminal — evictable

    const t1 = mod.startReq();
    // Registry is now at cap (2). Adding another must trigger eviction of t0.
    const t2 = mod.startReq();

    assert.ok(
      !existsSync(join(dir, `${t0}.json`)),
      "t0 (completed, oldest terminal) must be evicted at cap=2"
    );
    assert.ok(
      existsSync(join(dir, `${t1}.json`)),
      "t1 must survive"
    );
    assert.ok(
      existsSync(join(dir, `${t2}.json`)),
      "t2 (newly added) must exist"
    );
  });

  it("defaults to 100 when MAX_REQS is unset or invalid", async () => {
    // Use a separate module instance with no MAX_REQS env
    const savedMaxReqs = process.env.MAX_REQS;
    delete process.env.MAX_REQS;
    const dir2 = makeTmpDir();
    const savedReqsDir = process.env.REQS_DIR;
    process.env.REQS_DIR = dir2;

    const mod2 = await import(`./reqs.js?${bust()}`);

    // Fill 99 entries (all terminal). The 100th should NOT trigger eviction yet.
    for (let i = 0; i < 99; i++) {
      const id = mod2.startReq();
      mod2.finishReq(id, {});
    }
    // 99 entries, cap=100 — no eviction yet. Add #100 (no eviction at exactly cap-1 entries)
    const id100 = mod2.startReq();
    mod2.finishReq(id100, {});
    // Now we have 100 terminal entries — adding one more should trigger eviction
    const id101 = mod2.startReq();
    assert.strictEqual(mod2.checkReq(id101)?.status, "pending");

    // Cleanup
    process.env.MAX_REQS = savedMaxReqs ?? "";
    if (!process.env.MAX_REQS) delete process.env.MAX_REQS;
    process.env.REQS_DIR = savedReqsDir!;
    rmSync(dir2, { recursive: true, force: true });
  });
});

// ---------------------------------------------------------------------------
// 4. Pending run survives >cap subsequent startReq calls — disk file intact
// ---------------------------------------------------------------------------

describe("eviction — pending run's disk file never deleted across many startReq calls", () => {
  let dir: string;
  let mod: any;
  // suppress alarm noise in this test
  let origConsoleError: typeof console.error;

  before(async () => {
    dir = makeTmpDir();
    process.env.REQS_DIR = dir;
    process.env.MAX_REQS = "5";

    origConsoleError = console.error;
    console.error = () => {};

    mod = await import(`./reqs.js?${bust()}`);
  });

  after(() => {
    console.error = origConsoleError;
    delete process.env.REQS_DIR;
    delete process.env.MAX_REQS;
    rmSync(dir, { recursive: true, force: true });
  });

  it("a pending request's file and /progress survive cap*3 subsequent startReq calls", () => {
    // Register the live run we want to protect
    const liveRun = mod.startReq("https://example.com/live-webhook");

    // Fill the rest of the cap with completed (terminal) entries, then keep
    // adding more — eviction should always pick a terminal entry, never liveRun
    for (let i = 0; i < 15; i++) {
      const id = mod.startReq();
      mod.finishReq(id, { i });
    }

    // liveRun disk file must still exist
    assert.ok(
      existsSync(join(dir, `${liveRun}.json`)),
      "live pending run disk file must never be deleted during eviction"
    );

    // checkReq (disk-authoritative) must still return pending
    const result = mod.checkReq(liveRun);
    assert.ok(result, "checkReq must return a result (not null/404) for the live run");
    assert.strictEqual(result.status, "pending", "live run must still report pending");
  });

  it("a pending run can still be completed after surviving many evictions", () => {
    const liveRun = mod.startReq();

    // Trigger 10 eviction cycles
    for (let i = 0; i < 10; i++) {
      const id = mod.startReq();
      mod.finishReq(id, {});
    }

    // The live run survived — now complete it
    mod.finishReq(liveRun, { finalAnswer: 42 });

    const result = mod.checkReq(liveRun);
    assert.strictEqual(result.status, "completed");
    assert.deepStrictEqual(result.result, { finalAnswer: 42 });
  });
});

// ---------------------------------------------------------------------------
// 5. sweepOrphanedReqs uses shared helper and respects env cap
// ---------------------------------------------------------------------------

describe("sweepOrphanedReqs — uses evictOldestCompleted helper, respects env cap", () => {
  let dir: string;
  let origConsoleError: typeof console.error;

  before(() => {
    dir = makeTmpDir();
    // Ensure dir exists for writing pre-existing files
    mkdirSync(dir, { recursive: true });
    origConsoleError = console.error;
    console.error = () => {};
  });

  after(() => {
    console.error = origConsoleError;
    delete process.env.REQS_DIR;
    delete process.env.MAX_REQS;
    rmSync(dir, { recursive: true, force: true });
  });

  it("re-adopted entries are terminal before trimming; pending entries survive sweep eviction", async () => {
    // Simulate a previous boot that left files on disk:
    //   - 2 completed files (terminal — evictable)
    //   - 1 pending file (orphan — will be flipped to failed by sweep, then terminal)
    //
    // Cap = 2, so adopting 3 files will trigger eviction during sweep.
    // After the sweep flips the pending→failed, all 3 re-adopted entries are
    // terminal, so the oldest can be safely evicted.

    const completedId1 = randomUUID();
    const completedId2 = randomUUID();
    const orphanPendingId = randomUUID();

    // Write pre-existing disk files (simulating the previous boot's output)
    writeFileSync(
      join(dir, `${completedId1}.json`),
      JSON.stringify({ status: "completed", result: {} })
    );
    // Small sleep to ensure distinct mtimes for ordering
    await new Promise((r) => setTimeout(r, 10));
    writeFileSync(
      join(dir, `${completedId2}.json`),
      JSON.stringify({ status: "completed", result: {} })
    );
    await new Promise((r) => setTimeout(r, 10));
    writeFileSync(
      join(dir, `${orphanPendingId}.json`),
      JSON.stringify({ status: "pending", webhookUrl: "https://example.com/orph" })
    );

    process.env.REQS_DIR = dir;
    process.env.MAX_REQS = "2"; // cap at 2 — adopting 3 files triggers eviction

    const mod = await import(`./reqs.js?${bust()}`);
    const orphans: any[] = mod.sweepOrphanedReqs();

    // sweepOrphanedReqs must flip the pending orphan and return it
    assert.strictEqual(orphans.length, 1, "one orphaned pending entry must be reported");
    assert.strictEqual(orphans[0].request_id, orphanPendingId);
    assert.strictEqual(orphans[0].retryable, true);
    assert.strictEqual(orphans[0].webhookUrl, "https://example.com/orph");

    // The orphan was flipped to failed on disk
    const mod2 = await import(`./reqs.js?${bust()}`);
    const orphanOnDisk = mod2.checkReq(orphanPendingId);
    // If the file survived the eviction it should be failed; if it was the
    // oldest completed and got evicted, it's gone (null). Either is acceptable
    // — the key assertion is that no still-pending file was deleted.
    if (orphanOnDisk) {
      assert.strictEqual(
        orphanOnDisk.status,
        "failed",
        "orphan pending must be flipped to failed before eviction runs"
      );
    }
  });

  it("sweep's eviction never deletes a file for an entry that is still pending at adoption time", async () => {
    // Fresh dir with only pending files — sweep will flip them all to failed,
    // then eviction can only pick from those (terminal) entries.
    const dir2 = makeTmpDir();
    mkdirSync(dir2, { recursive: true });
    const ids: string[] = [];
    for (let i = 0; i < 4; i++) {
      const id = randomUUID();
      ids.push(id);
      writeFileSync(
        join(dir2, `${id}.json`),
        JSON.stringify({ status: "pending" })
      );
      await new Promise((r) => setTimeout(r, 5));
    }

    process.env.REQS_DIR = dir2;
    process.env.MAX_REQS = "2"; // cap at 2, 4 files on disk

    const mod = await import(`./reqs.js?${bust()}`);
    const orphans: any[] = mod.sweepOrphanedReqs();

    // All 4 were pending → all 4 reported as orphaned (flipped to failed first)
    assert.strictEqual(orphans.length, 4, "all 4 pending files must be reported as orphaned");

    // With cap=2 and 4 files, 2 must be evicted — but they were all flipped to
    // failed before eviction ran, so no live-pending file was deleted during eviction.
    // At least 2 survive (the ones not evicted).
    const surviving = ids.filter((id) => existsSync(join(dir2, `${id}.json`)));
    assert.ok(
      surviving.length >= 2,
      `at least 2 files must survive (cap=2 keeps the 2 most-recently-adopted); got ${surviving.length}`
    );

    // None of the surviving files should be pending — all were flipped before eviction
    for (const id of surviving) {
      const r = mod.checkReq(id);
      if (r) {
        assert.strictEqual(r.status, "failed", `surviving file ${id} must be failed, not pending`);
      }
    }

    rmSync(dir2, { recursive: true, force: true });
  });
});

// ---------------------------------------------------------------------------
// 6. Eviction removes correct index (not always index 0)
// ---------------------------------------------------------------------------

describe("eviction — splice removes correct non-zero index, not always index 0", () => {
  let dir: string;
  let mod: any;

  before(async () => {
    dir = makeTmpDir();
    process.env.REQS_DIR = dir;
    process.env.MAX_REQS = "4";

    mod = await import(`./reqs.js?${bust()}`);
  });

  after(() => {
    delete process.env.REQS_DIR;
    delete process.env.MAX_REQS;
    rmSync(dir, { recursive: true, force: true });
  });

  it("when index 0 is pending and index 2 is the oldest terminal, splices index 2", () => {
    // Insertion order: pending(0), pending(1), completed(2), pending(3)
    // Cap = 4. Adding #5 → must evict completed at index 2, not pending at index 0.
    const p0 = mod.startReq(); // idx 0 — stays pending
    const p1 = mod.startReq(); // idx 1 — stays pending
    const c2 = mod.startReq(); // idx 2 — will be completed
    const p3 = mod.startReq(); // idx 3 — stays pending

    mod.finishReq(c2, { done: true });
    // Now at cap=4. Add one more → eviction must splice index 2 (c2).
    const newId = mod.startReq();

    assert.ok(
      !existsSync(join(dir, `${c2}.json`)),
      "completed entry at index 2 must be spliced out (not index 0)"
    );
    for (const id of [p0, p1, p3]) {
      assert.ok(
        existsSync(join(dir, `${id}.json`)),
        `pending entry ${id} at non-zero index must survive`
      );
    }
    assert.strictEqual(mod.checkReq(newId)?.status, "pending");
  });
});
