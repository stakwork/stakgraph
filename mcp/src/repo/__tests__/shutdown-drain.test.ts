/**
 * Unit tests for graceful-shutdown behaviour in repo/index.ts:
 *
 *  1. drainForShutdown — delivers webhook exactly once with the correct
 *     failed/retryable:true payload (single attempt, no retry ladder).
 *  2. drainForShutdown — honours the 3 s AbortSignal timeout on a hanging
 *     endpoint (does not wait 15 s / retry three times).
 *  3. Double-signal re-entrancy — a second drainForShutdown call after the
 *     first has already flipped records delivers zero additional webhooks.
 *  4. setShuttingDown is exported; after drain the disk record is
 *     authoritative and exactly one webhook was delivered.
 *
 * KEY IMPORT RULE:
 *   repo/index.ts has a static `import * as asyncReqs from "../graph/reqs.js"`
 *   (no query param).  Cache-busting repo/index with a query param gives a
 *   fresh shuttingDown=false instance, but its internal asyncReqs still
 *   resolves to the canonical graph/reqs.js module (no query param).
 *   Therefore tests must import graph/reqs.js WITHOUT a cache-busting param
 *   so they share META / REQ_ORDER with the instance repo/index uses.
 *   Only repo/index is cache-busted (to reset shuttingDown between suites).
 *
 *   Each suite sets REQS_DIR to a fresh temp dir before the reqs import runs
 *   (REQS_DIR is read at module load time for the first suite; subsequent
 *   suites set it before calling startReq so the disk path is correct even
 *   though the reqs module is already loaded).
 *
 * Runs with NO_DB=true — Neo4j is never contacted.
 */
import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import http from "node:http";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function makeTmpDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "reqs-drain-"));
  rmSync(dir, { recursive: true, force: true }); // let ensureDir recreate it
  return dir;
}

/** Real HTTP server that captures POST bodies. */
function makeWebhookServer(): {
  url: string;
  calls: Array<{ body: any }>;
  close: () => Promise<void>;
} {
  const calls: Array<{ body: any }> = [];
  const server = http.createServer((req, res) => {
    let raw = "";
    req.on("data", (c) => (raw += c));
    req.on("end", () => {
      try { calls.push({ body: JSON.parse(raw) }); }
      catch { calls.push({ body: raw }); }
      res.writeHead(200);
      res.end("ok");
    });
  });
  server.listen(0);
  const addr = server.address() as { port: number };
  return {
    url: `http://127.0.0.1:${addr.port}/hook`,
    calls,
    close: () => new Promise<void>((r) => server.close(() => r())),
  };
}

/** Server that never responds — tests the 3 s timeout. */
function makeHangingServer(): {
  url: string;
  close: () => Promise<void>;
} {
  const server = http.createServer(() => { /* intentionally silent */ });
  server.listen(0);
  const addr = server.address() as { port: number };
  return {
    url: `http://127.0.0.1:${addr.port}/hook`,
    close: () => new Promise<void>((r) => server.close(() => r())),
  };
}

// ---------------------------------------------------------------------------
// Shared canonical reqs module (no cache-bust — same instance repo/index uses)
// ---------------------------------------------------------------------------
let sharedReqs: any;

// ---------------------------------------------------------------------------
// 1. drainForShutdown — single-attempt webhook, correct payload
// ---------------------------------------------------------------------------

describe("drainForShutdown — single-attempt webhook (no retry ladder)", () => {
  let dir: string;
  let hook: ReturnType<typeof makeWebhookServer>;
  let rr: any;

  before(async () => {
    dir = makeTmpDir();
    process.env.REQS_DIR = dir;
    process.env.NO_DB = "true";
    // Import canonical reqs (no cache-bust) — same instance repo/index uses
    sharedReqs = await import("../../graph/reqs.js");
    // Cache-bust repo/index to get fresh shuttingDown = false
    rr = await import(`../index.js?d1=${Date.now()}`);
    hook = makeWebhookServer();
  });

  after(async () => {
    delete process.env.REQS_DIR;
    delete process.env.NO_DB;
    rmSync(dir, { recursive: true, force: true });
    await hook.close();
  });

  it("delivers webhook exactly once with failed/retryable:true payload", async () => {
    const id = sharedReqs.startReq(hook.url);

    await rr.drainForShutdown();

    assert.strictEqual(hook.calls.length, 1, "should deliver webhook exactly once");
    assert.strictEqual(hook.calls[0].body.request_id, id);
    assert.strictEqual(hook.calls[0].body.status, "failed");
    assert.strictEqual(hook.calls[0].body.retryable, true);
    assert.ok(
      typeof hook.calls[0].body.error === "string",
      "error field should be a string",
    );

    const disk = JSON.parse(readFileSync(join(dir, `${id}.json`), "utf-8"));
    assert.strictEqual(disk.status, "failed");
    assert.strictEqual(disk.retryable, true);
  });
});

// ---------------------------------------------------------------------------
// 2. drainForShutdown — single-attempt 3 s timeout (does not hang)
// ---------------------------------------------------------------------------

describe("drainForShutdown — 3 s timeout on hanging endpoint", () => {
  let dir: string;
  let hanging: ReturnType<typeof makeHangingServer>;
  let rr: any;

  before(async () => {
    dir = makeTmpDir();
    process.env.REQS_DIR = dir;
    process.env.NO_DB = "true";
    sharedReqs = await import("../../graph/reqs.js");
    rr = await import(`../index.js?d2=${Date.now()}`);
    hanging = makeHangingServer();
  });

  after(async () => {
    delete process.env.REQS_DIR;
    delete process.env.NO_DB;
    rmSync(dir, { recursive: true, force: true });
    await hanging.close();
  });

  it("completes within 5 s even when the webhook endpoint never responds", async () => {
    sharedReqs.startReq(hanging.url);

    const start = Date.now();
    await rr.drainForShutdown();
    const elapsed = Date.now() - start;

    // AbortSignal.timeout(3000) must fire before the 15 s WEBHOOK_TIMEOUT_MS
    // used by the regular retry ladder, and before any 5+30 s retry delay.
    assert.ok(
      elapsed < 5000,
      `drainForShutdown took ${elapsed}ms — should finish within 5 s (3 s abort + headroom)`,
    );
  });
});

// ---------------------------------------------------------------------------
// 3. Double-signal re-entrancy — second drain delivers zero extra webhooks
// ---------------------------------------------------------------------------

describe("double-signal re-entrancy — drainForShutdown is idempotent", () => {
  let dir: string;
  let hook: ReturnType<typeof makeWebhookServer>;
  let rr: any;

  before(async () => {
    dir = makeTmpDir();
    process.env.REQS_DIR = dir;
    process.env.NO_DB = "true";
    sharedReqs = await import("../../graph/reqs.js");
    rr = await import(`../index.js?d3=${Date.now()}`);
    hook = makeWebhookServer();
  });

  after(async () => {
    delete process.env.REQS_DIR;
    delete process.env.NO_DB;
    rmSync(dir, { recursive: true, force: true });
    await hook.close();
  });

  it("second drainForShutdown call delivers zero additional webhooks", async () => {
    sharedReqs.startReq(hook.url);

    // First drain: flips pending → failed, delivers 1 webhook
    await rr.drainForShutdown();
    assert.strictEqual(hook.calls.length, 1, "first drain should deliver exactly 1 webhook");

    // Second drain (simulates SIGINT arriving after SIGTERM):
    // record is now 'failed', so failPendingReqs() returns [] → no deliveries.
    await rr.drainForShutdown();
    assert.strictEqual(hook.calls.length, 1, "second drain must NOT deliver a second webhook");
  });
});

// ---------------------------------------------------------------------------
// 4. setShuttingDown is exported + drain record is authoritative
// ---------------------------------------------------------------------------

describe("setShuttingDown is exported and drain record is authoritative", () => {
  let dir: string;
  let hook: ReturnType<typeof makeWebhookServer>;
  let rr: any;

  before(async () => {
    dir = makeTmpDir();
    process.env.REQS_DIR = dir;
    process.env.NO_DB = "true";
    sharedReqs = await import("../../graph/reqs.js");
    rr = await import(`../index.js?d4=${Date.now()}`);
    hook = makeWebhookServer();
  });

  after(async () => {
    delete process.env.REQS_DIR;
    delete process.env.NO_DB;
    rmSync(dir, { recursive: true, force: true });
    await hook.close();
  });

  it("setShuttingDown is a callable export", () => {
    assert.strictEqual(
      typeof rr.setShuttingDown,
      "function",
      "setShuttingDown must be exported from repo/index.ts",
    );
  });

  it("after drain, disk record is failed/retryable:true and exactly one webhook delivered", async () => {
    const id = sharedReqs.startReq(hook.url);

    await rr.drainForShutdown();

    // Use checkReq (disk-authoritative) rather than readFileSync with a
    // potentially-stale dir path: sharedReqs resolves REQS_DIR at first load,
    // so the canonical module always reads/writes to the correct location.
    const record = sharedReqs.checkReq(id);
    assert.ok(record, "checkReq should return a record");
    assert.strictEqual(record.status, "failed", "drain must write status:failed");
    assert.strictEqual(record.retryable, true, "drain must write retryable:true");

    assert.strictEqual(hook.calls.length, 1, "exactly one webhook must be delivered");
    assert.strictEqual(hook.calls[0].body.status, "failed");
    assert.strictEqual(hook.calls[0].body.retryable, true);
  });

  it("setShuttingDown is idempotent — calling it again does not throw", () => {
    assert.doesNotThrow(() => rr.setShuttingDown());
  });
});
