/**
 * Non-streaming repo-agent background failures must persist the SDK
 * `isRetryable` flag so GET /progress and the terminal webhook agree.
 *
 * IMPORT RULE:
 *   No top-level static import of reqs.js or repo/index.js — those hoist and
 *   freeze REQS_DIR before this file assigns it. Set REQS_DIR and NO_DB,
 *   then dynamic-import graph/reqs.js with no cache-busting query, then
 *   repo/index.js. index.ts statically imports ../graph/reqs.js; a
 *   cache-busted reqs import is a different module and failReq writes would
 *   not be visible here.
 *
 *   REQS_DIR is captured once at reqs.ts load. If another file already
 *   imported it, a temp dir set here is ignored. Assert with that module's
 *   checkReq, never readFileSync on a path built from this file's REQS_DIR.
 *   Do not delete the canonical reqs directory.
 *
 *   request_id comes from startReq(). A fake id is rejected by reqFile and
 *   writeToDisk swallows that, so checkReq would return null.
 *
 *   Do not import graph/routes.ts. checkReq is the GET /progress body.
 *   Do not call drainForShutdown or setShuttingDown on the shared index.
 */
import { describe, it, before, after, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

process.env.NO_DB = "true";
process.env.REQS_DIR =
  process.env.REQS_DIR || mkdtempSync(join(tmpdir(), "reqs-retryable-"));

type CheckReq = (id: string) => {
  status: string;
  error?: unknown;
  retryable?: boolean;
  content?: unknown;
  result?: unknown;
};
type StartReq = (webhookUrl?: string) => string;
type FailReq = (id: string, error: unknown, retryable?: boolean) => void;

let checkReq: CheckReq;
let startReq: StartReq;
let failReq: FailReq;
let SHUTDOWN_ORPHAN_ERROR: string;
let recordRepoAgentBackgroundFailure: (opts: {
  request_id: string;
  error: unknown;
  aborted: boolean;
  shuttingDown: boolean;
  webhookUrl?: string;
  emit: (event: { type: "error"; error: string; timestamp: string }) => void;
}) => string | undefined;
let errorIsRetryable: (error: unknown) => boolean;

const ERROR_TEXT = "Failed to process successful response";
const TOKEN = "ghp_RetryableTimeoutSecretToken123";

before(async () => {
  const reqs = await import("../../graph/reqs.js");
  checkReq = reqs.checkReq;
  startReq = reqs.startReq;
  failReq = reqs.failReq;
  SHUTDOWN_ORPHAN_ERROR = reqs.SHUTDOWN_ORPHAN_ERROR;
  const repo = await import("../index.js");
  recordRepoAgentBackgroundFailure = repo.recordRepoAgentBackgroundFailure;
  const retryable = await import("../retryable.js");
  errorIsRetryable = retryable.errorIsRetryable;
});

function sdkError(
  overrides: { isRetryable?: unknown; cause?: unknown; message?: string } = {},
): Error & { isRetryable?: unknown; cause?: unknown } {
  const err = new Error(overrides.message ?? ERROR_TEXT) as Error & {
    isRetryable?: unknown;
    cause?: unknown;
  };
  if ("isRetryable" in overrides) err.isRetryable = overrides.isRetryable;
  if ("cause" in overrides) err.cause = overrides.cause;
  return err;
}

function record(
  error: unknown,
  extra: {
    aborted?: boolean;
    shuttingDown?: boolean;
    webhookUrl?: string;
    emit?: (event: { type: "error"; error: string; timestamp: string }) => void;
  } = {},
) {
  const request_id = startReq();
  const stored = recordRepoAgentBackgroundFailure({
    request_id,
    error,
    aborted: extra.aborted ?? false,
    shuttingDown: extra.shuttingDown ?? false,
    webhookUrl: extra.webhookUrl,
    emit: extra.emit ?? (() => {}),
  });
  return { request_id, stored, progress: checkReq(request_id) };
}

describe("errorIsRetryable", () => {
  it("is true only for isRetryable === true on the error or its cause", () => {
    assert.equal(errorIsRetryable(sdkError({ isRetryable: true })), true);
    const caused = sdkError({ cause: sdkError({ isRetryable: true }) });
    assert.equal(errorIsRetryable(caused), true);
    assert.equal(errorIsRetryable(sdkError({ isRetryable: false })), false);
    assert.equal(errorIsRetryable(sdkError()), false);
    assert.equal(errorIsRetryable(sdkError({ isRetryable: "true" })), false);
  });

  it("returns on a cause cycle and stops on a non-object cause", () => {
    const cycle = sdkError({ isRetryable: false });
    cycle.cause = cycle;
    assert.equal(errorIsRetryable(cycle), false);

    const wrapped = sdkError({ isRetryable: false, cause: "read ETIMEDOUT" });
    assert.equal(errorIsRetryable(wrapped), false);

    const viaString = sdkError({
      cause: Object.assign(new Error("inner"), {
        isRetryable: true,
      }),
    });
    // non-object stops; a real cause object is still walked
    assert.equal(errorIsRetryable(viaString), true);
    assert.equal(errorIsRetryable(sdkError({ cause: null })), false);
  });
});

describe("recordRepoAgentBackgroundFailure", () => {
  it("stores a retryable SDK timeout as retryable with the same error string", () => {
    const { progress } = record(sdkError({ isRetryable: true }));
    assert.deepEqual(
      { status: progress.status, retryable: progress.retryable, error: progress.error },
      { status: "failed", retryable: true, error: ERROR_TEXT },
    );
    assert.equal("content" in progress, false);
  });

  it("reads the flag from cause and does not append the cause to the error string", () => {
    const cause = Object.assign(new Error("read ETIMEDOUT"), { isRetryable: true });
    const { progress } = record(sdkError({ cause }));
    assert.equal(progress.status, "failed");
    assert.equal(progress.retryable, true);
    assert.equal(progress.error, ERROR_TEXT);
    assert.equal(String(progress.error).includes("ETIMEDOUT"), false);
  });

  it("stays false when the flag is false, missing, or the string true", () => {
    for (const error of [
      sdkError({ isRetryable: false }),
      sdkError(),
      sdkError({ isRetryable: "true" }),
    ]) {
      const { progress } = record(error);
      assert.equal(progress.status, "failed");
      assert.equal(progress.retryable, false);
      assert.equal(progress.error, ERROR_TEXT);
      assert.equal("content" in progress, false);
    }
  });

  it("stores an abort as non-retryable even when the error is marked retryable", () => {
    const { progress } = record(sdkError({ isRetryable: true }), { aborted: true });
    assert.equal(progress.status, "failed");
    assert.equal(progress.retryable, false);
    assert.equal(progress.error, "aborted");
  });

  it("does nothing when shutting down, leaving a pending record and skipping the webhook", async () => {
    const original = globalThis.fetch;
    let calls = 0;
    globalThis.fetch = (async () => {
      calls += 1;
      return new Response("ok", { status: 200 });
    }) as typeof fetch;
    try {
      const request_id = startReq("https://hooks.example/terminal");
      const stored = recordRepoAgentBackgroundFailure({
        request_id,
        error: sdkError({ isRetryable: true }),
        aborted: true,
        shuttingDown: true,
        webhookUrl: "https://hooks.example/terminal",
        emit: () => {
          throw new Error("emit should not run");
        },
      });
      assert.equal(stored, undefined);
      const progress = checkReq(request_id);
      assert.equal(progress.status, "pending");
      assert.equal(progress.error, undefined);
      assert.equal(progress.retryable, undefined);
      await new Promise((r) => setTimeout(r, 20));
      assert.equal(calls, 0);
    } finally {
      globalThis.fetch = original;
    }
  });

  it("does not overwrite a shutdown-orphan record", () => {
    const request_id = startReq();
    failReq(request_id, SHUTDOWN_ORPHAN_ERROR, true);
    const before = checkReq(request_id);
    const stored = recordRepoAgentBackgroundFailure({
      request_id,
      error: sdkError({ isRetryable: true }),
      aborted: false,
      shuttingDown: true,
      webhookUrl: "https://hooks.example/terminal",
      emit: () => {},
    });
    assert.equal(stored, undefined);
    assert.deepEqual(checkReq(request_id), before);
    assert.equal(before.status, "failed");
    assert.equal(before.retryable, true);
    assert.equal(before.error, SHUTDOWN_ORPHAN_ERROR);
  });

  it("redacts credentials in the stored string without a pat argument", () => {
    const message = `clone failed for https://user:${TOKEN}@github.com/org/repo`;
    const { progress } = record(sdkError({ isRetryable: true, message }));
    assert.equal(typeof progress.error, "string");
    assert.equal(String(progress.error).includes(TOKEN), false);
    assert.match(String(progress.error), /https:\/\/\*\*\*@github\.com/);
  });

  it("emits the stored string, not the Error object", () => {
    const events: Array<{ type: string; error: unknown }> = [];
    const err = sdkError({ isRetryable: true });
    const { stored } = record(err, {
      emit: (event) => events.push(event),
    });
    assert.equal(events.length, 1);
    assert.equal(events[0].type, "error");
    assert.equal(events[0].error, ERROR_TEXT);
    assert.equal(stored, ERROR_TEXT);
    assert.equal(typeof events[0].error, "string");
  });
});

describe("terminal webhook", () => {
  const original = globalThis.fetch;
  let bodies: Array<Record<string, unknown>>;

  before(() => {
    bodies = [];
    globalThis.fetch = (async (_url: unknown, init?: RequestInit) => {
      bodies.push(JSON.parse(String(init?.body ?? "{}")));
      return new Response("ok", { status: 200 });
    }) as typeof fetch;
  });

  after(() => {
    globalThis.fetch = original;
  });

  afterEach(() => {
    bodies = [];
  });

  async function posted(): Promise<Record<string, unknown>> {
    await new Promise((r) => setTimeout(r, 30));
    assert.equal(bodies.length, 1);
    return bodies[0];
  }

  it("posts retryable true and the same error string for an SDK timeout", async () => {
    const { request_id } = record(sdkError({ isRetryable: true }), {
      webhookUrl: "https://hooks.example/terminal",
    });
    const body = await posted();
    assert.deepEqual(body, {
      request_id,
      status: "failed",
      error: ERROR_TEXT,
      retryable: true,
    });
  });

  it("posts retryable false for a non-retryable error and for an abort", async () => {
    const nonRetryable = record(sdkError({ isRetryable: false }), {
      webhookUrl: "https://hooks.example/terminal",
    });
    const first = await posted();
    assert.equal(first.request_id, nonRetryable.request_id);
    assert.equal(first.retryable, false);
    assert.equal(first.error, ERROR_TEXT);

    bodies = [];
    const aborted = record(sdkError({ isRetryable: true }), {
      aborted: true,
      webhookUrl: "https://hooks.example/terminal",
    });
    const second = await posted();
    assert.equal(second.request_id, aborted.request_id);
    assert.equal(second.retryable, false);
    assert.equal(second.error, "aborted");
  });
});

describe("production logs", () => {
  it("logs the Error object on failure and the abort line on abort", () => {
    const errors: unknown[][] = [];
    const logs: unknown[][] = [];
    const origError = console.error;
    const origLog = console.log;
    console.error = (...args: unknown[]) => {
      errors.push(args);
    };
    console.log = (...args: unknown[]) => {
      logs.push(args);
    };
    try {
      const cause = Object.assign(new Error("read ETIMEDOUT"), { isRetryable: true });
      const err = sdkError({ cause });
      err.stack = "Error: Failed to process successful response\n    at test";
      const { request_id } = record(err);
      const failure = errors.find(
        (args) => args[0] === "[repo_agent] Background work failed with error:",
      );
      assert.ok(failure, "expected the background-failure console.error");
      assert.equal(failure[1], err);
      assert.equal((failure[1] as { cause?: { message?: string } }).cause?.message, "read ETIMEDOUT");
      assert.equal(typeof (failure[1] as Error).stack, "string");

      logs.length = 0;
      const aborted = record(sdkError({ isRetryable: true }), { aborted: true });
      assert.deepEqual(logs, [[`[repo_agent] Run aborted: ${aborted.request_id}`]]);
      assert.equal(
        logs.some((args) => args.includes("aborted") && !String(args[0]).includes(aborted.request_id)),
        false,
      );
      assert.notEqual(request_id, aborted.request_id);
    } finally {
      console.error = origError;
      console.log = origLog;
    }
  });
});

describe("non-streaming catch wiring", () => {
  it("delegates the catch to recordRepoAgentBackgroundFailure and does not failReq itself", () => {
    const here = fileURLToPath(new URL("../index.ts", import.meta.url));
    const src = readFileSync(here, "utf8");
    const start = src.indexOf("      .catch((error) => {\n        recordRepoAgentBackgroundFailure({");
    assert.notEqual(start, -1, "non-streaming .catch must call recordRepoAgentBackgroundFailure");
    const end = src.indexOf("      })", start);
    assert.notEqual(end, -1);
    const catchBlock = src.slice(start, end);
    assert.match(catchBlock, /aborted:\s*abortController\.signal\.aborted/);
    assert.match(catchBlock, /shuttingDown,/);
    assert.match(catchBlock, /webhookUrl:\s*body\.webhookUrl/);
    assert.match(catchBlock, /emit:\s*\(event\)\s*=>\s*bus\.emit\(event\)/);
    assert.equal(catchBlock.includes("failReq"), false);
    assert.equal(catchBlock.includes("retryable: false"), false);
    assert.equal(catchBlock.includes("postTerminalWebhook"), false);
    assert.equal(catchBlock.includes("bus.emit"), true);
  });
});
