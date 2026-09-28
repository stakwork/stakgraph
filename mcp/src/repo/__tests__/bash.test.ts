/**
 * `executeBashCommand` must never let a child's output grow the heap without
 * bound. Regression tests for the 2026-09-25 OOM: after a timeout or the
 * stdout cap, the whole process GROUP is killed and the capture is torn down,
 * so an orphaned grandchild cannot keep feeding the closure; stderr is capped.
 */
import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { executeBashCommand } from "../bash.js";

function alive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

async function readPid(file: string): Promise<number> {
  for (let i = 0; i < 50 && !existsSync(file); i++) await sleep(20);
  return parseInt(readFileSync(file, "utf-8").trim(), 10);
}

describe("executeBashCommand output bounds", () => {
  let dir: string;
  before(() => {
    dir = mkdtempSync(join(tmpdir(), "bash-test-"));
  });
  after(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  it("kills grandchildren on timeout, not just the shell", async () => {
    const pidFile = join(dir, "timeout.pid");
    const result = await executeBashCommand(
      `(sleep 30 & echo $! > ${pidFile}; wait)`,
      dir,
      300,
    );
    assert.match(result, /timed out/);
    const pid = await readPid(pidFile);
    await sleep(100);
    assert.equal(alive(pid), false, `orphaned sleep ${pid} survived the timeout`);
  });

  it("kills the producer when stdout hits the cap and returns truncated output", async () => {
    const pidFile = join(dir, "cap.pid");
    const result = await executeBashCommand(
      `(yes & echo $! > ${pidFile}; wait)`,
      dir,
      10000,
    );
    assert.match(result, /output truncated due to size limit/);
    assert.ok(result.length < 9 * 1024 * 1024, `result is ${result.length} chars`);
    const pid = await readPid(pidFile);
    await sleep(100);
    assert.equal(alive(pid), false, `orphaned yes ${pid} survived the output cap`);
  });

  it("caps stderr in the error message", async () => {
    const result = await executeBashCommand(
      `head -c 3000000 /dev/zero | tr '\\0' e >&2; exit 2`,
      dir,
      10000,
    );
    assert.match(result, /Command failed with code 2/);
    assert.ok(result.length < 300 * 1024, `error message is ${result.length} chars`);
  });

  it("still returns normal output and exit codes", async () => {
    assert.equal(await executeBashCommand("printf hello", dir, 5000), "hello");
    assert.equal(await executeBashCommand("grep zzz /dev/null", dir, 5000), "No matches found");
  });
});
