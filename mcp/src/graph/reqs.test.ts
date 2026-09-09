import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, isAbsolute } from "node:path";

/**
 * ensureDir() path-resolution tests.
 *
 * ensureDir() is private, so we test it indirectly:
 *   - Set REQS_DIR before importing the module (env is read at module load time).
 *   - Call startReq() which triggers ensureDir() → creates the directory.
 *   - Assert the directory was created at the expected resolved path.
 *
 * Each describe block uses a fresh dynamic import (with a cache-busting
 * query param) to isolate the module-level REQS_DIR constant.
 */

describe("ensureDir — absolute REQS_DIR", () => {
  let absDir: string;

  before(() => {
    absDir = mkdtempSync(join(tmpdir(), "reqs-abs-"));
    // Remove it so ensureDir has to create it, proving resolution worked
    rmSync(absDir, { recursive: true, force: true });
    process.env.REQS_DIR = absDir;
  });

  after(() => {
    delete process.env.REQS_DIR;
    rmSync(absDir, { recursive: true, force: true });
  });

  it("resolves an absolute REQS_DIR verbatim (not prepended with cwd)", async () => {
    // Dynamic import so REQS_DIR env is picked up fresh
    const { startReq } = await import(`./reqs.js?abs=${Date.now()}`);
    startReq(); // triggers ensureDir() internally
    assert.ok(
      isAbsolute(absDir),
      "test setup: absDir must be absolute"
    );
    assert.ok(
      existsSync(absDir),
      `directory should be created at the absolute path "${absDir}"`
    );
    // Guard: make sure it was NOT created under cwd instead
    const wrongPath = join(process.cwd(), absDir);
    assert.ok(
      !existsSync(wrongPath),
      `directory must NOT be created at cwd-prefixed path "${wrongPath}"`
    );
  });
});

describe("ensureDir — relative REQS_DIR", () => {
  let relName: string;
  let expectedDir: string;

  before(() => {
    relName = `.reqs-test-${Date.now()}`;
    expectedDir = join(process.cwd(), relName);
    process.env.REQS_DIR = relName;
  });

  after(() => {
    delete process.env.REQS_DIR;
    rmSync(expectedDir, { recursive: true, force: true });
  });

  it("resolves a relative REQS_DIR under process.cwd()", async () => {
    const { startReq } = await import(`./reqs.js?rel=${Date.now()}`);
    startReq();
    assert.ok(
      existsSync(expectedDir),
      `directory should be created at cwd-joined path "${expectedDir}"`
    );
  });
});
