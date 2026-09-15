/**
 * Tests for captureWorktreeDiff (git_pr.ts): the ground-truth diff of an
 * ephemeral preview worktree, read before releaseWorktree discards it.
 *
 * Harness: node:test (tsx --test). Real git fixtures, no mocks. The secret
 * scan shells out to gitleaks and fails closed when the binary is absent,
 * so the success-path assertions branch on whether it is installed.
 */

import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import os from "os";
import { execSync } from "child_process";
import { randomUUID } from "crypto";

import {
  acquireEphemeralWorktree,
  releaseWorktree,
  captureWorktreeDiff,
  type WorktreeHandle,
} from "../git_pr.js";

function localEnv(): NodeJS.ProcessEnv {
  return {
    PATH: process.env.PATH ?? "/usr/local/bin:/usr/bin:/bin",
    HOME: os.tmpdir(),
    GIT_TERMINAL_PROMPT: "0",
    GIT_ASKPASS: "",
    GIT_CONFIG_GLOBAL: "/dev/null",
    GIT_CONFIG_SYSTEM: "/dev/null",
    GIT_AUTHOR_NAME: "Test",
    GIT_AUTHOR_EMAIL: "test@example.com",
    GIT_COMMITTER_NAME: "Test",
    GIT_COMMITTER_EMAIL: "test@example.com",
  };
}

function hasGitleaks(): boolean {
  try {
    execSync("gitleaks version", { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

let testRootDir: string;
let baseCloneDir: string;

before(() => {
  testRootDir = fs.mkdtempSync(path.join(os.tmpdir(), "worktree_diff_test_"));
  baseCloneDir = path.join(testRootDir, "base-clone");
  fs.mkdirSync(baseCloneDir, { recursive: true });
  const env = localEnv();
  execSync("git init -b main .", { cwd: baseCloneDir, env, stdio: "ignore" });
  fs.writeFileSync(path.join(baseCloneDir, "README.md"), "# worktree diff test repo\n");
  fs.writeFileSync(path.join(baseCloneDir, ".gitignore"), "ignored.log\n");
  execSync("git add .", { cwd: baseCloneDir, env, stdio: "ignore" });
  execSync('git commit -m "Initial commit"', { cwd: baseCloneDir, env, stdio: "ignore" });
});

after(() => {
  fs.rmSync(testRootDir, { recursive: true, force: true });
});

async function withWorktree<T>(fn: (handle: WorktreeHandle) => Promise<T>): Promise<T> {
  const result = await acquireEphemeralWorktree({
    baseDir: baseCloneDir,
    owner: "owner",
    repo: "repo",
    runId: randomUUID(),
  });
  assert.ok(result.ok, `acquire failed: ${(result as any).error}`);
  try {
    return await fn(result.handle);
  } finally {
    await releaseWorktree(result.handle);
  }
}

describe("captureWorktreeDiff", () => {
  it("reports no_changes for an untouched worktree", async () => {
    await withWorktree(async (handle) => {
      const out = await captureWorktreeDiff(handle);
      assert.equal(out.ok, false);
      if (!out.ok) assert.equal(out.failure, "no_changes");
    });
  });

  it("includes a modified file AND a brand-new untracked file, ignoring .gitignore'd ones", async () => {
    await withWorktree(async (handle) => {
      fs.writeFileSync(path.join(handle.worktreePath, "README.md"), "# edited\n");
      fs.mkdirSync(path.join(handle.worktreePath, "src"), { recursive: true });
      fs.writeFileSync(path.join(handle.worktreePath, "src", "new.test.ts"), "export const x = 1;\n");
      fs.writeFileSync(path.join(handle.worktreePath, "ignored.log"), "noise\n");

      const out = await captureWorktreeDiff(handle);

      if (hasGitleaks()) {
        assert.ok(out.ok, `expected ok, got ${JSON.stringify(out)}`);
        if (out.ok) {
          assert.equal(out.filesChanged, 2);
          assert.ok(out.diff.includes("+++ b/README.md"), "modified file present");
          assert.ok(out.diff.includes("+++ b/src/new.test.ts"), "new untracked file present");
          assert.ok(out.diff.includes("new file mode"), "new file recorded as such");
          assert.ok(!out.diff.includes("ignored.log"), "ignored file excluded");
        }
      } else {
        // Fails closed on the scan, but only after staging succeeded and the
        // size checks passed — so the failure is the scan, nothing earlier.
        assert.equal(out.ok, false);
        if (!out.ok) {
          assert.equal(out.failure, "secrets_detected");
          assert.match(out.error, /gitleaks/);
        }
      }

      // The base checkout is never touched.
      const baseStatus = execSync("git status --porcelain", {
        cwd: baseCloneDir,
        env: localEnv(),
        encoding: "utf8",
      }).trim();
      assert.equal(baseStatus, "");
    });
  });

  it("enforces the file cap before reading the diff", async () => {
    await withWorktree(async (handle) => {
      fs.writeFileSync(path.join(handle.worktreePath, "a.txt"), "a\n");
      fs.writeFileSync(path.join(handle.worktreePath, "b.txt"), "b\n");
      const out = await captureWorktreeDiff(handle, { maxFiles: 1 });
      assert.equal(out.ok, false);
      if (!out.ok) assert.equal(out.failure, "change_too_large");
    });
  });

  it("enforces the byte cap", async () => {
    await withWorktree(async (handle) => {
      fs.writeFileSync(path.join(handle.worktreePath, "big.txt"), "x".repeat(4096) + "\n");
      const out = await captureWorktreeDiff(handle, { maxBytes: 1024 });
      assert.equal(out.ok, false);
      if (!out.ok) assert.equal(out.failure, "change_too_large");
    });
  });

  it("reports git_failed when the worktree is already gone", async () => {
    const handle = {
      worktreePath: path.join(testRootDir, "does-not-exist"),
      baseDir: baseCloneDir,
      baseSha: "",
      baseName: "main",
      branch: "",
      owner: "owner",
      repo: "repo",
      runId: randomUUID(),
      runHome: os.tmpdir(),
      _children: new Set(),
      _released: false,
    } as unknown as WorktreeHandle;
    const out = await captureWorktreeDiff(handle);
    assert.equal(out.ok, false);
    if (!out.ok) assert.equal(out.failure, "git_failed");
  });
});
