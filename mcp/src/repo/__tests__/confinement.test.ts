/**
 * Tests for write-path confinement (plan: code-change-async-and-confinement):
 *   tools.ts    — confinedBashRejection guard, bash env token withholding
 *   agent.ts    — terminalPrResult sentinel (create_pr_not_called)
 *   git_pr.ts   — acquireEphemeralWorktree lifecycle
 *
 * Harness: node:test (tsx --test). No mocks — real git fixtures, real bash.
 */

import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import os from "os";
import { execSync } from "child_process";
import { randomUUID } from "crypto";

import {
  confinedBashRejection,
  get_tools,
  type GetToolsOptions,
  type PrCollector,
} from "../tools.js";
import { terminalPrResult } from "../agent.js";
import {
  acquireWorktree,
  acquireEphemeralWorktree,
  releaseWorktree,
  gitEnv,
  type AgentIdentity,
  type GitHubClient,
  type WorktreeHandle,
  type LandChangeResult,
} from "../git_pr.js";

// ---------------------------------------------------------------------------
// Helpers / fixtures
// ---------------------------------------------------------------------------

function tmpDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "confinement_test_"));
}

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

const TEST_IDENTITY: AgentIdentity = {
  login: "octocat",
  name: "Octocat",
  email: "1+octocat@users.noreply.github.com",
  id: 1,
};

const FAKE_PAT = "ghp_fakepat123456789012345678901234";

function makeStubClient(): GitHubClient {
  return {
    users: {
      getAuthenticated: async () => ({
        data: { login: "octocat", id: 1, email: null, name: "Octocat" },
      }),
    },
    repos: {
      get: async () => ({
        data: { default_branch: "main", permissions: { push: true } },
      }),
    },
    pulls: {
      create: async (params: any) => ({
        data: { number: 1, html_url: `https://github.com/${params.owner}/${params.repo}/pull/1` },
      }),
    },
  } as GitHubClient;
}

function fakeHandle(worktreePath: string): WorktreeHandle {
  return {
    worktreePath,
    baseDir: "/tmp/owner/repo",
    baseSha: "abc",
    baseName: "main",
    branch: "swarm/swarm-change-abc12345",
    owner: "owner",
    repo: "repo",
    runId: randomUUID(),
    runHome: os.tmpdir(),
    _children: new Set(),
    _released: false,
  } as WorktreeHandle;
}

const PR_MODE_OPTS = (worktreePath: string): GetToolsOptions => ({
  prCollector: { result: undefined },
  prMode: {
    handle: fakeHandle(worktreePath),
    identity: TEST_IDENTITY,
    env: gitEnv(TEST_IDENTITY, FAKE_PAT, os.tmpdir()),
    githubClient: makeStubClient(),
  },
  baseCheckoutPath: "/tmp/owner/repo",
});

let testRootDir: string;
let baseCloneDir: string;

before(() => {
  testRootDir = tmpDir();
  baseCloneDir = path.join(testRootDir, "base-clone");
  fs.mkdirSync(baseCloneDir, { recursive: true });
  const env = localEnv();
  execSync("git init -b main .", { cwd: baseCloneDir, env, stdio: "ignore" });
  fs.writeFileSync(path.join(baseCloneDir, "README.md"), "# confinement test repo\n");
  execSync("git add README.md", { cwd: baseCloneDir, env, stdio: "ignore" });
  execSync('git commit -m "Initial commit"', { cwd: baseCloneDir, env, stdio: "ignore" });
});

after(() => {
  fs.rmSync(testRootDir, { recursive: true, force: true });
});

// ---------------------------------------------------------------------------
// 1. confinedBashRejection
// ---------------------------------------------------------------------------

describe("confinedBashRejection", () => {
  const armed = PR_MODE_OPTS("/tmp/.swarm-work/x/owner/repo");
  const ephemeral: GetToolsOptions = { ephemeral: true, baseCheckoutPath: "/tmp/owner/repo" };

  it("allows everything when unarmed", () => {
    assert.equal(confinedBashRejection("git push origin main", undefined), undefined);
    assert.equal(confinedBashRejection('git commit -m "x"', {}), undefined);
    assert.equal(confinedBashRejection("gh pr create --title x", {}), undefined);
  });

  it("still rejects shared-checkout references (baseCheckoutPath guard)", () => {
    const msg = confinedBashRejection("cat /tmp/owner/repo/README.md", armed);
    assert.match(msg ?? "", /shared checkout/);
  });

  it("prMode: rejects git push / commit / remote and gh writes", () => {
    for (const cmd of [
      "git push origin main",
      "git -C /somewhere push --force",
      'git commit -m "sneaky"',
      "git add -A && git commit -m x",
      "git remote get-url origin",
      "gh pr create --title x --body y",
      "gh api -X POST /repos/o/r/pulls",
      "gh repo clone o/r",
    ]) {
      const msg = confinedBashRejection(cmd, armed);
      assert.ok(msg, `should reject: ${cmd}`);
      assert.match(msg!, /rejected/);
    }
  });

  it("prMode: allows read-only git and unrelated commands", () => {
    for (const cmd of [
      "git status",
      "git diff HEAD",
      "git log --oneline -5",
      "git log | grep push",
      "ls -la && cat package.json",
      "grep -rn 'commit' src/",
      "echo push commit remote",
    ]) {
      assert.equal(confinedBashRejection(cmd, armed), undefined, `should allow: ${cmd}`);
    }
  });

  it("ephemeral: same blocklist, preview hint", () => {
    const msg = confinedBashRejection("git push origin main", ephemeral);
    assert.ok(msg);
    assert.match(msg!, /read-only preview/);
    assert.equal(confinedBashRejection("git diff HEAD", ephemeral), undefined);
  });
});

// ---------------------------------------------------------------------------
// 2. bash tool env — tokens withheld on confined runs
// ---------------------------------------------------------------------------

describe("bash tool GitHub token env", () => {
  async function runBashEcho(options: GetToolsOptions | undefined, repoPath: string): Promise<string> {
    const tools = await get_tools(
      repoPath,
      "fake-api-key",
      FAKE_PAT,
      { bash: true },
      undefined, // provider (non-Anthropic branch)
      undefined, undefined, undefined, undefined, undefined, undefined,
      undefined, undefined, undefined, undefined, undefined, undefined,
      options,
    );
    const bash = (tools as any).bash;
    assert.ok(bash?.execute, "bash tool must exist");
    return String(await bash.execute({ command: 'echo "T:${GH_TOKEN}:${GITHUB_TOKEN}:"' }));
  }

  it("unconfined run: PAT is exported to bash", async () => {
    const out = await runBashEcho(undefined, baseCloneDir);
    assert.ok(out.includes(`T:${FAKE_PAT}:${FAKE_PAT}:`), `got: ${out}`);
  });

  it("prMode run: tokens are blanked, masking ambient env too", async () => {
    const prevGh = process.env.GH_TOKEN;
    process.env.GH_TOKEN = "ambient-container-token";
    try {
      const out = await runBashEcho(PR_MODE_OPTS(baseCloneDir), baseCloneDir);
      assert.ok(out.includes("T:::"), `tokens must be empty, got: ${out}`);
      assert.ok(!out.includes(FAKE_PAT), "PAT must not leak into confined bash");
      assert.ok(!out.includes("ambient-container-token"), "ambient token must be masked");
    } finally {
      if (prevGh === undefined) delete process.env.GH_TOKEN;
      else process.env.GH_TOKEN = prevGh;
    }
  });

  it("ephemeral run: tokens are blanked", async () => {
    const out = await runBashEcho(
      { ephemeral: true, baseCheckoutPath: "/tmp/owner/repo" },
      baseCloneDir,
    );
    assert.ok(out.includes("T:::"), `tokens must be empty, got: ${out}`);
  });
});

// ---------------------------------------------------------------------------
// 3. terminalPrResult sentinel
// ---------------------------------------------------------------------------

describe("terminalPrResult", () => {
  const success: LandChangeResult = {
    ok: true,
    url: "https://github.com/o/r/pull/1",
    number: 1,
    branch: "swarm/swarm-change-abc12345",
    base: "main",
    headSha: "deadbeef",
    diff: "--- a\n+++ b\n",
    filesChanged: 1,
  };

  it("passes a collector result through untouched", () => {
    assert.equal(terminalPrResult(success, { create_pr: true }, undefined), success);
  });

  it("create_pr enabled + no result → create_pr_not_called sentinel", () => {
    const pr = terminalPrResult(undefined, { create_pr: true }, undefined);
    assert.ok(pr && !pr.ok);
    if (pr && !pr.ok) {
      assert.equal(pr.failure, "create_pr_not_called");
      assert.equal(pr.diff, "");
      assert.match(pr.error, /without invoking the create_pr tool/);
    }
  });

  it("sentinel names the worktree branch when prMode is present", () => {
    const prMode = {
      handle: fakeHandle("/tmp/.swarm-work/x/owner/repo"),
      identity: TEST_IDENTITY,
      env: {},
      githubClient: makeStubClient(),
    };
    const pr = terminalPrResult(undefined, { create_pr: true }, prMode);
    assert.ok(pr && !pr.ok);
    if (pr && !pr.ok) {
      assert.match(pr.error, /swarm\/swarm-change-abc12345/);
    }
  });

  it("create_pr disabled → undefined (no sentinel noise on normal runs)", () => {
    assert.equal(terminalPrResult(undefined, { bash: true }, undefined), undefined);
    assert.equal(terminalPrResult(undefined, undefined, undefined), undefined);
  });
});

// ---------------------------------------------------------------------------
// 4. acquireEphemeralWorktree lifecycle
// ---------------------------------------------------------------------------

describe("acquireEphemeralWorktree", () => {
  it("creates a detached worktree at HEAD; base checkout stays clean; release removes it", async () => {
    const runId = randomUUID();
    const env = localEnv();
    const baseHead = execSync("git rev-parse HEAD", { cwd: baseCloneDir, env, encoding: "utf8" }).trim();

    const result = await acquireEphemeralWorktree({
      baseDir: baseCloneDir,
      owner: "owner",
      repo: "repo",
      runId,
    });
    assert.ok(result.ok, `acquire failed: ${(result as any).error}`);
    const handle = result.handle;

    try {
      assert.ok(fs.existsSync(handle.worktreePath), "worktree dir must exist");
      const wtHead = execSync("git rev-parse HEAD", { cwd: handle.worktreePath, env, encoding: "utf8" }).trim();
      assert.equal(wtHead, baseHead, "worktree must be at the base HEAD");
      assert.equal(handle.branch, "", "ephemeral worktree is detached — no branch");

      // Dirty the worktree — the base must stay pristine.
      fs.writeFileSync(path.join(handle.worktreePath, "README.md"), "dirtied by preview\n");
      const baseStatus = execSync("git status --porcelain", { cwd: baseCloneDir, env, encoding: "utf8" }).trim();
      assert.equal(baseStatus, "", "base checkout must remain clean while worktree is dirty");
    } finally {
      await releaseWorktree(handle);
    }

    assert.ok(!fs.existsSync(handle.worktreePath), "worktree dir must be removed on release");
    const baseStatusAfter = execSync("git status --porcelain", { cwd: baseCloneDir, env: localEnv(), encoding: "utf8" }).trim();
    assert.equal(baseStatusAfter, "", "base checkout must remain clean after release");
  });

  it("PR worktree release deletes the swarm/swarm-change-* branch ref from the base repo", async () => {
    const runId = randomUUID();
    const env = localEnv();
    const baseSha = execSync("git rev-parse HEAD", { cwd: baseCloneDir, env, encoding: "utf8" }).trim();
    const result = await acquireWorktree({
      baseDir: baseCloneDir,
      owner: "owner",
      repo: "repo",
      runId,
      pat: FAKE_PAT,
      githubClient: makeStubClient(),
      commit: baseSha, // pin to a local sha so no fetch happens
    });
    assert.ok(result.ok, `acquire failed: ${(result as any).error}`);
    const handle = result.handle;
    assert.equal(handle.branch, `swarm/swarm-change-${runId.slice(0, 8)}`);

    const listBranches = () =>
      execSync("git branch --list 'swarm/*'", { cwd: baseCloneDir, env, encoding: "utf8" }).trim();
    assert.ok(listBranches().includes(handle.branch), "branch ref must exist while worktree is live");

    await releaseWorktree(handle);
    assert.equal(listBranches(), "", "branch ref must be deleted on release");
  });

  it("rejects a traversal runId before any filesystem side effect", async () => {
    const result = await acquireEphemeralWorktree({
      baseDir: baseCloneDir,
      owner: "owner",
      repo: "repo",
      runId: "../../etc",
    });
    assert.ok(!result.ok, "traversal runId must be rejected");
    if (!result.ok) assert.match(result.error, /Path traversal rejected/);
  });
});
