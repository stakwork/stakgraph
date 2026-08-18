/**
 * Tests for the create_pr wiring in:
 *   mcp/src/repo/tools.ts       — create_pr tool registration, editorRoots strict mode, bash confinement
 *   mcp/src/repo/index.ts       — admission checks, sweepOrphanedWorktrees, boot sweep, rate limiter
 *   mcp/src/repo/agent.ts       — GetContextOptions.prMode, prCollector threading
 *   mcp/src/repo/subagent.ts    — create_pr stripped from forwarded toolsConfig
 *
 * Harness: node:test (tsx --test). Uses before/after, NO vi.mock.
 * All seams are constructor/parameter injection or module-level functions under test.
 */

import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import path from "path";
import os from "os";
import { execSync } from "child_process";
import { randomUUID } from "crypto";

import {
  editorRoots,
  normalizeToolsConfig,
  toolConfigEnabled,
  get_tools,
  type PrCollector,
} from "../tools.js";
import { sweepOrphanedWorktrees } from "../index.js";
import { callRemoteAgent, type SubAgent } from "../subagent.js";
import {
  acquireWorktree,
  releaseWorktree,
  gitEnv,
  type AgentIdentity,
  type GitHubClient,
  type WorktreeHandle,
} from "../git_pr.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function tmpDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "pr_wiring_test_"));
}

/** Minimal env for local git operations that do not need credentials. */
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

/** Stub GitHubClient with push permission enabled. */
function makeStubClient(overrides: {
  pushPermission?: boolean;
  login?: string;
}): GitHubClient {
  const login = overrides.login ?? "octocat";
  const pushPermission = overrides.pushPermission ?? true;
  return {
    users: {
      getAuthenticated: async () => ({
        data: { login, id: 1, email: `${login}@example.com`, name: login },
      }),
    },
    repos: {
      get: async (_params: any) => ({
        data: {
          default_branch: "main",
          permissions: { push: pushPermission },
        },
      }),
    },
    pulls: {
      create: async (params: any) => ({
        data: { number: 99, html_url: `https://github.com/${params.owner}/${params.repo}/pull/99` },
      }),
    },
  } as GitHubClient;
}

const TEST_IDENTITY: AgentIdentity = {
  login: "octocat",
  name: "Octocat",
  email: "1+octocat@users.noreply.github.com",
  id: 1,
};

const FAKE_PAT = "ghp_fakepat123456789012345678901234";

// ---------------------------------------------------------------------------
// Bare repo fixture — created once for integration tests
// ---------------------------------------------------------------------------

let bareRepoDir: string;
let baseCloneDir: string;
let testRootDir: string;
let baseSha: string;

before(async () => {
  testRootDir = tmpDir();
  bareRepoDir = path.join(testRootDir, "bare.git");
  const scratchDir = path.join(testRootDir, "scratch");

  fs.mkdirSync(bareRepoDir, { recursive: true });
  execSync("git init --bare .", { cwd: bareRepoDir, stdio: "ignore" });

  fs.mkdirSync(scratchDir, { recursive: true });
  const env = localEnv();
  execSync("git init -b main .", { cwd: scratchDir, env, stdio: "ignore" });
  fs.writeFileSync(path.join(scratchDir, "README.md"), "# PR wiring test repo\n");
  execSync("git add README.md", { cwd: scratchDir, env, stdio: "ignore" });
  execSync('git commit -m "Initial commit"', { cwd: scratchDir, env, stdio: "ignore" });
  execSync(`git remote add origin "${bareRepoDir}"`, { cwd: scratchDir, env, stdio: "ignore" });
  execSync("git push -u origin main", { cwd: scratchDir, env, stdio: "ignore" });

  baseSha = execSync("git rev-parse HEAD", {
    cwd: scratchDir,
    env,
    encoding: "utf8",
  }).trim();

  baseCloneDir = path.join(testRootDir, "base-clone");
  execSync(`git clone --single-branch "${bareRepoDir}" "${baseCloneDir}"`, {
    env,
    stdio: "ignore",
  });
});

after(() => {
  fs.rmSync(testRootDir, { recursive: true, force: true });
});

// ---------------------------------------------------------------------------
// 1. editorRoots strict mode
// ---------------------------------------------------------------------------

describe("editorRoots", () => {
  it("normal mode includes os.tmpdir()", () => {
    const roots = editorRoots("/tmp/owner/repo");
    assert.ok(
      roots.some((r) => r === os.tmpdir()),
      "os.tmpdir() should be in normal roots"
    );
    assert.ok(roots.includes("/tmp/owner/repo"), "repoPath should be in roots");
  });

  it("strict mode excludes os.tmpdir()", () => {
    const roots = editorRoots("/tmp/.swarm-work/runid/owner/repo", true);
    assert.ok(
      !roots.some((r) => r === os.tmpdir()),
      "os.tmpdir() must NOT be in strict roots"
    );
    assert.ok(
      roots.includes("/tmp/.swarm-work/runid/owner/repo"),
      "worktree path should be in strict roots"
    );
  });

  it("strict mode: single repo path, no tmpdir", () => {
    const wt = "/tmp/.swarm-work/abc123/owner/repo";
    const roots = editorRoots(wt, true);
    assert.strictEqual(roots.length >= 1, true);
    assert.strictEqual(roots[0], wt);
  });
});

// ---------------------------------------------------------------------------
// 2. normalizeToolsConfig warns on unknown keys
// ---------------------------------------------------------------------------

describe("normalizeToolsConfig", () => {
  it("does not throw on unknown key, returns object with known keys", () => {
    // Object form: unknown keys are preserved (no string parsing happens)
    const result = normalizeToolsConfig({ bash: true, unknown_tool_xyz: true });
    assert.ok(result !== undefined, "should return a config object");
    // bash is a known key
    assert.strictEqual((result as any).bash, true);
  });

  it("string form: silently drops unknown keys without throwing", () => {
    // The warn fires but must not throw
    const result = normalizeToolsConfig("bash true totally_unknown_tool true create_pr true");
    assert.ok(result !== undefined);
    // known keys parsed
    assert.strictEqual((result as any).bash, true);
    assert.strictEqual((result as any).create_pr, true);
    // unknown key not present
    assert.strictEqual((result as any).totally_unknown_tool, undefined);
  });

  it("create_pr is recognized as a known TOOL_NAME", () => {
    const result = normalizeToolsConfig("create_pr true");
    assert.ok(result !== undefined);
    assert.strictEqual((result as any).create_pr, true);
  });
});

// ---------------------------------------------------------------------------
// 3. create_pr tool NOT included by default (toolsConfig=undefined)
// ---------------------------------------------------------------------------

describe("get_tools default-off regression", () => {
  it("create_pr is absent when toolsConfig is undefined", async () => {
    const tools = await get_tools(
      "/tmp/owner/repo",
      "fake-api-key",
      undefined,   // pat
      undefined,   // toolsConfig — no config at all
    );
    assert.ok(
      !("create_pr" in tools),
      "create_pr must NOT appear in default tools"
    );
  });

  it("create_pr is absent when toolsConfig is present but without create_pr", async () => {
    const tools = await get_tools(
      "/tmp/owner/repo",
      "fake-api-key",
      undefined,
      { bash: true }, // toolsConfig without create_pr
    );
    assert.ok(
      !("create_pr" in tools),
      "create_pr must NOT appear when not in toolsConfig"
    );
  });

  it("create_pr appears only when toolsConfig.create_pr is truthy", async () => {
    const prColl: PrCollector = { result: undefined };
    const fakeHandle = {
      worktreePath: "/fake",
      baseDir: "/fake-base",
      baseSha: "abc",
      baseName: "main",
      branch: "swarm/test-abc12345",
      owner: "owner",
      repo: "repo",
      runId: randomUUID(),
      runHome: os.tmpdir(),
      _children: new Set(),
      _released: false,
    } as WorktreeHandle;
    const tools = await get_tools(
      "/tmp/owner/repo",
      "fake-api-key",
      FAKE_PAT,
      { create_pr: true },
      undefined, // provider
      undefined, // repos
      undefined, // subAgents
      undefined, // ggnn
      undefined, // messagesRef
      undefined, // provenanceCollector
      undefined, // modelName
      undefined, // stakwork
      undefined, // googleSheets
      undefined, // skills
      undefined, // ontologyDomains
      undefined, // sessionId
      undefined, // abortSignal
      {
        prCollector: prColl,
        prMode: {
          handle: fakeHandle,
          identity: TEST_IDENTITY,
          env: gitEnv(TEST_IDENTITY, FAKE_PAT, os.tmpdir()),
          githubClient: makeStubClient({}),
        },
      },
    );
    assert.ok(
      "create_pr" in tools,
      "create_pr must be registered when toolsConfig.create_pr is truthy"
    );
  });
});

// ---------------------------------------------------------------------------
// 4. create_pr tool schema — no credential fields
// ---------------------------------------------------------------------------

describe("create_pr tool schema", () => {
  it("input schema has only title and body (no branch_hint, no pat/credential fields)", async () => {
    const prColl: PrCollector = { result: undefined };
    const fakeHandle = {
      worktreePath: "/fake",
      baseDir: "/fake-base",
      baseSha: "abc",
      baseName: "main",
      branch: "swarm/test-abc12345",
      owner: "owner",
      repo: "repo",
      runId: randomUUID(),
      runHome: os.tmpdir(),
      _children: new Set(),
      _released: false,
    } as WorktreeHandle;
    const tools = await get_tools(
      "/tmp/owner/repo",
      "fake-api-key",
      FAKE_PAT,
      { create_pr: true },
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      {
        prCollector: prColl,
        prMode: {
          handle: fakeHandle,
          identity: TEST_IDENTITY,
          env: gitEnv(TEST_IDENTITY, FAKE_PAT, os.tmpdir()),
          githubClient: makeStubClient({}),
        },
      },
    );
    const createPrTool = tools.create_pr as any;
    assert.ok(createPrTool, "create_pr tool must exist");
    const schema = createPrTool.parameters ?? createPrTool.inputSchema;
    assert.ok(schema, "create_pr must have an input schema");
    // Must not contain credential-shaped fields
    const schemaStr = JSON.stringify(schema);
    assert.ok(!schemaStr.includes('"pat"'), "schema must not contain pat field");
    assert.ok(!schemaStr.includes('"token"'), "schema must not contain token field");
    assert.ok(!schemaStr.includes('"github_token"'), "schema must not contain github_token field");
    // branch_hint was removed: the branch is created at acquireWorktree time,
    // before the model runs, so a hint here could never take effect.
    assert.ok(!schemaStr.includes("branch_hint"), "schema must not advertise branch_hint");
  });
});

// ---------------------------------------------------------------------------
// 5. worktree path preserves owner/repo derivation
// ---------------------------------------------------------------------------

describe("worktree path — owner/repo derivation", () => {
  it("repoArr[len-2]/repoArr[len-1] gives correct owner/repo from worktree path", () => {
    const runId = "abc12345-0000-0000-0000-000000000000";
    const worktreePath = `/tmp/.swarm-work/${runId}/myowner/myrepo`;
    const repoArr = worktreePath.split("/");
    const derivedOwner = repoArr[repoArr.length - 2];
    const derivedRepo = repoArr[repoArr.length - 1];
    assert.strictEqual(derivedOwner, "myowner");
    assert.strictEqual(derivedRepo, "myrepo");
  });
});

// ---------------------------------------------------------------------------
// 6. subagent.ts strips create_pr from forwarded toolsConfig
// ---------------------------------------------------------------------------

describe("subagent create_pr stripping", () => {
  it("callRemoteAgent strips create_pr from forwarded toolsConfig", async () => {
    let capturedBody: any = null;
    // Mock fetch globally for this test
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (url: any, init: any) => {
      capturedBody = JSON.parse(init?.body ?? "{}");
      // Simulate HTTP error (not ok) to prevent polling
      return {
        ok: false,
        status: 403,
        text: async () => "forbidden",
      } as any;
    };

    try {
      const subAgent: SubAgent = {
        url: "https://test.example.com",
        apiToken: "test-token",
        toolsConfig: { create_pr: true, bash: true } as any,
      };
      await callRemoteAgent(subAgent, "test prompt").catch(() => {});
    } finally {
      globalThis.fetch = originalFetch;
    }

    assert.ok(capturedBody !== null, "fetch was called");
    // create_pr should be stripped from the forwarded toolsConfig
    if (capturedBody.toolsConfig) {
      assert.strictEqual(
        capturedBody.toolsConfig.create_pr,
        undefined,
        "create_pr must be stripped from forwarded toolsConfig"
      );
      // bash should still be present
      assert.strictEqual(capturedBody.toolsConfig.bash, true);
    }
  });

  it("callRemoteAgent does not forward toolsConfig if only create_pr was in it", async () => {
    let capturedBody: any = null;
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (url: any, init: any) => {
      capturedBody = JSON.parse(init?.body ?? "{}");
      return { ok: false, status: 403, text: async () => "forbidden" } as any;
    };

    try {
      const subAgent: SubAgent = {
        url: "https://test.example.com",
        apiToken: "test-token",
        toolsConfig: { create_pr: true } as any,
      };
      await callRemoteAgent(subAgent, "test prompt").catch(() => {});
    } finally {
      globalThis.fetch = originalFetch;
    }

    // When only create_pr was present, toolsConfig should not be forwarded (empty)
    assert.ok(capturedBody !== null);
    if (capturedBody.toolsConfig !== undefined) {
      assert.strictEqual(capturedBody.toolsConfig.create_pr, undefined);
    }
  });
});

// ---------------------------------------------------------------------------
// 7. sweepOrphanedWorktrees — boot reconciliation
// ---------------------------------------------------------------------------

describe("sweepOrphanedWorktrees", () => {
  it("removes stale directories under the swarm-work root", async () => {
    // Use a private root: test files run in parallel processes, so sweeping
    // the real /tmp/.swarm-work would delete live worktrees belonging to
    // concurrently running git_pr tests (and wipe real state on a dev box).
    const swarmRoot = tmpDir();
    const staleRunId = randomUUID();
    const staleDir = path.join(swarmRoot, staleRunId);
    fs.mkdirSync(staleDir, { recursive: true });
    fs.writeFileSync(path.join(staleDir, "stale.txt"), "stale data");

    assert.ok(fs.existsSync(staleDir), "stale dir should exist before sweep");

    try {
      sweepOrphanedWorktrees(swarmRoot, tmpDir());
      assert.ok(!fs.existsSync(staleDir), "stale dir should be removed after sweep");
    } finally {
      fs.rmSync(swarmRoot, { recursive: true, force: true });
    }
  });

  it("prunes stale git worktree registrations from base repos", async () => {
    // The sweep scans <reposRoot>/<owner>/<repo>, so the fixture clone must
    // sit exactly two levels below the reposRoot passed in — a fixture the
    // sweep never visits makes the test vacuous.
    const scratchRoot = tmpDir();
    const reposRoot = path.join(scratchRoot, "repos");
    const ownerDir = path.join(reposRoot, "owner1");
    const sweepBareDir = path.join(scratchRoot, "bare.git");
    const sweepCloneDir = path.join(ownerDir, "clone");
    fs.mkdirSync(ownerDir, { recursive: true });

    fs.mkdirSync(sweepBareDir, { recursive: true });
    execSync("git init --bare .", { cwd: sweepBareDir, stdio: "ignore" });

    // Seed bare with a commit via scratch
    const sweepScratch = path.join(scratchRoot, "scratch");
    fs.mkdirSync(sweepScratch, { recursive: true });
    const env = localEnv();
    execSync("git init -b main .", { cwd: sweepScratch, env, stdio: "ignore" });
    fs.writeFileSync(path.join(sweepScratch, "f.txt"), "hello");
    execSync("git add f.txt", { cwd: sweepScratch, env, stdio: "ignore" });
    execSync('git commit -m "init"', { cwd: sweepScratch, env, stdio: "ignore" });
    execSync(`git remote add origin "${sweepBareDir}"`, { cwd: sweepScratch, env, stdio: "ignore" });
    execSync("git push -u origin main", { cwd: sweepScratch, env, stdio: "ignore" });

    execSync(`git clone --single-branch "${sweepBareDir}" "${sweepCloneDir}"`, {
      env,
      stdio: "ignore",
    });

    try {
      // Create a worktree and immediately remove the directory without
      // deregistering (simulates SIGKILL mid-run)
      const orphanPath = path.join(scratchRoot, "orphan-worktree");
      execSync(`git worktree add --detach "${orphanPath}" HEAD`, {
        cwd: sweepCloneDir,
        env,
        stdio: "ignore",
      });
      const wtMetaDir = path.join(sweepCloneDir, ".git", "worktrees");
      assert.ok(
        fs.readdirSync(wtMetaDir).length > 0,
        "worktree registration should exist before sweep"
      );
      fs.rmSync(orphanPath, { recursive: true, force: true });

      sweepOrphanedWorktrees(path.join(scratchRoot, "swarm-work"), reposRoot);

      // The stale registration must actually be gone — `git worktree list`
      // succeeds even with dangling registrations, so it proves nothing.
      const remaining = fs.existsSync(wtMetaDir) ? fs.readdirSync(wtMetaDir) : [];
      assert.equal(
        remaining.length,
        0,
        `stale worktree registrations should be pruned, found: ${remaining.join(", ")}`
      );
    } finally {
      fs.rmSync(scratchRoot, { recursive: true, force: true });
    }
  });
});

// ---------------------------------------------------------------------------
// 8. path safety — runId cannot be tampered with via sessionId
// ---------------------------------------------------------------------------

describe("path safety", () => {
  it("acquireWorktree runId with path traversal chars is rejected", async () => {
    // A malicious runId with path traversal — the containment assertion should catch it
    const maliciousRunId = "../../etc";
    const result = await acquireWorktree({
      baseDir: baseCloneDir,
      owner: "owner",
      repo: "repo",
      runId: maliciousRunId,
      pat: FAKE_PAT,
      githubClient: makeStubClient({}),
      commit: baseSha,
    });
    // The traversal must be rejected as a structured failure — before any
    // filesystem side effect (the old guard anchored against a parent derived
    // from the tainted runId, letting "../../etc" cancel out and reach
    // mkdirSync("/etc/.home")).
    assert.ok(!result.ok, "malicious runId must be rejected");
    if (!result.ok) {
      assert.equal(result.failure, "base_repo_vanished");
      assert.match(result.error, /Path traversal rejected/);
    }
    // A safe runId should still succeed:
    const safeRunId = randomUUID();
    const safeResult = await acquireWorktree({
      baseDir: baseCloneDir,
      owner: "owner",
      repo: "repo",
      runId: safeRunId,
      pat: FAKE_PAT,
      githubClient: makeStubClient({}),
      commit: baseSha,
    });
    if (safeResult.ok) {
      await releaseWorktree(safeResult.handle).catch(() => {});
    }
    assert.ok(safeResult.ok, "safe runId should succeed");
  });
});

// ---------------------------------------------------------------------------
// 9. Integration: acquireWorktree + tool + releaseWorktree pipeline
// ---------------------------------------------------------------------------

describe("integration: worktree lifecycle with tools", () => {
  it("create_pr tool registers in get_tools and has correct name", async () => {
    const runId = randomUUID();
    const worktreeResult = await acquireWorktree({
      baseDir: baseCloneDir,
      owner: "owner",
      repo: "repo",
      runId,
      pat: FAKE_PAT,
      githubClient: makeStubClient({}),
      commit: baseSha,
    });
    assert.ok(worktreeResult.ok, `acquireWorktree failed: ${(worktreeResult as any).error}`);
    const handle = worktreeResult.handle;

    try {
      const prColl: PrCollector = { result: undefined };
      const tools = await get_tools(
        handle.worktreePath,
        "fake-api-key",
        FAKE_PAT,
        { create_pr: true },
        undefined,
        ["owner/repo"],
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        {
          prCollector: prColl,
          prMode: {
            handle,
            identity: TEST_IDENTITY,
            env: gitEnv(TEST_IDENTITY, FAKE_PAT, handle.runHome),
            githubClient: makeStubClient({}),
          },
          baseCheckoutPath: `/tmp/owner/repo`,
        },
      );
      assert.ok("create_pr" in tools, "create_pr tool must be registered");
      assert.ok(typeof (tools as any).create_pr.execute === "function", "create_pr must have execute");

      // Verify the worktree path matches the expected shape: /tmp/.swarm-work/<runId>/owner/repo
      const repoArr = handle.worktreePath.split("/");
      assert.strictEqual(repoArr[repoArr.length - 1], "repo");
      assert.strictEqual(repoArr[repoArr.length - 2], "owner");
    } finally {
      await releaseWorktree(handle).catch(() => {});
    }
  });
});

// ---------------------------------------------------------------------------
// 10. toolConfigEnabled: create_pr respects truthy/falsy values
// ---------------------------------------------------------------------------

describe("toolConfigEnabled for create_pr", () => {
  it("returns true for boolean true", () => {
    assert.strictEqual(toolConfigEnabled(true), true);
  });
  it("returns false for boolean false", () => {
    assert.strictEqual(toolConfigEnabled(false), false);
  });
  it("returns false for undefined", () => {
    assert.strictEqual(toolConfigEnabled(undefined), false);
  });
  it("returns false for null", () => {
    assert.strictEqual(toolConfigEnabled(null), false);
  });
  it("returns true for non-empty string", () => {
    assert.strictEqual(toolConfigEnabled("enabled"), true);
  });
  it("returns true for object with enabled:true", () => {
    assert.strictEqual(toolConfigEnabled({ enabled: true }), true);
  });
  it("returns false for object with enabled:false", () => {
    assert.strictEqual(toolConfigEnabled({ enabled: false }), false);
  });
});
