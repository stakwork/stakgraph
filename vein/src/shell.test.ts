import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { runShell, runCmd, minimalEnv } from "./shell.js";
import { buildTools } from "./ai/tools.js";
import type { AiDeps } from "./ai/prompts.js";

let dir: string;
before(async () => {
  dir = await mkdtemp(join(tmpdir(), "vein-shell-"));
});
after(async () => {
  await rm(dir, { recursive: true, force: true });
});

// ── shell helpers ───────────────────────────────────────────────────────────

describe("shell helpers", () => {
  it("runShell executes in the given cwd", async () => {
    const out = await runShell("pwd", dir);
    // macOS tmpdir may resolve through /private — compare the tail.
    assert.ok(out.trim().endsWith(dir.replace(/^\/private/, "")) || out.trim() === dir);
  });

  it("scrubs the env: allowlisted vars pass, secrets don't", async () => {
    process.env.VEIN_TEST_FAKE_KEY = "sk-super-secret";
    try {
      const out = await runShell("env", dir);
      assert.doesNotMatch(out, /VEIN_TEST_FAKE_KEY|sk-super-secret/);
      assert.match(out, /(^|\n)PATH=/);
    } finally {
      delete process.env.VEIN_TEST_FAKE_KEY;
    }
  });

  it("minimalEnv contains only allowlisted keys", () => {
    process.env.VEIN_TEST_FAKE_KEY = "x";
    try {
      const env = minimalEnv();
      assert.equal(env.VEIN_TEST_FAKE_KEY, undefined);
      assert.ok(env.PATH);
    } finally {
      delete process.env.VEIN_TEST_FAKE_KEY;
    }
  });

  it("caps oversized output", async () => {
    const out = await runShell("yes x | head -c 5000", dir, 10_000, 100);
    assert.ok(out.length < 200);
    assert.match(out, /truncated/);
  });

  it("times out a hung command", async () => {
    await assert.rejects(() => runShell("sleep 5", dir, 300), /timed out/);
  });

  it("rejects on failure with stderr", async () => {
    await assert.rejects(() => runShell("ls /definitely/not/a/path", dir), /Command failed/);
  });

  it("runCmd passes args without shell interpolation", async () => {
    const out = await runCmd("echo", ["$HOME && rm -rf /"], dir);
    assert.equal(out.trim(), "$HOME && rm -rf /");
  });
});

// ── chat builder bash tool ──────────────────────────────────────────────────

/** buildTools only touches deps lazily inside execute(); a stub is enough. */
const stubDeps = (over: Partial<AiDeps> = {}): AiDeps =>
  ({ workspace: {}, registry: {}, store: {}, getRegistry: async () => ({}), ...over }) as AiDeps;

describe("chat bash tool", () => {
  it("is absent without deps.shell, present with it", () => {
    assert.equal("bash" in buildTools(stubDeps()), false);
    assert.equal("bash" in buildTools(stubDeps({ shell: { cwd: dir } })), true);
  });

  it("web_search is gated on deps.webSearch", () => {
    assert.equal("web_search" in buildTools(stubDeps()), false);
    assert.equal("web_search" in buildTools(stubDeps({ webSearch: true })), true);
  });

  it("executes in the workspace dir and creates scratch/", async () => {
    const tools = buildTools(stubDeps({ shell: { cwd: dir } })) as Record<
      string,
      { execute: (a: { command: string; timeoutMs: number }) => Promise<{ output?: string; error?: string }> }
    >;
    const res = await tools.bash.execute({ command: "echo hello-from-bash", timeoutMs: 10_000 });
    assert.equal(res.output?.trim(), "hello-from-bash");
    assert.ok((await readdir(dir)).includes("scratch"));
  });

  it("returns errors as data, not throws", async () => {
    const tools = buildTools(stubDeps({ shell: { cwd: dir } })) as Record<
      string,
      { execute: (a: { command: string; timeoutMs: number }) => Promise<{ output?: string; error?: string }> }
    >;
    const res = await tools.bash.execute({ command: "exit 7", timeoutMs: 10_000 });
    assert.match(res.error ?? "", /Command failed \(7\)/);
  });

  it("does not leak server env to commands", async () => {
    process.env.VEIN_TEST_FAKE_KEY = "sk-super-secret";
    try {
      const tools = buildTools(stubDeps({ shell: { cwd: dir } })) as Record<
        string,
        { execute: (a: { command: string; timeoutMs: number }) => Promise<{ output?: string; error?: string }> }
      >;
      const res = await tools.bash.execute({ command: "env", timeoutMs: 10_000 });
      assert.doesNotMatch(res.output ?? "", /sk-super-secret/);
    } finally {
      delete process.env.VEIN_TEST_FAKE_KEY;
    }
  });
});
