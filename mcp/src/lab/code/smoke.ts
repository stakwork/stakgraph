/**
 * code OFFLINE smoke — no LLM, no network. Seeds `code-change-propose` and
 * `code-change-land` into a throwaway workspace, checks the steps they lean
 * on are discoverable, static-validates both through the authoring
 * capability (what meta/validate-workflow runs), then RUNS them end to end
 * against a local BARE git origin:
 *
 *  - propose, with the `agent` step swapped for a fake that edits a file —
 *    the wiring (templates, git/checkout → agent cwd → git/diff → pack,
 *    worktree cleanup) proven without a model;
 *  - land, with that run's diff: checkout → apply → commit → push are the
 *    REAL steps over real git into the bare (no GITHUB_TOKEN in the secrets,
 *    so git/push uses its fallback author and the `file://` origin needs no
 *    credential); only `github/create-pr` is a fake that records what it was
 *    asked to open. Then land again after main moved under the diff, for the
 *    `patch_conflict:` contract — nothing pushed, no pull request.
 *
 *   npx tsx src/lab/code/smoke.ts
 *
 * With CODE_SMOKE_LIVE=1 (and a provider key in env) it also runs the REAL
 * propose workflow against CODE_SMOKE_REPO (default: stakwork/strut) with a
 * trivial prompt and prints the diff. Land is never run live from here — it
 * pushes a branch and opens a pull request on the repository.
 */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";
import { WorkspaceManager, buildRegistry, createStrut, defineStep, runWorkflow, standardServices } from "strut";
import { seedCodeWorkflows } from "./seed.js";

const PROPOSE = "code-change-propose";
const LAND = "code-change-land";
const STEPS = ["git/checkout", "git/diff", "git/apply", "git/push", "github/create-pr", "agent", "pack"];

const sha256 = (s: string) => createHash("sha256").update(s, "utf8").digest("hex");

function git(args: string[], cwd: string): string {
  const r = spawnSync("git", ["-c", "user.name=smoke", "-c", "user.email=smoke@example.com", ...args], { cwd, encoding: "utf8" });
  if (r.status !== 0) throw new Error(`git ${args.join(" ")}: ${r.stderr}`);
  return r.stdout.trim();
}

/** Stands in for the core `agent` step: "makes the change" by editing files
 *  in cwd, returns the agent step's output shape. */
const fakeAgent = defineStep({
  type: "agent",
  input: z.object({ cwd: z.string(), prompt: z.string() }).passthrough(),
  output: z.any(),
  async run(cfg) {
    writeFileSync(join(cfg.cwd, "README.md"), "# smoke\n\nchanged by the fake agent\n");
    writeFileSync(join(cfg.cwd, "hello.txt"), `${cfg.prompt}\n`);
    return { result: "Edited README.md and added hello.txt.", steps: 2, usage: { input: 1, output: 1 }, cost: 0 };
  },
});

interface PrCall {
  repo: string;
  head: string;
  base: string;
  title: string;
  body?: string;
}

/** Stands in for `github/create-pr` — GitHub's REST API is the one thing the
 *  smoke cannot reach. Records what it was asked to open and answers with the
 *  step's output shape; `headSha` is read off the bare origin, so it also
 *  proves the branch was there when the PR was "opened", as GitHub needs. */
function fakeCreatePr(bare: string, calls: PrCall[]) {
  return defineStep({
    type: "github/create-pr",
    input: z.object({ repo: z.string(), head: z.string(), base: z.string(), title: z.string(), body: z.string().optional() }).passthrough(),
    output: z.any(),
    async run(cfg) {
      calls.push({ repo: cfg.repo, head: cfg.head, base: cfg.base, title: cfg.title, ...(cfg.body !== undefined ? { body: cfg.body } : {}) });
      const n = calls.length;
      return { url: `https://github.com/smoke/repo/pull/${n}`, number: n, headSha: git(["rev-parse", `refs/heads/${cfg.head}`], bare), base: cfg.base, head: cfg.head, created: true };
    },
  });
}

async function main() {
  const dir = mkdtempSync(join(process.cwd(), "code-smoke-"));
  try {
    // ── 1. seed + discover ───────────────────────────────────────────────
    const workspace = new WorkspaceManager(dir);
    await seedCodeWorkflows(workspace);
    const { registry } = await buildRegistry(await workspace.materializeCustomSteps());
    for (const t of STEPS) assert.ok(registry[t], `registry missing ${t}`);
    const listed = await workspace.listWorkflows();
    const entries = Object.fromEntries(
      [PROPOSE, LAND].map((name) => {
        const entry = listed.find((w) => w.name === name);
        assert.ok(entry, `${name} not seeded`);
        assert.equal(entry!.category, "code");
        return [name, entry!];
      }),
    );
    console.log(`✔ seeded ${PROPOSE} + ${LAND} (category code); ${STEPS.join(", ")} discoverable`);

    // ── 2. static validation (what meta/validate-workflow runs) ──────────
    const strut = await createStrut({ workspace, serveUi: false, enableChat: false });
    const authoring = (strut.services as any).authoring;
    for (const name of [PROPOSE, LAND]) {
      const yaml = await workspace.getWorkflowSource(name, entries[name]!.activeVersion);
      const v = await authoring.validateWorkflow(yaml, name);
      assert.equal(v.ok, true, `${name}: ${JSON.stringify(v.errors, null, 2)}`);
      if (v.warnings.length) console.log(`  warnings:`, v.warnings.map((w: any) => `${w.path}: ${w.message}`));
      console.log(`✔ ${name} validates (${v.summary.steps} steps)`);
    }

    // ── 3. a local origin: a bare repo (what a push needs), seeded from a
    //       working copy that later moves main under the diff ──────────────
    const bare = join(dir, "origin.git");
    const seed = join(dir, "seed");
    mkdirSync(seed);
    git(["init", "-q", "-b", "main"], seed);
    writeFileSync(join(seed, "README.md"), "# smoke\n");
    git(["add", "-A"], seed);
    git(["commit", "-q", "-m", "init"], seed);
    git(["init", "-q", "--bare", "-b", "main", bare], dir);
    git(["push", "-q", bare, "main"], seed);
    const repo = `file://${bare}`;
    const dataDir = join(dir, "data");
    // No GITHUB_TOKEN: git/push's fallback author, a remote that needs no credential.
    const services = standardServices({ secretsSource: {}, dataDir });

    // ── 4. propose, offline: the fake agent ──────────────────────────────
    const propose = await workspace.getWorkflow(PROPOSE);
    const res = await runWorkflow(propose, { repo, prompt: "say hello" }, { ...registry, agent: fakeAgent } as typeof registry, { services });
    assert.equal(res.status, "success", JSON.stringify(res.error));
    const out = res.output as Record<string, unknown>;
    assert.deepEqual(out["files"], ["README.md", "hello.txt"]);
    assert.equal(out["filesChanged"], 2);
    const diff = out["diff"] as string;
    assert.match(diff, /\+\+\+ b\/hello\.txt/);
    assert.match(diff, /\+say hello/);
    assert.equal(out["baseBranch"], "main");
    assert.equal(out["baseSha"], git(["rev-parse", "main"], bare));
    assert.equal(out["summary"], "Edited README.md and added hello.txt.");
    const diffSha256 = sha256(diff);
    assert.equal(out["diffSha256"], diffSha256, "git/diff's sha256 is the sha256 of the diff bytes");
    assert.ok(!existsSync(join(dataDir, "worktrees", res.runId)), "worktree removed at run end");
    console.log(`✔ ${PROPOSE} ran end to end offline: ${out["filesChanged"]} files, diff ${diff.length} bytes, worktree cleaned up`);

    // ── 5. land, offline: REAL checkout → apply → push into the bare; a fake
    //       create-pr records what it was asked to open ───────────────────
    const prCalls: PrCall[] = [];
    const landRegistry = { ...registry, "github/create-pr": fakeCreatePr(bare, prCalls) } as typeof registry;
    const land = await workspace.getWorkflow(LAND);
    const input = { repo, baseBranch: "main", diff, diffSha256, branch: "smoke/land-000001", title: "smoke: say hello", body: "landed by the smoke" };
    const landed = await runWorkflow(land, input, landRegistry, { services });
    assert.equal(landed.status, "success", JSON.stringify(landed.error));
    const lo = landed.output as Record<string, unknown>;
    const headSha = lo["headSha"] as string;
    const mainSha = git(["rev-parse", "main"], bare);
    assert.equal(git(["rev-parse", "refs/heads/smoke/land-000001"], bare), headSha, "the branch is on the origin at headSha");
    assert.equal(git(["rev-parse", `${headSha}^`], bare), mainSha, "one commit on top of the base");
    assert.equal(git(["rev-parse", "main"], bare), out["baseSha"], "main untouched");
    assert.equal(git(["diff", "--name-only", mainSha, headSha], bare), "README.md\nhello.txt");
    assert.equal(git(["show", `${headSha}:README.md`], bare), "# smoke\n\nchanged by the fake agent");
    assert.equal(git(["show", `${headSha}:hello.txt`], bare), "say hello");
    assert.equal(git(["log", "-1", "--format=%s", headSha], bare), "smoke: say hello");
    assert.equal(git(["log", "-1", "--format=%an <%ae>", headSha], bare), "strut <strut@users.noreply.github.com>", "no token → the fallback author");
    assert.deepEqual(lo["files"], ["README.md", "hello.txt"]);
    assert.equal(lo["filesChanged"], 2);
    assert.equal(lo["diffSha256"], diffSha256, "the hash of the bytes applied");
    assert.equal(lo["branch"], "smoke/land-000001");
    assert.equal(lo["base"], "main");
    assert.equal(lo["url"], "https://github.com/smoke/repo/pull/1");
    assert.equal(lo["number"], 1);
    assert.deepEqual(prCalls, [{ repo, head: "smoke/land-000001", base: "main", title: "smoke: say hello", body: "landed by the smoke" }]);
    assert.ok(!existsSync(join(dataDir, "worktrees", landed.runId)), "worktree removed at run end");
    console.log(`✔ ${LAND} ran end to end offline: pushed ${headSha.slice(0, 8)} to ${lo["branch"]} on the bare origin, PR asked for on ${lo["base"]}, worktree cleaned up`);

    // ── 6. main moves under the diff → patch_conflict:, nothing pushed, no PR
    writeFileSync(join(seed, "README.md"), "# smoke, moved on main\n");
    git(["commit", "-q", "-am", "main moved"], seed);
    git(["push", "-q", bare, "main"], seed);
    const stale = await runWorkflow(land, { ...input, branch: "smoke/land-000002" }, landRegistry, { services });
    assert.equal(stale.status, "error", JSON.stringify(stale.output));
    assert.match(stale.error!.message, /^patch_conflict: /);
    assert.throws(() => git(["rev-parse", "--verify", "refs/heads/smoke/land-000002"], bare), "nothing was pushed");
    assert.equal(prCalls.length, 1, "no pull request asked for");
    assert.ok(!existsSync(join(dataDir, "worktrees", stale.runId)), "worktree removed at run end");
    console.log(`✔ ${LAND} on a moved base: "${stale.error!.message.split(" — ")[0]}" — nothing pushed, no pull request`);

    // ── 7. optional: the real propose ────────────────────────────────────
    if (process.env["CODE_SMOKE_LIVE"] === "1") {
      const liveRepo = process.env["CODE_SMOKE_REPO"] ?? "https://github.com/stakwork/strut";
      const t0 = Date.now();
      const live = await runWorkflow(
        propose,
        { repo: liveRepo, prompt: "Add a single line 'Smoke test.' to the end of README.md. Change nothing else." },
        registry,
        { services: strut.services as any },
      );
      console.log(`live run ${live.status} in ${Date.now() - t0} ms`);
      if (live.status === "success") {
        const o = live.output as Record<string, unknown>;
        console.log(`  filesChanged ${o["filesChanged"]} files ${JSON.stringify(o["files"])} cost $${o["cost"]}`);
        console.log(`  summary: ${o["summary"]}`);
        console.log((o["diff"] as string).slice(0, 2000));
      } else console.log("  error:", live.error?.message);
    }
    console.log("\ncode smoke: all good");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

main().catch((e) => {
  console.error("code smoke FAILED:", e);
  process.exit(1);
});
