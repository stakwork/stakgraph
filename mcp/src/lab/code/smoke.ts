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
 *    `patch_conflict:` contract — nothing pushed, no pull request;
 *  - code-change-pr, three turns: the fake agent again, with the REAL
 *    checkout → diff → push into the bare and the fake create-pr — a first
 *    turn that makes the branch (`params.branch_prefix` + run id) and the
 *    PR; a follow-up on that `branch` that checks it out, adds one commit
 *    and gets the SAME PR back (`created: false`); a turn whose agent
 *    changes nothing, which fails at git/push with nothing pushed.
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
const PR = "code-change-pr";
const STEPS = ["git/checkout", "git/diff", "git/apply", "git/push", "github/create-pr", "job/read", "agent", "pack"];

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

/** An agent that decides the task needs no change — edits nothing. */
const noopAgent = defineStep({
  type: "agent",
  input: z.object({ cwd: z.string(), prompt: z.string() }).passthrough(),
  output: z.any(),
  async run() {
    return { result: "No change needed.", steps: 1, usage: { input: 1, output: 1 }, cost: 0 };
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
 *  proves the branch was there when the PR was "opened", as GitHub needs.
 *  Idempotent by head like the real step: the same head again returns the
 *  pull request already "opened" for it, `created: false`. */
function fakeCreatePr(bare: string, calls: PrCall[]) {
  const open = new Map<string, number>();
  return defineStep({
    type: "github/create-pr",
    input: z.object({ repo: z.string(), head: z.string(), base: z.string(), title: z.string(), body: z.string().optional() }).passthrough(),
    output: z.any(),
    async run(cfg) {
      calls.push({ repo: cfg.repo, head: cfg.head, base: cfg.base, title: cfg.title, ...(cfg.body !== undefined ? { body: cfg.body } : {}) });
      const existing = open.get(cfg.head);
      const n = existing ?? open.size + 1;
      if (existing === undefined) open.set(cfg.head, n);
      return { url: `https://github.com/smoke/repo/pull/${n}`, number: n, headSha: git(["rev-parse", `refs/heads/${cfg.head}`], bare), base: cfg.base, head: cfg.head, created: existing === undefined };
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
      [PROPOSE, LAND, PR].map((name) => {
        const entry = listed.find((w) => w.name === name);
        assert.ok(entry, `${name} not seeded`);
        assert.equal(entry!.category, "code");
        return [name, entry!];
      }),
    );
    console.log(`✔ seeded ${PROPOSE} + ${LAND} + ${PR} (category code); ${STEPS.join(", ")} discoverable`);

    // ── 2. static validation (what meta/validate-workflow runs) ──────────
    const strut = await createStrut({ workspace, serveUi: false, enableChat: false });
    const authoring = (strut.services as any).authoring;
    for (const name of [PROPOSE, LAND, PR]) {
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

    // ── 7. code-change-pr: task → pull request in one run, then a follow-up
    //       on the same branch, then an agent that changes nothing ────────
    const pr = await workspace.getWorkflow(PR);
    // The land section's fake create-pr, so this origin already "has" PR #1.
    const prRegistry = { ...landRegistry, agent: fakeAgent } as typeof registry;
    const prInput = { repo, prompt: "say hello from the pr workflow", title: "smoke: hello via pr" };
    const first = await runWorkflow(pr, prInput, prRegistry, { services });
    assert.equal(first.status, "success", JSON.stringify(first.error));
    const po = first.output as Record<string, unknown>;
    const mainNow = git(["rev-parse", "main"], bare);
    const branch = po["branch"] as string;
    assert.equal(branch, `strut/${first.runId}`, "first turn: params.branch_prefix + the run id");
    const sha1 = po["headSha"] as string;
    assert.equal(git(["rev-parse", `refs/heads/${branch}`], bare), sha1, "the branch is on the origin at headSha");
    assert.equal(git(["rev-parse", `${sha1}^`], bare), mainNow, "one commit on top of the default branch");
    assert.equal(git(["log", "-1", "--format=%s", sha1], bare), "smoke: hello via pr", "the title is the commit's first line");
    assert.equal(git(["log", "-1", "--format=%b", sha1], bare), "Edited README.md and added hello.txt.", "the agent's summary is the commit body");
    assert.equal(git(["show", `${sha1}:hello.txt`], bare), "say hello from the pr workflow");
    assert.equal(po["base"], "main");
    assert.equal(po["created"], true);
    assert.equal(po["number"], 2, "the second pull request this origin has seen");
    assert.equal(po["url"], "https://github.com/smoke/repo/pull/2");
    assert.ok((po["repo"] as string).endsWith("/origin"), `repo is owner/name, got ${po["repo"]}`);
    // main moved in step 6, so the fake's README differs from it again.
    assert.equal(po["filesChanged"], 2);
    assert.deepEqual(po["files"], ["README.md", "hello.txt"]);
    assert.equal(po["summary"], "Edited README.md and added hello.txt.");
    assert.deepEqual(prCalls.at(-1), { repo, head: branch, base: "main", title: "smoke: hello via pr", body: "Edited README.md and added hello.txt." });
    assert.ok(!existsSync(join(dataDir, "worktrees", first.runId)), "worktree removed at run end");
    console.log(`✔ ${PR} turn 1: pushed ${sha1.slice(0, 8)} to ${branch}, PR #${po["number"]} opened on ${po["base"]}`);

    // A follow-up: the same branch, one more commit, the SAME pull request.
    const again = await runWorkflow(pr, { ...prInput, branch, prompt: "say hello again" }, prRegistry, { services });
    assert.equal(again.status, "success", JSON.stringify(again.error));
    const ao = again.output as Record<string, unknown>;
    const sha2 = ao["headSha"] as string;
    assert.equal(ao["branch"], branch);
    assert.equal(git(["rev-parse", `refs/heads/${branch}`], bare), sha2, "the branch advanced");
    assert.equal(git(["rev-parse", `${sha2}^`], bare), sha1, "…by one commit on top of the first turn's");
    assert.equal(git(["show", `${sha2}:hello.txt`], bare), "say hello again");
    assert.equal(ao["filesChanged"], 1, "README.md was already the fake's on the branch; hello.txt changed");
    assert.equal(ao["base"], "main", "the base is still the default branch, not the branch checked out");
    assert.equal(ao["created"], false, "the open pull request for the branch is reused");
    assert.equal(ao["number"], po["number"]);
    assert.equal(ao["url"], po["url"]);
    assert.deepEqual(prCalls.at(-1), { repo, head: branch, base: "main", title: "smoke: hello via pr", body: "Edited README.md and added hello.txt." });
    console.log(`✔ ${PR} turn 2 (branch: ${branch}): ${sha2.slice(0, 8)} on top of ${sha1.slice(0, 8)}, the same PR #${ao["number"]} (created: false)`);

    // Notes: under a job, each path is read from the job's directory
    // (job/read — faked here: the index is the server's) and handed to the
    // agent whole, under the task. A one-shot has no job directory: it
    // ignores them and the agent gets the task alone.
    const reads: Array<{ job: string; path: string }> = [];
    const fakeJobRead = defineStep({
      type: "job/read",
      input: z.object({ job: z.string(), path: z.string() }).passthrough(),
      output: z.any(),
      async run(cfg) {
        reads.push({ job: cfg.job, path: cfg.path });
        return { path: cfg.path, kind: "markdown", text: `What was found, in ${cfg.path}.` };
      },
    });
    const prompts: string[] = [];
    const promptAgent = defineStep({
      type: "agent",
      input: z.object({ cwd: z.string(), prompt: z.string() }).passthrough(),
      output: z.any(),
      async run(cfg) {
        prompts.push(cfg.prompt);
        writeFileSync(join(cfg.cwd, "noted.txt"), `turn ${prompts.length}\n`);
        return { result: "Changed what the notes named.", steps: 1, usage: { input: 1, output: 1 }, cost: 0 };
      },
    });
    const notesRegistry = { ...prRegistry, agent: promptAgent, "job/read": fakeJobRead } as typeof registry;
    const noted = await runWorkflow(pr, { ...prInput, title: "smoke: notes", notes: ["notes/auth.md", "notes/tests.md"] }, notesRegistry, { services, job: "job-notes" });
    assert.equal(noted.status, "success", JSON.stringify(noted.error));
    assert.deepEqual(reads, [{ job: "job-notes", path: "notes/auth.md" }, { job: "job-notes", path: "notes/tests.md" }], "each note read from the job's directory, in order");
    assert.equal(
      prompts.at(-1),
      `${prInput.prompt}\n\nNotes from an earlier look at this code, kept in the job's directory. Start from what they name — files, conventions, checks — and verify as you go rather than surveying the repository again.` +
        `\n\n--- notes/auth.md ---\n\nWhat was found, in notes/auth.md.` +
        `\n\n--- notes/tests.md ---\n\nWhat was found, in notes/tests.md.`,
      "the task, then each note whole under it",
    );
    const oneShot = await runWorkflow(pr, { ...prInput, title: "smoke: notes, no job", notes: ["notes/auth.md"] }, notesRegistry, { services });
    assert.equal(oneShot.status, "success", JSON.stringify(oneShot.error));
    assert.equal(reads.length, 2, "a one-shot reads no notes");
    assert.equal(prompts.at(-1), prInput.prompt, "a one-shot's agent gets the task alone");
    console.log(`✔ ${PR} with notes: under a job each is read (job/read) and handed whole under the task; a one-shot ignores them`);

    // An agent that made no change: nothing to push, no pull request.
    const calls = prCalls.length;
    const none = await runWorkflow(pr, { ...prInput, title: "smoke: nothing" }, { ...prRegistry, agent: noopAgent } as typeof registry, { services });
    assert.equal(none.status, "error", JSON.stringify(none.output));
    assert.match(none.error!.message, /nothing is staged/);
    assert.throws(() => git(["rev-parse", "--verify", `refs/heads/strut/${none.runId}`], bare), "nothing was pushed");
    assert.equal(prCalls.length, calls, "no pull request asked for");
    assert.ok(!existsSync(join(dataDir, "worktrees", none.runId)), "worktree removed at run end");
    console.log(`✔ ${PR} with no change: "${none.error!.message.split(" — ")[0]}" — nothing pushed, no pull request`);

    // ── 8. optional: the real propose ────────────────────────────────────
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
