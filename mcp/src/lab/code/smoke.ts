/**
 * code OFFLINE smoke — no LLM, no network. Seeds `code-change-propose` into a
 * throwaway workspace, checks the steps it leans on are discoverable,
 * static-validates it through the authoring capability (what
 * meta/validate-workflow runs), then RUNS it end to end against a local git
 * origin with the `agent` step swapped for a fake that edits a file — so the
 * wiring (templates, git/checkout → agent cwd → git/diff → pack, worktree
 * cleanup) is proven without a model.
 *
 *   npx tsx src/lab/code/smoke.ts
 *
 * With CODE_SMOKE_LIVE=1 (and a provider key in env) it also runs the REAL
 * workflow against CODE_SMOKE_REPO (default: stakwork/strut) with a trivial
 * prompt and prints the diff.
 */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";
import { WorkspaceManager, buildRegistry, createStrut, defineStep, runWorkflow, standardServices } from "strut";
import { seedCodeWorkflows } from "./seed.js";

const NAME = "code-change-propose";

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

async function main() {
  const dir = mkdtempSync(join(process.cwd(), "code-smoke-"));
  try {
    // ── 1. seed + discover ───────────────────────────────────────────────
    const workspace = new WorkspaceManager(dir);
    await seedCodeWorkflows(workspace);
    const { registry } = await buildRegistry(await workspace.materializeCustomSteps());
    for (const t of ["git/checkout", "git/diff", "agent", "pack"]) assert.ok(registry[t], `registry missing ${t}`);
    const entry = (await workspace.listWorkflows()).find((w) => w.name === NAME);
    assert.ok(entry, `${NAME} not seeded`);
    assert.equal(entry!.category, "code");
    console.log(`✔ seeded ${NAME} (category ${entry!.category}); git/checkout, git/diff, agent, pack discoverable`);

    // ── 2. static validation (what meta/validate-workflow runs) ──────────
    const strut = await createStrut({ workspace, serveUi: false, enableChat: false });
    const authoring = (strut.services as any).authoring;
    const yaml = await workspace.getWorkflowSource(NAME, entry!.activeVersion);
    const v = await authoring.validateWorkflow(yaml, NAME);
    assert.equal(v.ok, true, `${NAME}: ${JSON.stringify(v.errors, null, 2)}`);
    if (v.warnings.length) console.log(`  warnings:`, v.warnings.map((w: any) => `${w.path}: ${w.message}`));
    console.log(`✔ ${NAME} validates (${v.summary.steps} steps)`);

    // ── 3. run it, offline: a local origin + the fake agent ──────────────
    const origin = join(dir, "origin");
    git(["init", "-q", "-b", "main", origin], dir);
    writeFileSync(join(origin, "README.md"), "# smoke\n");
    git(["add", "-A"], origin);
    git(["commit", "-q", "-m", "init"], origin);
    const dataDir = join(dir, "data");
    const flow = await workspace.getWorkflow(NAME);
    const res = await runWorkflow(
      flow,
      { repo: `file://${origin}`, prompt: "say hello" },
      { ...registry, agent: fakeAgent } as typeof registry,
      { services: standardServices({ secretsSource: {}, dataDir }) },
    );
    assert.equal(res.status, "success", JSON.stringify(res.error));
    const out = res.output as Record<string, unknown>;
    assert.deepEqual(out["files"], ["README.md", "hello.txt"]);
    assert.equal(out["filesChanged"], 2);
    assert.match(out["diff"] as string, /\+\+\+ b\/hello\.txt/);
    assert.match(out["diff"] as string, /\+say hello/);
    assert.equal(out["baseBranch"], "main");
    assert.equal(out["baseSha"], git(["rev-parse", "main"], origin));
    assert.equal(out["summary"], "Edited README.md and added hello.txt.");
    assert.equal(typeof out["diffSha256"], "string");
    assert.ok(!existsSync(join(dataDir, "worktrees", res.runId)), "worktree removed at run end");
    console.log(`✔ ${NAME} ran end to end offline: ${out["filesChanged"]} files, diff ${(out["diff"] as string).length} bytes, worktree cleaned up`);

    // ── 4. optional: the real thing ──────────────────────────────────────
    if (process.env["CODE_SMOKE_LIVE"] === "1") {
      const repo = process.env["CODE_SMOKE_REPO"] ?? "https://github.com/stakwork/strut";
      const t0 = Date.now();
      const live = await runWorkflow(
        flow,
        { repo, prompt: "Add a single line 'Smoke test.' to the end of README.md. Change nothing else." },
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
