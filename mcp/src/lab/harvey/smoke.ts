/**
 * Offline smoke test for the Harvey LAB verification service + steps — no
 * uv/python, no LLM. Uses a REAL throwaway git repo as a fake harvey-labs
 * checkout and a FAKE exec for the `uv` eval invocation (writes a plausible
 * scores.json like the real harness would). Verifies:
 *   - checkout integrity enforcement: missing dir, dirty tree, rev-pin
 *     mismatch all refuse; untracked results/ runs are tolerated
 *   - getTask returns instructions/documents and STRIPS the rubric
 *   - evaluate stages artifact deliverables into results/<runId>/output/,
 *     invokes the eval CLI with the right args, and returns scores +
 *     benchmarkRev
 *   - the seeded steps wire through ctx.services (registry discovery)
 *
 * Run: npx tsx src/lab/harvey/smoke.ts
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, mkdirSync, writeFileSync, readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { WorkspaceManager, buildRegistry, fileArtifactsCapability, type StepContext } from "vein";
import { buildHarveyServices, type ExecFn, type ExecResult } from "./service.js";
import { seedHarveySteps } from "./seed.js";

const TASK = "corporate-ma/test-task";

function git(cwd: string, ...args: string[]): string {
  return execFileSync("git", args, { cwd, encoding: "utf-8" });
}

/** A minimal fake harvey-labs checkout: tasks/<TASK>/{task.json,documents}, committed. */
function makeFakeCheckout(dir: string): void {
  git(dir, "init", "-q");
  git(dir, "config", "user.email", "smoke@test");
  git(dir, "config", "user.name", "smoke");
  const taskDir = join(dir, "tasks", TASK);
  mkdirSync(join(taskDir, "documents"), { recursive: true });
  writeFileSync(join(taskDir, "documents", "psa.md"), "# purchase agreement");
  writeFileSync(
    join(taskDir, "task.json"),
    JSON.stringify({
      title: "Test task",
      work_type: "review",
      tags: ["m&a"],
      instructions: "Review the PSA. Output: `memo.md`.",
      deliverables: { "memo.md": "memo.md" },
      criteria: [{ id: "c1", title: "secret rubric", match_criteria: "must mention X", deliverables: ["memo.md"] }],
    }),
  );
  writeFileSync(join(dir, ".gitignore"), "results/\n");
  git(dir, "add", "-A");
  git(dir, "commit", "-qm", "init");
}

/** Fake `uv run python -m evaluation.run_eval …`: writes scores.json the way
 *  the real harness does. git commands pass through to the real git. */
function makeFakeExec(checkout: string, log: Array<{ cmd: string; args: string[] }>): ExecFn {
  return async (cmd, args, opts): Promise<ExecResult> => {
    log.push({ cmd, args });
    if (cmd === "git") {
      try {
        const out = execFileSync("git", args, { cwd: opts.cwd, encoding: "utf-8" });
        return { code: 0, stdout: out, stderr: "" };
      } catch (err: any) {
        return { code: err.status ?? 1, stdout: err.stdout?.toString() ?? "", stderr: err.stderr?.toString() ?? String(err) };
      }
    }
    // the eval CLI
    const runId = args[args.indexOf("--run-id") + 1];
    const runDir = join(checkout, "results", runId);
    // the real harness requires the staged output dir — mimic that check
    if (!existsSync(join(runDir, "output", "memo.md"))) {
      return { code: 1, stdout: "", stderr: "run directory not found or empty" };
    }
    const scores = {
      score: 1.0, max_score: 1.0, summary: "1/1 criteria passed.  ALL-PASS.",
      all_pass: true, n_criteria: 1, n_passed: 1,
      criteria_results: [{ id: "c1", verdict: "pass", reasoning: "mentions X" }],
      run_id: runId, task: args[args.indexOf("--task") + 1],
      judge_model: "claude-sonnet-4-6", scored_at: "2026-08-21T00:00:00Z",
      ...(existsSync(join(runDir, "metrics.json"))
        ? { cost: JSON.parse(readFileSync(join(runDir, "metrics.json"), "utf-8")) }
        : {}),
    };
    writeFileSync(join(runDir, "scores.json"), JSON.stringify(scores));
    return { code: 0, stdout: "ok", stderr: "" };
  };
}

async function main() {
  const base = mkdtempSync(join(process.cwd(), ".harvey-smoke-"));
  try {
    const checkout = join(base, "harvey-labs");
    mkdirSync(checkout);
    makeFakeCheckout(checkout);
    const headRev = git(checkout, "rev-parse", "HEAD").trim();
    const log: Array<{ cmd: string; args: string[] }> = [];
    const harvey = buildHarveyServices({ dir: checkout, exec: makeFakeExec(checkout, log) });

    // ── integrity: happy path ────────────────────────────────────────────
    const v = await harvey.verifyCheckout();
    assert.equal(v.rev, headRev);
    console.log("✔ verifyCheckout (clean tree)");

    // missing dir
    await assert.rejects(
      () => buildHarveyServices({ dir: join(base, "nope"), exec: makeFakeExec(checkout, []) }).verifyCheckout(),
      /does not exist/,
    );
    // unset dir
    delete process.env.HARVEY_LABS_DIR;
    await assert.rejects(
      () => buildHarveyServices({ exec: makeFakeExec(checkout, []) }).verifyCheckout(),
      /HARVEY_LABS_DIR not configured/,
    );
    // rev-pin mismatch
    await assert.rejects(
      () => buildHarveyServices({ dir: checkout, rev: "deadbeef", exec: makeFakeExec(checkout, []) }).verifyCheckout(),
      /does not match pinned/,
    );
    // matching pin (prefix) passes
    await buildHarveyServices({ dir: checkout, rev: headRev.slice(0, 8), exec: makeFakeExec(checkout, []) }).verifyCheckout();
    console.log("✔ integrity: missing dir / unset env / rev pin");

    // dirty tree refuses; untracked results/ tolerated
    writeFileSync(join(checkout, "tasks", TASK, "task.json.bak"), "tamper");
    await assert.rejects(() => harvey.verifyCheckout(), /local modifications/);
    rmSync(join(checkout, "tasks", TASK, "task.json.bak"));
    mkdirSync(join(checkout, "results", "old-run"), { recursive: true });
    writeFileSync(join(checkout, "results", "old-run", "scores.json"), "{}");
    // results/ is gitignored in the fake checkout, but ALSO verify the
    // untracked-results tolerance directly with the ignore removed:
    rmSync(join(checkout, ".gitignore"));
    git(checkout, "add", "-A");
    git(checkout, "commit", "-qm", "drop gitignore");
    await harvey.verifyCheckout(); // untracked results/** only → clean
    console.log("✔ integrity: dirty tree refuses; untracked results/ tolerated");

    // ── getTask strips the rubric ────────────────────────────────────────
    const task = await harvey.getTask(TASK);
    assert.equal(task.title, "Test task");
    assert.deepEqual(task.deliverables, ["memo.md"]);
    assert.deepEqual(task.documents, ["psa.md"]);
    assert.ok(!("criteria" in task), "criteria must be stripped");
    assert.ok(!JSON.stringify(task).includes("secret rubric"), "rubric text must not leak");
    await assert.rejects(() => harvey.getTask("../../etc"), /invalid task/);
    console.log("✔ getTask strips rubric + rejects traversal");

    // ── evaluate: stage artifacts → run CLI → scores + rev ───────────────
    const artifactsRoot = join(base, "artifacts");
    const artifacts = fileArtifactsCapability(artifactsRoot);
    await artifacts.write("run42", "output/memo.md", "memo mentioning X");

    const result = await harvey.evaluate({
      task: TASK,
      sourceDir: join(artifactsRoot, "run42", "output"),
      runId: "vein-run42",
      metrics: { input_tokens: 100, output_tokens: 50 },
    });
    assert.equal(result.all_pass, true);
    assert.equal(result.benchmarkRev, git(checkout, "rev-parse", "HEAD").trim());
    assert.ok(String(result.reportPath).endsWith("report.html"));
    // staged where the harness looks
    assert.equal(
      readFileSync(join(checkout, "results", "vein-run42", "output", "memo.md"), "utf-8"),
      "memo mentioning X",
    );
    // CLI invoked correctly
    const uvCall = log.find((c) => c.cmd === "uv")!;
    assert.deepEqual(uvCall.args.slice(0, 4), ["run", "python", "-m", "evaluation.run_eval"]);
    assert.ok(uvCall.args.includes("vein-run42") && uvCall.args.includes(TASK));
    console.log("✔ evaluate: staging + CLI + scores + benchmarkRev");

    // runId sanitization
    await assert.rejects(
      () => harvey.evaluate({ task: TASK, sourceDir: join(artifactsRoot, "run42", "output"), runId: "." }),
      /invalid runId/,
    );
    console.log("✔ evaluate rejects bad runId");

    // ── seeded steps: discovery + wiring through ctx.services ────────────
    const wsDir = join(base, "ws");
    const workspace = new WorkspaceManager(wsDir);
    await seedHarveySteps(workspace);
    const { registry } = await buildRegistry(workspace.path);
    assert.ok(registry["harvey/get-task"] && registry["harvey/evaluate"]);

    const ctx = {
      runId: "run42", path: "smoke", scope: {}, input: undefined,
      emit: async () => {},
      services: { harvey, artifacts },
    } as unknown as StepContext;

    const stepTask: any = await registry["harvey/get-task"].run(
      registry["harvey/get-task"].input.parse({ task: TASK }), ctx,
    );
    assert.equal(stepTask.title, "Test task");
    assert.ok(!("criteria" in stepTask));

    const stepScores: any = await registry["harvey/evaluate"].run(
      registry["harvey/evaluate"].input.parse({ task: TASK }), ctx,
    );
    assert.equal(stepScores.all_pass, true);
    console.log("✔ seeded steps wire through ctx.services");

    console.log("\nALL HARVEY SMOKE CHECKS PASSED");
  } finally {
    rmSync(base, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
