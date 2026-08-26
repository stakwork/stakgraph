/**
 * Smoke test for the GAIA LAB scoring service — no numpy, no dataset
 * download, no LLM. Uses a REAL throwaway git repo as a fake dataset
 * checkout (fixture metadata.jsonl + a stub scorer.py) and REAL python3 for
 * the driver, so the actual subprocess plumbing is exercised end-to-end.
 * Verifies:
 *   - dataset integrity enforcement: missing dir, dirty tree, missing
 *     scorer, sha-pin mismatch all refuse; untracked scorer.py is tolerated
 *   - getTask/listTasks return question/level/file and STRIP the gold
 *   - score() runs the pinned scorer via the python driver, aggregates
 *     accuracy + byLevel, flags unknown task ids, and stamps
 *     benchmarkRev/scorerSha256
 *
 * Run: npx tsx src/lab/gaia/smoke.ts
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { buildGaiaServices } from "./service.js";

function git(cwd: string, ...args: string[]): string {
  return execFileSync("git", args, { cwd, encoding: "utf-8" });
}

const T1 = "task-aaa";
const T2 = "task-bbb";
const T3 = "task-ccc";

/** Stub scorer.py: same signature as the real one, trivial logic (case-
 *  insensitive trimmed match; prints chatter to prove redirect_stdout works). */
const STUB_SCORER = `
def question_scorer(model_answer, ground_truth):
    print("chatter that must not corrupt the json")
    return model_answer.strip().lower() == ground_truth.strip().lower()
`;

function makeFakeDataset(dir: string): void {
  git(dir, "init", "-q");
  git(dir, "config", "user.email", "smoke@test");
  git(dir, "config", "user.name", "smoke");
  const split = join(dir, "2023", "validation");
  mkdirSync(split, { recursive: true });
  const rows = [
    { task_id: T1, Question: "What is 2+2?", Level: 1, file_name: "", "Final answer": "4" },
    { task_id: T2, Question: "Read the sheet.", Level: 2, file_name: "sheet.xlsx", "Final answer": "blue" },
    { task_id: T3, Question: "Hardest one.", Level: 3, file_name: "", "Final answer": "42" },
  ];
  writeFileSync(join(split, "metadata.jsonl"), rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  writeFileSync(join(split, "sheet.xlsx"), "fake-xlsx-bytes");
  git(dir, "add", "-A");
  git(dir, "commit", "-qm", "init");
  // scorer.py arrives UNTRACKED — from the leaderboard Space, not this repo.
  writeFileSync(join(dir, "scorer.py"), STUB_SCORER);
}

async function main() {
  const dir = mkdtempSync(join(tmpdir(), "gaia-smoke-"));
  try {
    makeFakeDataset(dir);

    // ── missing dir refuses ──
    await assert.rejects(
      () => buildGaiaServices({ dir: join(dir, "nope") }).verifyDataset(),
      /does not exist/,
    );

    // ── happy verify: untracked scorer.py tolerated, rev + sha reported ──
    const gaia = buildGaiaServices({ dir });
    const v = await gaia.verifyDataset();
    assert.equal(v.rev.length, 40);
    assert.match(v.scorerSha256, /^[0-9a-f]{64}$/);

    // ── sha pin mismatch refuses ──
    await assert.rejects(
      () => buildGaiaServices({ dir, scorerSha256: "f".repeat(64) }).verifyDataset(),
      /does not match pinned/,
    );

    // ── listTasks / getTask: levels, files, and NO gold ──
    const tasks = await gaia.listTasks();
    assert.equal(tasks.length, 3);
    assert.deepEqual(await gaia.listTasks({ level: 2 }), [
      { taskId: T2, level: 2, hasFile: true },
    ]);
    const t2 = await gaia.getTask(T2);
    assert.equal(t2.question, "Read the sheet.");
    assert.equal(t2.fileName, "sheet.xlsx");
    assert.ok(t2.filePath?.endsWith(join("2023", "validation", "sheet.xlsx")));
    assert.ok(!JSON.stringify(t2).includes("blue"), "gold leaked from getTask");
    assert.ok(!JSON.stringify(tasks).includes("4"), "gold leaked from listTasks");
    await assert.rejects(() => gaia.getTask("nope"), /not in the validation split/);

    // ── score(): real python3 driver + stub scorer ──
    const report = await gaia.score([
      { taskId: T1, answer: " 4 " }, // correct after normalization
      { taskId: T2, answer: "red" }, // wrong
      { taskId: T3, answer: "42" }, // correct
      { taskId: "ghost", answer: "x" }, // unknown id
    ]);
    assert.equal(report.correct, 2);
    assert.equal(report.total, 4);
    assert.equal(report.accuracy, 2 / 3); // unknown id excluded from denominator
    assert.deepEqual(report.byLevel, {
      "1": { correct: 1, total: 1 },
      "2": { correct: 0, total: 1 },
      "3": { correct: 1, total: 1 },
    });
    const ghost = report.results.find((r) => r.taskId === "ghost");
    assert.match(ghost?.error ?? "", /unknown task_id/);
    assert.equal(report.benchmarkRev, v.rev);
    assert.equal(report.scorerSha256, v.scorerSha256);

    // ── empty pairs refuses ──
    await assert.rejects(() => gaia.score([]), /no pairs/);

    // ── dirty tree refuses (tracked file modified) ──
    writeFileSync(join(dir, "2023", "validation", "metadata.jsonl"), "{}\n");
    await assert.rejects(() => gaia.score([{ taskId: T1, answer: "4" }]), /local modifications/);
    git(dir, "checkout", "--", ".");

    // ── missing scorer refuses ──
    rmSync(join(dir, "scorer.py"));
    await assert.rejects(() => buildGaiaServices({ dir }).verifyDataset(), /scorer\.py not found/);

    console.log("gaia smoke: ALL PASS");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
