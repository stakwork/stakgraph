/**
 * Offline validation for eval/matrix (the task×version matrix —
 * plans/evolve-scoreboard-and-task-matrix.md, Phase 1). The fixtures are
 * modeled on prod run gaia-evolve/1788061734710, whose pathologies the step
 * exists to surface:
 *  1. bands: tasks correct in every / no / some measurement(s)
 *  2. empirical noise floor from same-version re-measurements (v11 was
 *     accidentally measured twice, 0.76 vs 0.80 — identical YAML)
 *  3. bias tag: a never-correct task answering byte-identically ≥3 times
 *     ("36" ×9 in prod) vs a variance-tagged one (distinct wrong answers)
 *  4. no same-version pairs → the floor reads UNKNOWN, never zero
 *  5. the three correctness shapes gaia-run / gaia-candidate-run emit
 * No LLM, no network, no dataset.
 * Run: npx tsx src/lab/eval/matrix-smoke.ts
 */
import assert from "node:assert/strict";
import matrix from "./steps/matrix.js";

type AnyRec = Record<string, unknown>;

/** gaia-candidate-run shape: boolean `correct`. */
function res(taskId: string, correct: boolean, answer: string, level = 2, extra: AnyRec = {}): AnyRec {
  return { taskId, level, correct, answer, ...extra };
}

const ctx = { runId: "smoke", path: "matrix", scope: {}, input: {}, emit: async () => {}, services: {}, registry: {} } as never;

async function main() {
  // ── the main fixture: 4 tasks × 4 measurements of 3 versions ───────────
  // floor-1: correct everywhere; bias-1: never correct, identical "36";
  // var-1: never correct, scattered answers; flip-1: movable, flips on the
  // v11 re-measurement (identical YAML — pure sampling noise).
  const meas = (version: string, flip1: boolean, varAns: string) => ({
    version,
    results: [
      res("floor-1", true, "right"),
      res("bias-1", false, "36", 3),
      res("var-1", false, varAns, 3),
      res("flip-1", true && flip1, flip1 ? "ok" : "nope"),
    ],
  });
  const out = (await matrix.run(
    {
      measurements: [
        meas("gaia-produce", false, "0.00073"),
        meas("v10", true, "0.00022"),
        meas("v11", true, "0.0031"),
        meas("v11", false, "7"), // the accidental re-measurement
      ],
      maxAnswerChars: 120,
    },
    ctx,
  )) as AnyRec;

  // 1. bands
  const tasks = out.tasks as AnyRec[];
  const byId = Object.fromEntries(tasks.map((t) => [t.taskId as string, t]));
  assert.equal(byId["floor-1"]!.band, "floor");
  assert.equal(byId["bias-1"]!.band, "ceiling");
  assert.equal(byId["var-1"]!.band, "ceiling");
  assert.equal(byId["flip-1"]!.band, "movable");
  assert.deepEqual(out.bands, { floor: 1, movable: 1, ceiling: 2 });
  console.log("✔ bands: floor / movable / ceiling split as observed");

  // 2. noise floor from the v11 pair: fitness 3/4 vs 2/4 → |Δ| = 0.25, 1 flip
  const noise = out.noise as AnyRec;
  assert.equal(noise.sameVersionPairs, 1);
  assert.equal(noise.floorKnown, true);
  assert.equal(noise.maxAbsFitnessDelta, 0.25);
  assert.equal(noise.maxTaskFlips, 1);
  assert.equal(noise.suggestedMargin, 0.25);
  console.log("✔ noise floor measured from the same-version pair (Δ0.25, 1 flip)");

  // 3. bias vs variance on ceiling tasks
  assert.equal(byId["bias-1"]!.bias, true);
  assert.equal(byId["bias-1"]!.repeatedAnswer, "36");
  assert.ok(!byId["var-1"]!.bias, "distinct wrong answers must not tag as bias");
  assert.ok(((byId["var-1"]!.wrongAnswers as string[]) ?? []).length >= 3);
  console.log("✔ bias (identical ×4) vs variance (distinct answers) tagged");

  // versions aggregate + text rendering carries the actionable lines
  const versions = out.versions as AnyRec[];
  assert.deepEqual(versions.map((v) => v.version), ["gaia-produce", "v10", "v11"]);
  assert.deepEqual((versions[2] as AnyRec).fitness, [0.5, 0.25]);
  const text = out.text as string;
  assert.ok(text.includes("BIAS"), "text names the bias failure");
  assert.ok(text.includes("±0.25"), "text states the measured margin");
  console.log("✔ version rows + text rendering");

  // 4. single measurement per version → floor UNKNOWN, never zero
  const single = (await matrix.run(
    { measurements: [meas("gaia-produce", true, "x"), meas("v10", false, "y")], maxAnswerChars: 120 },
    ctx,
  )) as AnyRec;
  const n2 = single.noise as AnyRec;
  assert.equal(n2.sameVersionPairs, 0);
  assert.equal(n2.floorKnown, false);
  assert.equal(n2.suggestedMargin, undefined);
  assert.ok((single.text as string).includes("UNKNOWN"), "text says the floor is unmeasured");
  console.log("✔ no same-version pair → noise floor UNKNOWN (not 0)");

  // 5. correctness shape normalization: gaia-run's results-array and count
  const shapes = (await matrix.run(
    {
      measurements: [
        {
          version: "base",
          results: [
            { taskId: "a", level: 1, results: [{ correct: true }], answer: "1" },
            { taskId: "b", level: 1, correct: 1, total: 1, answer: "2" },
            { taskId: "c", level: 1, correct: 0, total: 1, answer: "3" },
            { taskId: "d", level: 1, answer: "unreadable shape" },
          ],
        },
      ],
      maxAnswerChars: 120,
    },
    ctx,
  )) as AnyRec;
  const sById = Object.fromEntries((shapes.tasks as AnyRec[]).map((t) => [t.taskId as string, t]));
  assert.equal(sById["a"]!.band, "floor");
  assert.equal(sById["b"]!.band, "floor");
  assert.equal(sById["c"]!.band, "ceiling");
  assert.equal(sById["d"]!.band, "ceiling"); // unreadable → false, never true
  console.log("✔ all three graded-result shapes normalize");

  console.log("\nmatrix smoke: all checks passed");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
