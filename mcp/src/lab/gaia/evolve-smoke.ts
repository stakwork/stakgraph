/**
 * Offline validation for the gaia evolve harness (the gaia instance of the
 * generic eval/evolve-loop):
 *  1. all six gaia workflows seed + parse into Flows (YAML → Flow schema)
 *  2. the template expressions used in them resolve as intended
 *  3. gaia/evaluate's fromRun mode unpacks candidate runs safely in code
 *  4. gaia/digest-results normalizes every result shape + tags misses
 *  5. eval/evolve-loop climbs the digest's `fitness` (accuracy) field
 * No python, no dataset checkout, no LLM, no network.
 * Run: npx tsx src/lab/gaia/evolve-smoke.ts
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { WorkspaceManager, buildRegistry, resolveConfig } from "vein";
import { seedGaiaSteps, seedGaiaWorkflows } from "./seed.js";
import { seedEvalSteps } from "../eval/seed.js";
import { seedArtifactSteps } from "../artifacts/seed.js";

async function main() {
  const base = mkdtempSync(join(process.cwd(), ".gaia-evolve-validate-"));
  try {
    const workspace = new WorkspaceManager(join(base, "ws"));
    await seedGaiaSteps(workspace);
    await seedEvalSteps(workspace); // eval/evolve-loop — the generic hill-climb gaia-evolve wires
    await seedArtifactSteps(workspace);
    await seedGaiaWorkflows(workspace);

    // 1. every seeded workflow parses into a Flow
    for (const name of ["gaia-produce", "gaia-run", "gaia-batch", "gaia-candidate-run", "gaia-evolve-gen", "gaia-evolve"]) {
      const flow = await workspace.getWorkflow(name);
      assert.equal(flow.name, name);
      assert.ok(Array.isArray(flow.steps) && flow.steps.length > 0, `${name} has steps`);
      console.log(`✔ workflow parses: ${name} (${flow.steps.length} steps)`);
    }

    // step types referenced by the workflows all exist in the registry
    const { registry } = await buildRegistry(workspace.path);
    const wanted = [
      "gaia/list-tasks", "gaia/get-task", "gaia/evaluate", "gaia/pack-result",
      "gaia/summarize-batch", "gaia/digest-results", "eval/evolve-loop",
      "artifacts/dir", "meta/run-workflow", "agent", "subflow", "foreach",
    ];
    for (const t of wanted) assert.ok(registry[t], `registry has ${t}`);
    console.log("✔ all referenced step types resolve");

    // 2. template expressions used in the new workflows
    const scope: Record<string, unknown> = {
      input: { taskId: "t-1", tasks: ["t-1", "t-2"] },
      run: { runId: "r1", status: "success", output: { taskId: "t-1", answer: "42", cost: 0.5, steps: 3 } },
      grade: { isCorrect: true, answer: "42", benchmarkRev: "abc" },
      vactive: { version: "v2" },
      vpin: {},
      params: {},
    };
    const r = (s: string) => (resolveConfig as any)(s, scope);
    assert.equal(r("{{ input.produceWorkflow || 'gaia-produce' }}"), "gaia-produce");
    assert.equal(r("{{ vpin.version || vactive.version }}"), "v2");
    // The evaluator does NOT short-circuit — a failed candidate run has no
    // `output`, so the workflows pass the WHOLE run into gaia/evaluate's
    // fromRun and unpack in code, never `run.output.answer` in YAML.
    const failedScope = { ...scope, run: { runId: "r2", status: "failed", error: { message: "boom" } } };
    assert.throws(
      () => (resolveConfig as any)("{{ run.output ? run.output.answer : '' }}", failedScope),
      /Cannot access/,
    );
    assert.deepEqual((resolveConfig as any)("{{ run }}", failedScope), failedScope.run);
    // gaia-candidate-run's grade fallback keeps isCorrect/answer shape-stable,
    // so the result pack's one-level reads stay safe:
    const fallbackScope = { ...scope, grade: { taskId: "t-1", isCorrect: false, answer: "", level: null, gradeError: "boom" } };
    assert.equal((resolveConfig as any)("{{ grade.isCorrect }}", fallbackScope), false);
    assert.equal((resolveConfig as any)("{{ grade.benchmarkRev }}", fallbackScope), undefined);
    console.log("✔ template expressions resolve (fallback, version pin, whole-run pass; no-short-circuit guarded)");

    // 3. gaia/evaluate fromRun: unpack-in-code semantics against a faked
    //    scoring service (no python, no dataset).
    const evaluate = registry["gaia/evaluate"]!;
    const scored: any[] = [];
    const fakeGaia = {
      getTask: async (taskId: string) => {
        if (taskId === "t-2") throw new Error("metadata unavailable"); // must not fail the grade
        return { taskId, question: `question for ${taskId}`, level: 1, fileName: "" };
      },
      score: async (pairs: Array<{ taskId: string; answer: string }>) => {
        scored.push(pairs);
        const results = pairs.map((p) => ({ taskId: p.taskId, level: 1, correct: p.answer === "42" }));
        const correct = results.filter((x) => x.correct).length;
        return {
          accuracy: results.length ? correct / results.length : 0,
          correct,
          total: results.length,
          byLevel: { "1": { correct, total: results.length } },
          results,
          benchmarkRev: "rev0",
          scorerSha256: "sha0",
        };
      },
    };
    const ctxStub = { runId: "r", path: "p", scope: {}, input: undefined, emit: async () => {}, services: {}, registry } as any;
    const evalCtx = { ...ctxStub, services: { gaia: fakeGaia } };
    const ok: any = await evaluate.run(
      evaluate.input.parse({ fromRun: { taskId: "t-1", run: { runId: "r1", status: "success", output: { answer: "42" } } } }),
      evalCtx,
    );
    assert.equal(ok.isCorrect, true);
    assert.equal(ok.answer, "42");
    assert.equal(ok.level, 1);
    assert.equal(ok.question, "question for t-1");
    assert.equal(ok.benchmarkRev, "rev0");
    // a failed run (no output) scores as "" — an honest zero, not a throw —
    // and a failing metadata lookup degrades to question null, never a throw
    const failed: any = await evaluate.run(
      evaluate.input.parse({ fromRun: { taskId: "t-2", run: { runId: "r2", status: "failed", error: { message: "boom" } } } }),
      evalCtx,
    );
    assert.equal(failed.isCorrect, false);
    assert.equal(failed.answer, "");
    assert.equal(failed.question, null);
    assert.deepEqual(scored[1], [{ taskId: "t-2", answer: "" }]);
    // exactly one of pairs/fromRun
    await assert.rejects(() => evaluate.run(evaluate.input.parse({}), evalCtx), /exactly one/);
    await assert.rejects(
      () =>
        evaluate.run(
          evaluate.input.parse({ pairs: [{ taskId: "t", answer: "a" }], fromRun: { taskId: "t", run: {} } }),
          evalCtx,
        ),
      /exactly one/,
    );
    console.log("✔ gaia/evaluate fromRun: unpacks in code, honest zero on failed runs, mode exclusivity");

    // 4. digest step over every result shape that reaches it
    const digest = registry["gaia/digest-results"]!;
    const out: any = await digest.run(
      digest.input.parse({
        results: [
          // gaia-candidate-run shape: boolean correct, cost/steps only
          // inside runResult.output
          {
            taskId: "t-1", level: 1, correct: true, answer: "42", question: "What is…",
            runResult: { runId: "x", status: "success", output: { cost: 0.5, steps: 12 } },
          },
          // gaia-run shape: correct as a COUNT + the score call's results array
          {
            taskId: "t-2", level: 1, correct: 0, total: 1,
            results: [{ taskId: "t-2", level: 1, correct: false }],
            question: "Which bird…", answer: "eagle (Aquila)", cost: 0.8, steps: 40,
          },
          // produce blew up: onError fallback answer + error message
          { taskId: "t-3", level: 2, correct: false, answer: "", question: "How many…", produceError: "AI_NoObjectGeneratedError" },
          // gave up: empty answer, no error
          { taskId: "t-4", level: 2, correct: false, answer: "", question: "In what year…" },
        ],
      }),
      ctxStub,
    );
    assert.equal(out.n, 4);
    assert.equal(out.correctCount, 1);
    assert.equal(out.accuracy, 0.25);
    assert.equal(out.fitness, 0.25); // the field eval/evolve-loop reads
    assert.deepEqual(out.byLevel, { "1": { correct: 1, total: 2 }, "2": { correct: 0, total: 2 } });
    assert.equal(out.results[0].correct, true);
    assert.equal(out.results[0].cost, 0.5); // unpacked from runResult.output in code
    assert.equal(out.results[0].steps, 12);
    assert.equal(out.results[0].question, undefined); // question excerpts ride on MISSES only
    assert.equal(out.results[1].correct, false); // count+results normalization
    assert.equal(out.results[1].miss, "wrong-answer");
    assert.equal(out.results[2].miss, "produce-error");
    assert.equal(out.results[3].miss, "empty-answer");
    assert.ok(out.text.includes("accuracy 0.25 (1/4 correct)"));
    assert.ok(out.text.includes("1 wrong-answer") && out.text.includes("1 empty-answer") && out.text.includes("1 produce-error"));
    assert.ok(out.text.includes('answered: "eagle (Aquila)"'));
    assert.ok(out.text.includes("question: Which bird…"));
    assert.ok(out.text.includes("ERROR: AI_NoObjectGeneratedError"));
    console.log("✔ gaia/digest-results: shape normalization, miss taxonomy, fitness, text");

    // 5. the generic loop climbs the gaia digest's `fitness` field (accuracy)
    //    with improveMargin 0 — any task flip counts — and names the fitness
    //    "accuracy" in briefings.
    const loop = registry["eval/evolve-loop"]!;
    const genCalls: any[] = [];
    const rates = [0.4, 0.6];
    const fakeOpt = {
      run: async (_name: string, input: any) => {
        genCalls.push(input);
        const g = input.generation as number;
        return {
          runId: `genrun-${g}`,
          status: "success",
          output: {
            candidate: input.candidateName,
            version: `v${g + 1}`,
            // gen 0 echoes filler (the observed schema-mode failure); the
            // loop must replace it with the honest no-summary marker.
            summary: g === 0 ? "placeholder" : `approach ${g}`,
            authorCost: 1,
            digest: { fitness: rates[g], text: `digest ${g}`, results: [{ cost: 2 }] },
          },
        };
      },
    };
    const loopOut: any = await loop.run(
      loop.input.parse({
        tasks: ["t-1", "t-2"],
        mission: "m",
        baseline: { fitness: 0.4, text: "baseline digest" },
        candidateName: "gaia-produce-ai",
        baseWorkflow: "gaia-produce",
        genWorkflow: "gaia-evolve-gen",
        fitnessName: "accuracy",
        maxGenerations: 5,
        stopFitness: 0.6,
        improveMargin: 0,
        exploreAfter: 2,
      }),
      { ...ctxStub, services: { optimizer: fakeOpt } },
    );
    assert.equal(genCalls.length, 2); // gen 1 hit stopFitness 0.6
    assert.equal(loopOut.stopReason, "stopFitness 0.6 reached");
    assert.equal(loopOut.bestGen, 1);
    assert.equal(loopOut.bestVersion, "v2");
    assert.equal(loopOut.bestFitness, 0.6);
    assert.equal(loopOut.baselineFitness, 0.4);
    assert.equal(loopOut.improved, true);
    assert.ok(genCalls[0].briefing.includes("mean accuracy 0.4"));
    assert.ok(genCalls[1].briefing.includes('the seeded produce workflow "gaia-produce"'));
    // margin 0 semantics: a TIE (0.4 vs baseline 0.4) does not become best
    assert.ok(genCalls[1].briefing.includes("BEST SO FAR: the baseline itself"));
    // junk-summary guard: gen 0's "placeholder" echo must NOT reach the next
    // briefing or the report — both carry the honest no-summary marker.
    assert.ok(!genCalls[1].briefing.includes("placeholder"));
    assert.ok(genCalls[1].briefing.includes("no usable approach summary"));
    assert.equal(
      loopOut.generations[0].summary.includes("no usable approach summary"),
      true,
    );
    assert.equal(loopOut.generations[1].summary, "approach 1");
    console.log("✔ eval/evolve-loop: climbs gaia `fitness`, accuracy naming, margin-0 tie handling, junk-summary guard");

    console.log("\nALL GAIA EVOLVE VALIDATION CHECKS PASSED");
  } finally {
    rmSync(base, { recursive: true, force: true });
  }
}

main().then(
  () => process.exit(0),
  (err) => {
    console.error(err);
    process.exit(1);
  },
);
