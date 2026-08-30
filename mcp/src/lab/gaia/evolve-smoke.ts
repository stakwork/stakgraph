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
    // gaia-evolve-gen's `published` gate: did this generation's author ship a
    // NEW version, or did the version fallback just land back on the previous
    // generation's publish? Every access here must stay undefined-safe, since
    // meta/get-workflow returns { error } (never undefined) for a miss.
    const GATE =
      "{{ (vpin.version || vactive.version) && (vpin.version || vactive.version) !== vbefore.version }}";
    const gate = (vbefore: unknown, vpin: unknown, vactive: unknown) =>
      Boolean((resolveConfig as any)(GATE, { ...scope, vbefore, vpin, vactive }));
    // gen 0: candidate does not exist yet, author publishes v1 → shipped
    assert.equal(gate({ error: "not found" }, { version: "v1" }, { version: "v1" }), true);
    // gen 0: author publishes nothing at all → no-op
    assert.equal(gate({ error: "not found" }, {}, { error: "not found" }), false);
    // gen N: author publishes a new version → shipped
    assert.equal(gate({ version: "v11" }, { version: "v12" }, { version: "v12" }), true);
    // gen N: author echoes garbage and published nothing, so the fallback
    // resolves to the PREVIOUS generation's publish → no-op (the live bug)
    assert.equal(gate({ version: "v11" }, {}, { version: "v11" }), false);
    // the no-op flag handed to the loop is the gate's negation
    assert.equal((resolveConfig as any)("{{ !published }}", { ...scope, published: true }), false);
    assert.equal((resolveConfig as any)("{{ !published }}", { ...scope, published: false }), true);
    console.log("✔ template expressions resolve (fallback, version pin, whole-run pass, published gate; no-short-circuit guarded)");

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

    // 6. NO-OP generation: the gen workflow's `published` gate reports that an
    //    author shipped nothing, so nothing was graded. The loop must record
    //    it without a fitness, leave `best` untouched, spend only the author
    //    cost, and tell the next generation not to read it as evidence.
    const noopCalls: any[] = [];
    const noopOpt = {
      run: async (_name: string, input: any) => {
        noopCalls.push(input);
        const g = input.generation as number;
        return {
          runId: `genrun-${g}`,
          status: "success",
          output:
            g === 1
              ? { candidate: input.candidateName, noop: true, authorCost: 1, summary: "ran out of steps" }
              : {
                  candidate: input.candidateName,
                  version: `v${g + 1}`,
                  summary: `approach ${g}`,
                  authorCost: 1,
                  digest: { fitness: 0.6, text: `digest ${g}`, results: [{ cost: 2 }] },
                },
        };
      },
    };
    const noopBase = {
      tasks: ["t-1", "t-2"],
      mission: "m",
      baseline: { fitness: 0.4, text: "baseline digest" },
      candidateName: "gaia-produce-ai",
      baseWorkflow: "gaia-produce",
      genWorkflow: "gaia-evolve-gen",
      fitnessName: "accuracy",
      stopFitness: 1,
      improveMargin: 0,
      exploreAfter: 2,
    };
    const noopOut: any = await loop.run(
      loop.input.parse({ ...noopBase, maxGenerations: 3 }),
      { ...ctxStub, services: { optimizer: noopOpt } },
    );
    assert.equal(noopOut.generations[1].noop, true);
    assert.equal(noopOut.bestGen, 0); // gen 1 did not displace gen 0's v1
    assert.equal(noopOut.bestVersion, "v1");
    assert.equal(noopOut.bestFitness, 0.6);
    // no-op spends the author budget only — never the 2 tasks × cost 2 produce
    assert.equal(noopOut.totalKnownCost, 3 + 1 + 3);
    // the briefing must not libel an approach that was never tried
    assert.ok(noopCalls[2].briefing.includes("NO CANDIDATE PUBLISHED"));
    assert.ok(!noopCalls[2].briefing.includes("attempt 1 → published"));
    console.log("✔ eval/evolve-loop: no-op generation records no fitness, spends only the author budget");

    // 7. RE-SCORE guard: the same version graded twice cannot be promoted on
    //    the luckier sample — the run's best stays pinned to first measurement.
    const dupOpt = {
      run: async (_name: string, input: any) => {
        const g = input.generation as number;
        return {
          runId: `genrun-${g}`,
          status: "success",
          output: {
            candidate: input.candidateName,
            version: "v1", // gen 1 re-runs gen 0's version…
            summary: `approach ${g}`,
            authorCost: 1,
            digest: { fitness: g === 0 ? 0.6 : 0.9, text: `digest ${g}`, results: [{ cost: 2 }] }, // …and gets lucky
          },
        };
      },
    };
    const dupOut: any = await loop.run(
      loop.input.parse({ ...noopBase, maxGenerations: 2 }),
      { ...ctxStub, services: { optimizer: dupOpt } },
    );
    assert.equal(dupOut.bestGen, 0);
    assert.equal(dupOut.bestFitness, 0.6); // NOT 0.9 — a resample, not a climb
    assert.equal(dupOut.generations[1].fitness, 0.9); // still reported honestly
    console.log("✔ eval/evolve-loop: a re-scored version cannot become the best on sampling luck");

    // 8. BUDGET caps stop the loop between generations.
    const costOpt = {
      run: async (_name: string, input: any) => ({
        runId: `genrun-${input.generation}`,
        status: "success",
        output: {
          candidate: input.candidateName,
          version: `v${input.generation + 1}`,
          summary: `approach ${input.generation}`,
          authorCost: 1,
          digest: { fitness: 0.5, text: "d", results: [{ cost: 2 }] },
        },
      }),
    };
    const cappedOut: any = await loop.run(
      loop.input.parse({ ...noopBase, maxGenerations: 10, maxCost: 8 }),
      { ...ctxStub, services: { optimizer: costOpt } },
    );
    // $3/gen (author 1 + produce 2); the gate trips before gen 3, at $9 ≥ $8
    assert.equal(cappedOut.generations.length, 3);
    assert.ok(cappedOut.stopReason.includes("maxCost $8 reached"));
    const uncappedOut: any = await loop.run(
      loop.input.parse({ ...noopBase, maxGenerations: 10 }),
      { ...ctxStub, services: { optimizer: costOpt } },
    );
    assert.equal(uncappedOut.generations.length, 10); // no cap = unchanged
    // gaia-evolve wires `{{ input.maxCost || params.maxCost }}`, and an unset
    // YAML param resolves to null — the schema must read that as "uncapped"
    // rather than rejecting the whole step.
    assert.equal(loop.input.parse({ ...noopBase, maxGenerations: 1, maxCost: null, maxMinutes: null }).maxCost, null);
    console.log("✔ eval/evolve-loop: maxCost stops between generations, absent caps change nothing");

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
