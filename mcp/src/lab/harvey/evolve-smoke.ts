/**
 * Offline validation for the harvey produce/grade split + evolve harness:
 *  1. all four workflows seed + parse into Flows (YAML → Flow schema)
 *  2. the template expressions used in them resolve as intended
 *  3. harvey/digest-results aggregates fixture score reports correctly
 *  4. artifacts/dir `sub` creates/guards subdirs
 * No uv/python, no LLM, no network.
 * Run: npx tsx src/lab/harvey/evolve-smoke.ts
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { WorkspaceManager, buildRegistry, fileArtifactsCapability, resolveConfig } from "vein";
import { seedHarveySteps, seedHarveyWorkflows } from "./seed.js";
import { seedArtifactSteps } from "../artifacts/seed.js";

async function main() {
  const base = mkdtempSync(join(process.cwd(), ".evolve-validate-"));
  try {
    const workspace = new WorkspaceManager(join(base, "ws"));
    await seedHarveySteps(workspace);
    await seedArtifactSteps(workspace);
    await seedHarveyWorkflows(workspace);

    // 1. every seeded workflow parses into a Flow
    for (const name of ["harvey-produce", "harvey-run", "harvey-candidate-run", "harvey-evolve-gen", "harvey-evolve"]) {
      const flow = await workspace.getWorkflow(name);
      assert.equal(flow.name, name);
      assert.ok(Array.isArray(flow.steps) && flow.steps.length > 0, `${name} has steps`);
      console.log(`✔ workflow parses: ${name} (${flow.steps.length} steps)`);
    }

    // step types referenced by the workflows all exist in the registry
    const { registry } = await buildRegistry(workspace.path);
    const wanted = [
      "harvey/get-task", "harvey/evaluate", "harvey/pack-result", "harvey/digest-results",
      "harvey/evolve-loop", "artifacts/dir", "meta/run-workflow", "agent", "subflow", "foreach",
    ];
    for (const t of wanted) assert.ok(registry[t], `registry has ${t}`);
    console.log("✔ all referenced step types resolve");

    // 2. template expressions used in the new workflows
    const scope: Record<string, unknown> = {
      input: { task: "a/b", tasks: ["a/b", "c/d"] },
      produced: { outputDir: "/x/output", cost: 1.2, steps: 7, usage: { inputTokens: 100, outputTokens: 50 } },
      run: { runId: "r1", status: "success", output: { outputDir: "/y/output", cost: 0.5, steps: 3 } },
      grade: { score: 0.5 },
      author: { object: { candidate: "harvey-produce-ai", version: "v3" }, cost: 2, steps: 9 },
      basedigest: { meanScore: 0.4, text: "digest" },
      canddigest: { meanScore: 0.7 },
      params: {},
    };
    const r = (s: string) => (resolveConfig as any)(s, scope);
    assert.equal(r("{{ input.produceWorkflow || 'harvey-produce' }}"), "harvey-produce");
    // The evaluator does NOT short-circuit (ternary/&&/|| still evaluate both
    // sides), so the workflows never deep-access possibly-undefined objects.
    const failedScope = { ...scope, run: { runId: "r2", status: "failed", error: { message: "boom" } } };
    assert.throws(
      () => (resolveConfig as any)("{{ run.output ? run.output.outputDir : undefined }}", failedScope),
      /Cannot access/,
    );
    // Instead: whole objects are passed and unpacked in step code…
    assert.deepEqual((resolveConfig as any)("{{ run }}", failedScope), failedScope.run);
    assert.deepEqual((resolveConfig as any)("{{ run.error }}", failedScope), { message: "boom" });
    // …and error-path packs keep usage always-present ({}), so one-level-deep
    // access on it stays safe:
    const onErrorScope = { ...scope, produced: { usage: {}, cost: 0, steps: 0 } };
    assert.equal((resolveConfig as any)("{{ produced.usage.inputTokens }}", onErrorScope), undefined);
    assert.equal(r("{{ produced.usage.inputTokens }}"), 100);
    assert.equal(r("{{ canddigest.meanScore - basedigest.meanScore }}"), 0.7 - 0.4);
    assert.equal(r("{{ input.tasks.length }}"), 2);
    console.log("✔ template expressions resolve (fallback, whole-object pass, arithmetic; no-short-circuit guarded)");

    // 3. digest step over fixture score reports (incl. onError shapes)
    const digest = registry["harvey/digest-results"]!;
    const ctxStub = { runId: "r", path: "p", scope: {}, input: undefined, emit: async () => {}, services: {}, registry } as any;
    const out: any = await digest.run(
      digest.input.parse({
        results: [
          {
            task: "a/b", score: 0.5, all_pass: false,
            criteria_results: [
              { id: "c1", verdict: "pass", reasoning: "ok" },
              { id: "c2", verdict: "fail", reasoning: "missed the HSR threshold update" },
              { id: "c3", verdict: "fail" },
            ],
            produceCost: 1.1,
          },
          { task: "c/d", score: 1, all_pass: true, criteria_results: [{ id: "c1", verdict: "pass" }] },
          {
            task: "e/f", score: 0, all_pass: false,
            // object-shaped RunResult error + cost only inside runResult —
            // the harvey-candidate-run failure shape
            error: { message: "candidate run failed" },
            gradeError: "eval failed (exit 1)",
            runResult: { runId: "x", status: "failed", output: { cost: 0.33 } },
          },
        ],
        maxCriteria: 1,
      }),
      ctxStub,
    );
    assert.equal(out.n, 3);
    assert.equal(out.meanScore, 0.5);
    // pass-rate fitness: 1/3, 1/1, and 0 (no readable criteria → 0, never 1)
    assert.equal(out.results[0].passRate, 0.333);
    assert.equal(out.results[1].passRate, 1);
    assert.equal(out.results[2].passRate, 0);
    assert.equal(out.meanPassRate, 0.444);
    assert.ok(out.text.includes("mean criteria pass-rate 0.444"));
    assert.equal(out.allPassCount, 1);
    assert.equal(out.results[0].nFailed, 2);
    assert.equal(out.results[0].failed.length, 1);
    assert.equal(out.results[0].failedOmitted, 1);
    assert.equal(out.results[0].cost, 1.1);
    assert.equal(out.results[2].error, "candidate run failed");
    assert.equal(out.results[2].cost, 0.33);
    assert.ok(out.text.includes("✗ c2"));
    assert.ok(out.text.includes("(+1 more failed criteria not shown)"));
    console.log("✔ harvey/digest-results aggregates + formats");

    // 4. artifacts/dir sub
    const artifacts = fileArtifactsCapability(join(base, "artifacts"));
    const dirStep = registry["artifacts/dir"]!;
    const dirCtx = { ...ctxStub, services: { artifacts } };
    const noSub: any = await dirStep.run(dirStep.input.parse({}), dirCtx);
    assert.ok(noSub.path.endsWith("/r"));
    const withSub: any = await dirStep.run(dirStep.input.parse({ sub: "a/b" }), dirCtx);
    assert.ok(withSub.path.endsWith("/r/a/b"));
    await assert.rejects(() => dirStep.run(dirStep.input.parse({ sub: "../nope" }), dirCtx), /relative path/);
    await assert.rejects(() => dirStep.run(dirStep.input.parse({ sub: "/abs" }), dirCtx), /relative path/);
    console.log("✔ artifacts/dir sub: create + traversal guards");

    // 5. the hill-climb loop, driven by a fake optimizer:
    //    rates 0.85, 0.86, 0.86, 0.95 vs baseline 0.90 (improveMargin 0.02,
    //    exploreAfter 2) → gens 0-1 exploit and fail to improve, gens 2-3
    //    get the EXPLORE directive, gen 3 beats best and hits stopPassRate.
    const loop = registry["harvey/evolve-loop"]!;
    const genCalls: any[] = [];
    const rates = [0.85, 0.86, 0.86, 0.95];
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
            summary: `approach ${g}`,
            authorCost: 1,
            digest: { meanPassRate: rates[g], allPassCount: 0, text: `digest ${g}`, results: [{ cost: 2 }] },
          },
        };
      },
    };
    const loopCtx = { ...ctxStub, services: { optimizer: fakeOpt } };
    const loopOut: any = await loop.run(
      loop.input.parse({
        tasks: ["a/b"],
        mission: "m",
        baseline: { meanPassRate: 0.9, text: "baseline digest" },
        candidateName: "harvey-produce-ai",
        maxGenerations: 6,
        stopPassRate: 0.94,
        improveMargin: 0.02,
        exploreAfter: 2,
      }),
      loopCtx,
    );
    assert.equal(genCalls.length, 4); // stopped at gen 3 (0.95 ≥ 0.94), not 6
    assert.equal(loopOut.stopReason, "stopPassRate 0.94 reached");
    assert.equal(loopOut.bestGen, 3);
    assert.equal(loopOut.bestVersion, "v4");
    assert.equal(loopOut.bestPassRate, 0.95);
    assert.equal(loopOut.baselinePassRate, 0.9);
    assert.equal(loopOut.improved, true);
    assert.equal(loopOut.totalKnownCost, 12); // 4 gens × (author 1 + produce 2)
    // directive flip: gens 0-1 exploit, gens 2-3 explore
    assert.ok(!genCalls[0].briefing.includes("GENUINELY DIFFERENT"));
    assert.ok(!genCalls[1].briefing.includes("GENUINELY DIFFERENT"));
    assert.ok(genCalls[2].briefing.includes("GENUINELY DIFFERENT"));
    assert.ok(genCalls[3].briefing.includes("GENUINELY DIFFERENT"));
    // history accumulates: gen 3's briefing lists all prior attempts + baseline anchor
    assert.ok(genCalls[0].briefing.includes("none — this is the first attempt"));
    assert.ok(genCalls[3].briefing.includes("attempt 0"));
    assert.ok(genCalls[3].briefing.includes("attempt 2"));
    assert.ok(genCalls[3].briefing.includes("harvey-produce-ai@v3"));
    assert.ok(genCalls[3].briefing.includes("BEST SO FAR: the baseline itself"));
    console.log("✔ harvey/evolve-loop: best-anchoring, explore flip, history, early stop");

    // two consecutive generation failures abort the loop
    const failOpt = { run: async () => ({ runId: "x", status: "failed", error: { message: "boom" } }) };
    const failOut: any = await loop.run(
      loop.input.parse({
        tasks: ["a/b"],
        mission: "m",
        baseline: { meanPassRate: 0.9, text: "b" },
        candidateName: "c",
        maxGenerations: 6,
      }),
      { ...ctxStub, services: { optimizer: failOpt } },
    );
    assert.equal(failOut.stopReason, "two consecutive generation failures");
    assert.equal(failOut.generations.length, 2);
    assert.equal(failOut.improved, false);
    console.log("✔ harvey/evolve-loop: aborts after consecutive failures");

    console.log("\nALL EVOLVE VALIDATION CHECKS PASSED");
  } finally {
    rmSync(base, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
