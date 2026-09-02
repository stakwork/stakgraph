/**
 * Offline smoke test for the harvey-deliver pipeline scaffold — no vein
 * server, no Neo4j, no Jarvis, no LLM. Verifies:
 *   1. seeding: the deliver steps + workflows publish into a temp workspace
 *      and buildRegistry discovers every step from disk;
 *   2. every deliver workflow YAML parses and publishes;
 *   3. the pure/plumbing steps' logic against fixtures (fail-open /
 *      fail-soft / hard-gate semantics). The graph/* steps the pipeline
 *      writes through are vein lib steps with their own live test
 *      (vein/src/steps/lib/graph/graph-steps.test.ts).
 *
 * Run: npx tsx src/lab/harvey/deliver-smoke.ts
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { WorkspaceManager, buildRegistry, type StepContext } from "vein";
import { seedHarveySteps, seedHarveyWorkflows } from "./seed.js";

async function main() {
  const dir = mkdtempSync(join(process.cwd(), ".harvey-deliver-smoke-"));
  try {
    // ── 1. seed + discover ───────────────────────────────────────────────
    const workspace = new WorkspaceManager(dir);
    await seedHarveySteps(workspace);
    await seedHarveyWorkflows(workspace);
    const { registry } = await buildRegistry(workspace.path);

    const expectedSteps = [
      "harvey/normalize-documents",
      "harvey/graph-sub-agent",
      "harvey/ingest-state",
      "harvey/drafter-plan",
      "harvey/validate-deliverables",
      "harvey/filter-contested",
      "harvey/aggregate-scores",
      "harvey/merge-disputes",
      "harvey/build-eval-chain",
      "harvey/criterion-refs",
      "harvey/generate-docx",
      "harvey/generate-xlsx",
      // graph/* are vein LIB steps — discovered from the engine, not seeded.
      "graph/register-namespace",
      "graph/create-node",
      "graph/create-batch-triplet",
    ];
    for (const t of expectedSteps) assert.ok(registry[t], `registry missing ${t}`);
    console.log(`✔ seeded + discovered ${expectedSteps.length} deliver steps`);

    const expectedWorkflows = [
      "harvey-deliver",
      "harvey-ingest-doc",
      "harvey-knowledge",
      "harvey-draft",
      "harvey-judge-criterion",
      "harvey-dispute-criterion",
      "harvey-score",
    ];
    for (const name of expectedWorkflows) {
      const wf = await workspace.getWorkflow(name);
      assert.ok(wf?.steps?.length, `workflow ${name} missing/empty`);
      // @@include expansion happened at seed time — no markers survive.
      assert.ok(!JSON.stringify(wf).includes("@@include"), `workflow ${name} has unexpanded @@include`);
    }
    // Spot-check the splice actually carried prompt bodies in: the persona
    // text lands in harvey-draft's params, the ingestion prompt in the
    // ingest-doc agent step.
    const draftWf = JSON.stringify(await workspace.getWorkflow("harvey-draft"));
    assert.ok(draftWf.includes("experienced practicing attorney"), "persona not spliced into harvey-draft");
    const ingestWf = JSON.stringify(await workspace.getWorkflow("harvey-ingest-doc"));
    assert.ok(ingestWf.includes("ledger of assertions"), "ingestion prompt not spliced into harvey-ingest-doc");
    console.log(`✔ ${expectedWorkflows.length} deliver workflows published + prompts spliced`);

    const run = (type: string, input: unknown, ctx?: StepContext) =>
      registry[type].run(registry[type].input.parse(input), ctx ?? ({} as StepContext));

    // ── 2. normalize-documents ───────────────────────────────────────────
    const docsDir = join(dir, "documents");
    mkdirSync(docsDir, { recursive: true });
    writeFileSync(join(docsDir, "agreement.docx"), "x");
    writeFileSync(join(docsDir, "model.xlsx"), "x");

    let out: any = await run("harvey/normalize-documents", {
      task: "Corporate-MA/review-data-room",
      documentsDir: docsDir,
      documents: ["agreement.docx", "model.xlsx"],
    });
    assert.equal(out.namespace, "corporate-ma-review-data-room");
    assert.equal(out.count, 2);
    assert.equal(out.hasSpreadsheets, true);
    assert.equal(out.documents[1].isSpreadsheet, true);
    assert.ok(out.documents[0].path.endsWith("agreement.docx"));

    await assert.rejects(
      () => run("harvey/normalize-documents", { task: "a/b", documentsDir: docsDir, documents: ["ghost.pdf"] }),
      /missing/,
    );
    await assert.rejects(
      () => run("harvey/normalize-documents", { task: "a/b", documentsDir: docsDir, documents: [] }),
      /no input documents/,
    );
    out = await run("harvey/normalize-documents", {
      task: "a/b",
      documentsDir: docsDir,
      documents: [],
      requireDocuments: false,
    });
    assert.equal(out.count, 0);
    console.log("✔ normalize-documents (classify / missing hard-fail / doc-less opt-out)");

    // ── 3. ingest-state (completion-marker gate) ─────────────────────────
    assert.equal(await run("harvey/ingest-state", { node: { properties: { status: "ingested" } } }), false);
    assert.equal(await run("harvey/ingest-state", { node: { properties: { status: "other" } } }), true);
    assert.equal(await run("harvey/ingest-state", { node: { properties: {} } }), true);
    assert.equal(await run("harvey/ingest-state", { node: "HTTP 500: boom" }), true);
    assert.equal(await run("harvey/ingest-state", { node: null }), true);
    console.log("✔ ingest-state (marker gate, tolerant of errors)");

    // ── 4. drafter-plan ──────────────────────────────────────────────────
    out = await run("harvey/drafter-plan", { deliverables: ["memo.docx", "markup.docx"], drafters: 2 });
    assert.equal(out.basename, "memo");
    assert.deepEqual(out.drafts.map((d: any) => d.dir), ["draft_1", "draft_2"]);
    assert.deepEqual(out.drafts[0].files, ["draft_1/memo.docx", "draft_1/markup.docx"]);
    assert.deepEqual(out.outputFiles, ["output/memo.docx", "output/markup.docx"]);
    assert.equal(out.critiqueFiles.length, 4);
    assert.ok(out.critiqueFiles.includes("critiques/critique-doctrine.md"));
    console.log("✔ drafter-plan");

    // ── 5. validate-deliverables (hard gate) ─────────────────────────────
    const outDir = join(dir, "output");
    mkdirSync(outDir, { recursive: true });
    writeFileSync(join(outDir, "memo.docx"), "content");
    out = await run("harvey/validate-deliverables", {
      outputDir: outDir,
      deliverables: ["memo.docx"],
      rubric: [{ id: "c1", deliverables: ["memo.docx"] }],
    });
    assert.equal(out.ok, true);
    assert.equal(out.files[0].file, "memo.docx");
    await assert.rejects(
      () =>
        run("harvey/validate-deliverables", {
          outputDir: outDir,
          deliverables: ["memo.docx"],
          rubric: [{ id: "c1", deliverables: ["missing.docx"] }],
        }),
      /missing\.docx \(missing\)/,
    );
    writeFileSync(join(outDir, "empty.docx"), "");
    await assert.rejects(
      () => run("harvey/validate-deliverables", { outputDir: outDir, deliverables: ["empty.docx"], rubric: [] }),
      /empty/,
    );
    console.log("✔ validate-deliverables (exact-name hard gate)");

    // ── 6. filter-contested (fail open) ──────────────────────────────────
    const rubric = [
      { id: "c1", title: "A", match_criteria: "...", deliverables: ["memo.docx"] },
      { id: "c2", title: "B", match_criteria: "...", deliverables: ["memo.docx"] },
    ];
    out = await run("harvey/filter-contested", {
      rubric,
      evalsetId: "slug",
      requirements: [
        { ref_id: "rq1", properties: { id: "slug-c1", contested: true } },
        { ref_id: "rq2", properties: { id: "slug-c2", contested: false } },
      ],
    });
    assert.deepEqual(out.dropped, ["c1"]);
    assert.equal(out.kept, 1);
    // fail-open: garbage requirements filter nothing
    out = await run("harvey/filter-contested", { rubric, requirements: "HTTP 500: boom" });
    assert.equal(out.kept, 2);
    out = await run("harvey/filter-contested", { rubric, requirements: { error: "packed" } });
    assert.equal(out.kept, 2);
    console.log("✔ filter-contested (drop + fail-open)");

    // ── 7. aggregate-scores (zip; null = honest fail) ────────────────────
    out = await run("harvey/aggregate-scores", {
      rubric,
      results: [
        { object: { verdict: "pass", reasoning: "ok" }, cost: 0.1 },
        { error: "judge blew up" },
      ],
      judge_model: "claude-sonnet-5",
    });
    assert.equal(out.n_total, 2);
    assert.equal(out.n_passed, 1);
    assert.equal(out.all_pass, false);
    assert.equal(out.score, 1);
    assert.equal(out.pass_rate, 0.5);
    assert.equal(out.criteria_results[1].verdict, "fail");
    assert.match(out.criteria_results[1].reasoning, /judge error/);
    assert.equal(out.failed.length, 1);
    assert.equal(out.failed[0].id, "c2");
    assert.equal(out.failed[0].match_criteria, "...");
    assert.equal(out.judgeCost, 0.1);
    await assert.rejects(
      () => run("harvey/aggregate-scores", { rubric, results: [null] }),
      /refusing to zip/,
    );
    console.log("✔ aggregate-scores (zip / honest fail / length guard)");

    // ── 8. merge-disputes (annotate failed only; fail-soft) ──────────────
    const criteria_results = out.criteria_results;
    let merged: any = await run("harvey/merge-disputes", {
      criteria_results,
      failed: out.failed,
      disputes: [
        {
          object: {
            id: "c2",
            flagged: true,
            flag_basis: "criterion_validity",
            contested: true,
            llm_flag_reason: "**Criterion Validity:** defective…",
            document_excerpt: "quoted passage",
          },
        },
      ],
      criterionRefs: [
        { criterion_id: "c1", ref_id: "cr1" },
        { criterion_id: "c2", ref_id: "cr2" },
      ],
      requirements: [{ ref_id: "rq2", properties: { id: "slug-c2" } }],
      evalsetId: "slug",
    });
    assert.equal(merged.criteria_results[0].flagged, undefined); // pass untouched
    assert.equal(merged.criteria_results[1].flagged, true);
    assert.equal(merged.criteria_results[1].flag_basis, "criterion_validity");
    assert.equal(merged.criteria_results[1].llm_flag_reason, "**Criterion Validity:** defective…");
    assert.equal(merged.criteria_results[1].document_excerpt, "quoted passage");
    assert.equal(merged.flagged_count, 1);
    assert.equal(merged.contested_count, 1);
    assert.deepEqual(merged.contested_requirements, [{ criterion_id: "c2", ref_id: "rq2" }]);
    // the write-back list carries the CriterionResult ref
    assert.equal(merged.annotations.length, 1);
    assert.equal(merged.annotations[0].ref_id, "cr2");
    assert.equal(merged.annotations[0].contested, true);
    // fail-soft: garbage disputes annotate nothing
    merged = await run("harvey/merge-disputes", { criteria_results, failed: out.failed, disputes: "boom" });
    assert.equal(merged.flagged_count, 0);
    assert.equal(merged.criteria_results[1].flagged, undefined);
    assert.deepEqual(merged.annotations, []);
    console.log("✔ merge-disputes (left-join + refs + fail-soft + contested)");

    // ── 8b. criterion-refs (slot × batch-result zip, fail-soft) ──────────
    let refs: any = await run("harvey/criterion-refs", {
      slots: [
        { criterion_id: "c1", index: 2 },
        { criterion_id: "c2", index: 4 },
      ],
      record: {
        results: [
          { target_ref_id: "t0" },
          { target_ref_id: "o0" },
          { target_ref_id: "cr1" },
          { target_ref_id: "x" },
          { error: "edge write failed", target_ref_id: "cr2" },
        ],
      },
    });
    assert.deepEqual(refs, [{ criterion_id: "c1", ref_id: "cr1" }]); // errored slot dropped
    refs = await run("harvey/criterion-refs", { slots: "garbage", record: { error: "record failed" } });
    assert.deepEqual(refs, []);
    console.log("✔ criterion-refs (zip + fail-soft)");

    // ── 9. build-eval-chain (ontology shape) ─────────────────────────────
    const ctx = { runId: "run42" } as StepContext;
    const chain: any = await run(
      "harvey/build-eval-chain",
      {
        evalsetId: "slug",
        task: "a/b",
        scores: out,
        criteria_results: [
          { id: "c1", criterion_id: "c1", title: "A", verdict: "pass", reasoning: "ok" },
          {
            id: "c2", criterion_id: "c2", title: "B", verdict: "fail", reasoning: "no",
            flagged: true, llm_flag_reason: "r", contested: true,
          },
        ],
        judge_model: "claude-sonnet-5",
      },
      ctx,
    );
    assert.equal(chain.trigger_id, "trigger-run42");
    assert.equal(chain.output_id, "output-run42");
    // 2 spine + 2 per criterion
    assert.equal(chain.triplets.length, 2 + 2 * 2);
    const [setTrig, trigOut, critA, reqA] = chain.triplets;
    assert.equal(setTrig.edge_type, "HAS_TRIGGER");
    assert.equal(setTrig.source_type, "EvalSet");
    assert.equal(trigOut.edge_type, "HAS_OUTPUT");
    assert.equal(trigOut.target_data.result, "fail");
    assert.equal(trigOut.target_data.n_passed, 1);
    assert.equal(critA.edge_type, "HAS_CRITERION_RESULT");
    assert.equal(critA.target_data.id, "crit-run42-c1");
    // The criterion side carries the FULL output node_data (identical to the
    // spine's) so the batch write's dedup cache resolves it once — a bare
    // { id } missed the cache and failed schema validation live.
    assert.equal(critA.source_data.id, "output-run42");
    assert.equal(critA.source_data.result, trigOut.target_data.result);
    assert.equal(reqA.source_type, "EvalRequirement");
    assert.equal(reqA.source_data.id, "slug-c1");
    const critB = chain.triplets[4];
    assert.equal(critB.target_data.flagged, true);
    assert.equal(critB.target_data.contested, true);
    // criterionSlots name each HAS_CRITERION_RESULT triplet's index
    assert.deepEqual(chain.criterionSlots, [
      { criterion_id: "c1", index: 2 },
      { criterion_id: "c2", index: 4 },
    ]);
    console.log("✔ build-eval-chain (unified eval chain triplets + slots)");

    // ── 9b. generate-docx / generate-xlsx (skipped if tools absent) ──────
    const artDir = join(dir, "artifacts-run");
    mkdirSync(artDir, { recursive: true });
    const artCtx = {
      runId: "smoke",
      services: { artifacts: { dir: async () => artDir } },
    } as unknown as StepContext;
    const { execSync } = await import("node:child_process");
    const have = (cmd: string) => {
      try {
        execSync(cmd, { stdio: "ignore" });
        return true;
      } catch {
        return false;
      }
    };
    if (have("pandoc --version")) {
      const gen: any = await run(
        "harvey/generate-docx",
        { filename: "draft_1/memo.docx", markdown: "# Memo\n\nHello." },
        artCtx,
      );
      assert.ok(gen.bytes > 0, `generate-docx failed: ${JSON.stringify(gen)}`);
      assert.ok(gen.path.endsWith("draft_1/memo.docx"));
      // containment guard
      const esc: any = await run("harvey/generate-docx", { filename: "../out.docx", markdown: "x" }, artCtx);
      assert.match(String(esc), /relative path/);
      console.log("✔ generate-docx (pandoc + containment)");
    } else {
      console.log("· generate-docx skipped (no pandoc on PATH)");
    }
    if (have("python3 -c 'import openpyxl'")) {
      const xlsx: any = await run(
        "harvey/generate-xlsx",
        { filename: "output/model.xlsx", sheets: [{ name: "FACTS", rows: [["label", "value"], ["a", 1], ["sum", "=SUM(B2:B2)"]] }] },
        artCtx,
      );
      assert.ok(xlsx.bytes > 0, `generate-xlsx failed: ${JSON.stringify(xlsx)}`);
      console.log("✔ generate-xlsx (openpyxl)");
    } else {
      console.log("· generate-xlsx skipped (no python3/openpyxl)");
    }

    console.log("\nALL DELIVER SMOKE CHECKS PASSED");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
