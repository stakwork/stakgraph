/**
 * Offline smoke test for the harvey-deliver pipeline scaffold — no vein
 * server, no Neo4j, no Jarvis, no LLM. Verifies:
 *   1. seeding: the deliver steps + workflows publish into a temp workspace
 *      and buildRegistry discovers every step from disk;
 *   2. every deliver workflow YAML parses and publishes;
 *   3. the pure/plumbing steps' logic against fixtures (fail-open /
 *      fail-soft / hard-gate semantics), plus jarvis/register-namespace
 *      against a fake ctx.services.http.
 *
 * Run: npx tsx src/lab/harvey/deliver-smoke.ts
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { WorkspaceManager, buildRegistry, type StepContext, type HttpResponse } from "vein";
import { seedHarveySteps, seedHarveyWorkflows } from "./seed.js";
import { seedJarvisSteps } from "../jarvis/seed.js";

async function main() {
  const dir = mkdtempSync(join(process.cwd(), ".harvey-deliver-smoke-"));
  try {
    // ── 1. seed + discover ───────────────────────────────────────────────
    const workspace = new WorkspaceManager(dir);
    await seedHarveySteps(workspace);
    await seedJarvisSteps(workspace);
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
      "jarvis/register-namespace",
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
    }
    console.log(`✔ ${expectedWorkflows.length} deliver workflows published + parse`);

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
        { ref_id: "rq1", properties: { id: "slug/c1", contested: true } },
        { ref_id: "rq2", properties: { id: "slug/c2", contested: false } },
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
      disputes: [{ object: { flagged: true, reason: "actually satisfied", contested: true } }],
      requirements: [{ ref_id: "rq2", properties: { id: "slug/c2" } }],
      evalsetId: "slug",
    });
    assert.equal(merged.criteria_results[0].flagged, undefined); // pass untouched
    assert.equal(merged.criteria_results[1].flagged, true);
    assert.equal(merged.criteria_results[1].llm_flag_reason, "actually satisfied");
    assert.equal(merged.flagged_count, 1);
    assert.equal(merged.contested_count, 1);
    assert.deepEqual(merged.contested_requirements, [{ criterion_id: "c2", ref_id: "rq2" }]);
    // fail-soft: garbage disputes annotate nothing
    merged = await run("harvey/merge-disputes", { criteria_results, failed: out.failed, disputes: "boom" });
    assert.equal(merged.flagged_count, 0);
    assert.equal(merged.criteria_results[1].flagged, undefined);
    console.log("✔ merge-disputes (left-join + fail-soft + contested refs)");

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
    assert.equal(reqA.source_type, "EvalRequirement");
    assert.equal(reqA.source_data.id, "slug/c1");
    const critB = chain.triplets[4];
    assert.equal(critB.target_data.flagged, true);
    assert.equal(critB.target_data.contested, true);
    console.log("✔ build-eval-chain (unified eval chain triplets)");

    // ── 10. jarvis/register-namespace (fake http) ────────────────────────
    const calls: Array<{ url: string; opts: any }> = [];
    const makeCtx = (routes: Array<{ match: (u: string, o: any) => boolean; body: unknown; status?: number }>) =>
      ({
        runId: "smoke",
        path: "smoke",
        scope: {},
        input: undefined,
        emit: async () => {},
        services: {
          http: async (url: string, opts: any = {}): Promise<HttpResponse> => {
            calls.push({ url, opts });
            for (const r of routes) {
              if (r.match(url, opts)) {
                return { status: r.status ?? 200, ok: (r.status ?? 200) < 300, headers: {}, body: r.body };
              }
            }
            return { status: 404, ok: false, headers: {}, body: "no route" };
          },
          secrets: {
            get: async (n: string) => (n === "JARVIS_URL" ? "http://jarvis.fake" : n === "API_TOKEN" ? "tok" : undefined),
          },
        },
      }) as unknown as StepContext;

    out = await run(
      "jarvis/register-namespace",
      { namespace: "slug" },
      makeCtx([{ match: (u, o) => u.endsWith("/namespace") && o.method === "POST", body: { ok: true } }]),
    );
    assert.deepEqual(out, { namespace: "slug", registered: true });
    assert.equal(calls[calls.length - 1].opts.headers["X-Api-Token"], "tok");

    // duplicate POST fails, but the list confirms it exists → success
    out = await run(
      "jarvis/register-namespace",
      { namespace: "slug" },
      makeCtx([
        { match: (u, o) => u.endsWith("/namespace") && o.method === "POST", body: "exists", status: 400 },
        { match: (u, o) => u.endsWith("/namespace") && !o.method, body: { data: { namespace: ["other", "slug"] } } },
      ]),
    );
    assert.equal(out.registered, true);
    assert.equal(out.alreadyExisted, true);

    // hard failure surfaces as an error string
    out = await run(
      "jarvis/register-namespace",
      { namespace: "slug" },
      makeCtx([{ match: (u, o) => u.endsWith("/namespace") && o.method === "POST", body: "boom", status: 500 }]),
    );
    assert.match(String(out), /HTTP 500/);
    console.log("✔ register-namespace (create / already-exists / hard fail)");

    console.log("\nALL DELIVER SMOKE CHECKS PASSED");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
