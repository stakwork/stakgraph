/**
 * janitor LIVE smoke — the `graph-janitor` engine end to end with a REAL
 * model over a THROWAWAY Neo4j. Seeds the artifacts steps, the engine and
 * the committed `Janitor` tree for real, plants a small `Law` subtree with
 * two overfit Concepts and three clean ones, runs the stock mandate
 * (`Overfit Concept Janitor`) over it through `createStrut`, and checks:
 *
 *  - the run succeeds and starts from Law's ref_id;
 *  - both planted-overfit nodes are flagged `overfit`, the clean doctrine
 *    node is not (or only `low`);
 *  - `visited` is Concepts only, and every finding's ref_id is in it;
 *  - the agent never called a graph WRITE step (the read-only grant held);
 *  - `report.md` + `cleanup-report.json` are in the run's artifacts dir;
 *
 * then prints the findings and the agent's cost, and runs the refusal
 * (`Law` as a mandate → `not_a_janitor:`, no model involved).
 *
 *   ANTHROPIC_API_KEY=… NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=struttest \
 *     STRUT_GRAPH_SEED_ONTOLOGY=1 STRUT_GRAPH_EMBEDDINGS=off npx tsx src/lab/janitor/smoke.ts
 *
 * The graph steps read their connection through the secrets capability
 * (store → env), so NEO4J_* must be in the ENVIRONMENT, not just passed to
 * the workspace. Refuses :7687 (a local jarvis seed). JANITOR_SMOKE_PORT=3100
 * also serves the UI there while the run goes, to watch it. One to five
 * minutes, cents.
 */
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { closeGraphBackends, createStrut, graphWorkspaceFromEnv, type RunEvent } from "strut";
import { seedArtifactSteps } from "../artifacts/seed.js";
import { seedJanitorConcepts, seedJanitorWorkflows } from "./seed.js";

const WORKFLOW = "graph-janitor";
const MANDATE = "Overfit Concept Janitor";
const SEEDED = ["Janitor", MANDATE];
const GRANTED = ["graph/graph-search", "graph/graph-get", "graph/graph-get-batched", "graph/graph-neighbors", "graph/walk", "graph/get-ontology", "graph/get-ontology-type"];

interface Planted {
  name: string;
  description: string;
  docs: string;
  parent: string;
}

// The fixture: two overfit Concepts (a criterion code in sound doctrine;
// grading instructions with a test case's fictional parties as precedent)
// and three clean ones (generalized doctrine, a hub, the hub's child).
const LAW = { name: "Law", description: "Legal knowledge.", docs: "Root of the legal-knowledge Concept tree: doctrine, drafting and analysis skills, and practice areas." };
const CLOSED_RECORD = "Legal Draft Tips: Closed-Record Citation Discipline";
const FACT_TO_LAW = "Legal Analysis Skill: Fact-to-Law Mapping";
const CLEAN_DOCTRINE = "Authority and Citation Discipline";
const CONTRACTS = "Practice Area: Contracts";
const OFFER = "Offer and Acceptance";
const PLANTED: Planted[] = [
  {
    name: CLOSED_RECORD,
    parent: LAW.name,
    description: "Drafting discipline for citing only the materials in a closed record.",
    docs:
      "When drafting from a closed record, cite only the authorities and exhibits the record contains. Never introduce outside case law or statutes; if an authority is needed but absent, flag the gap rather than supplying it from memory. Pin-cite every factual assertion to the exhibit and page it comes from (D1.NEG.02 — Critical). Quote the record verbatim where the exact wording matters, and characterize holdings only as the record's own materials state them.",
  },
  {
    name: FACT_TO_LAW,
    parent: LAW.name,
    description: "How to map the facts of a matter onto the elements of a legal rule.",
    docs:
      "PASS if the answer maps each element of negligence to a specific fact, as in Dr. Whitfield v. Apex Health, where the court held that Apex's failure to update its triage protocol breached the standard of care. FAIL if the answer states the rule without applying it to the facts, or if it omits the Whitfield causation analysis. Full credit requires citing Whitfield for the proposition that a hospital's outdated protocol is a breach per se.",
  },
  {
    name: CLEAN_DOCTRINE,
    parent: LAW.name,
    description: "General principles for supporting legal propositions with authority.",
    docs:
      "Support every legal proposition with the authority that establishes it. Prefer binding authority from the controlling jurisdiction over persuasive authority, and say which is which. Cite the specific page or paragraph that supports the point, quote sparingly and accurately, and confirm an authority is still good law before relying on it. When no authority supports a point, present it as argument, not as settled law.",
  },
  {
    name: CONTRACTS,
    parent: LAW.name,
    description: "Contract law.",
    docs: "Hub for contract-law Concepts: formation, performance, breach and remedies.",
  },
  {
    name: OFFER,
    parent: CONTRACTS,
    description: "How a contract is formed by offer and acceptance.",
    docs:
      "A contract forms when an offer is met by an acceptance that mirrors its terms. An offer creates a power of acceptance in the offeree; that power ends on rejection, counter-offer, revocation before acceptance, lapse of time, or the death or incapacity of either party. Under the common-law mirror-image rule an acceptance that changes the terms is a counter-offer; for sales of goods between merchants, UCC §2-207 relaxes that rule.",
  },
];
const NAMES = [LAW.name, ...PLANTED.map((p) => p.name)];

async function main() {
  const uri = process.env["NEO4J_URI"];
  if (!uri) throw new Error("set NEO4J_URI to a THROWAWAY Neo4j (bolt://localhost:7688): the graph steps read it from the environment");
  if (/:7687\/?$/.test(uri)) throw new Error(`refusing ${uri}: :7687 is the jarvis seed — point NEO4J_URI at the throwaway (bolt://localhost:7688)`);
  if (!process.env["ANTHROPIC_API_KEY"]) throw new Error("set ANTHROPIC_API_KEY: the workflow's default model is `sonnet`");

  const dataDir = mkdtempSync(join(tmpdir(), "janitor-smoke-"));
  const { workspace } = await graphWorkspaceFromEnv(process.env, { dataDir });
  const graph = workspace.graph!;
  const ns = graph.cfg.namespace;
  let strut: Awaited<ReturnType<typeof createStrut>> | undefined;
  try {
    // ── 1. seed the engine + the committed tree ──────────────────────────
    await graph.bolt.run(`MATCH (n:Concept) WHERE n.name IN $names DETACH DELETE n`, { names: [...NAMES, ...SEEDED] });
    await seedArtifactSteps(workspace);
    await seedJanitorWorkflows(workspace);
    await seedJanitorConcepts(workspace);

    // ── 2. plant the fixture ─────────────────────────────────────────────
    const refs: Record<string, string> = {};
    refs[LAW.name] = (await graph.nodes.write({ type: "Concept", data: LAW }, "create", { namespace: ns })).ref_id;
    for (const p of PLANTED) {
      refs[p.name] = (await graph.nodes.write({ type: "Concept", data: { name: p.name, description: p.description, docs: p.docs } }, "create", { namespace: ns })).ref_id;
    }
    for (const p of PLANTED) await graph.edges.write({ edge: "PARENT_OF", source_ref_id: refs[p.parent]!, target_ref_id: refs[p.name]! });
    console.log(`✔ planted ${NAMES.length} Concepts under ${LAW.name} (${refs[LAW.name]})`);

    // ── 3. the instance ──────────────────────────────────────────────────
    const port = Number(process.env["JANITOR_SMOKE_PORT"] ?? 0);
    strut = await createStrut({ workspace, dataDir, serveUi: port > 0, enableChat: false, scheduler: false, claims: false });
    if (port > 0) console.log(`  watching at http://localhost:${await strut.listen(port)}`);

    // ── 4. the real run ──────────────────────────────────────────────────
    const t0 = Date.now();
    const res = await strut.run(WORKFLOW, { concept: MANDATE, start: LAW.name }, { params: { max_steps: 30 } });
    const secs = ((Date.now() - t0) / 1000).toFixed(0);
    const events = await strut.store.getRunEvents(WORKFLOW, res.runId);
    const agentEnd = events.find((e) => e.type === "step.end" && e.stepType === "agent") as (RunEvent & { output?: any }) | undefined;
    const tools = events.filter((e) => e.type === "step.start" && String(e.stepType ?? "").startsWith("tool:")).map((e) => String(e.stepType).slice(5));
    if (res.status !== "success") {
      console.error(`run ${res.runId} ${res.status}:`, res.error);
      for (const e of events.slice(-12)) console.error("  ", JSON.stringify(e).slice(0, 400));
    }
    assert.equal(res.status, "success", JSON.stringify(res.error));
    const out = res.output as Record<string, any>;
    const findings = out["findings"] as Array<Record<string, string>>;
    const visited = out["visited"] as Array<Record<string, string>>;
    const byRef = (name: string) => findings.find((f) => f["ref_id"] === refs[name]);

    console.log(`\n✔ ${WORKFLOW} ran in ${secs}s: visited ${visited.length}, ${findings.length} finding(s), ${tools.length} tool call(s)`);
    console.log(`  agent: steps ${agentEnd?.output?.steps}, cost $${Number(agentEnd?.output?.cost ?? 0).toFixed(4)}, usage ${JSON.stringify(agentEnd?.output?.usage)}`);
    const counts: Record<string, number> = {};
    for (const t of tools) counts[t] = (counts[t] ?? 0) + 1;
    console.log(`  tools: ${Object.entries(counts).map(([t, n]) => `${t}×${n}`).join(", ")}`);
    console.log(`  visited: ${visited.map((v) => v["name"]).join(" | ")}`);
    console.log(`\n  summary: ${out["summary"]}\n`);
    for (const f of findings) {
      console.log(`  [${f["severity"]}] ${f["issue"]} — ${f["name"]} (${f["ref_id"]})`);
      console.log(`    problem:    ${f["problem"]}`);
      console.log(`    evidence:   "${f["evidence"]}"`);
      console.log(`    suggestion: ${f["suggestion"]}\n`);
    }

    // ── 5. assertions ────────────────────────────────────────────────────
    assert.equal(out["start_ref_id"], refs[LAW.name], "start_ref_id is Law's");
    for (const name of [CLOSED_RECORD, FACT_TO_LAW]) {
      assert.equal(byRef(name)?.["issue"], "overfit", `${name} flagged overfit`);
    }
    const clean = byRef(CLEAN_DOCTRINE);
    assert.ok(!clean || clean["severity"] === "low", `${CLEAN_DOCTRINE} not flagged, or only low: ${JSON.stringify(clean)}`);
    for (const v of visited) assert.equal(v["node_type"], "Concept", `visited only Concepts: ${JSON.stringify(v)}`);
    const seen = new Set(visited.map((v) => v["ref_id"]));
    for (const f of findings) assert.ok(seen.has(f["ref_id"]), `finding ${f["name"]} (${f["ref_id"]}) is in visited`);
    const writes = tools.filter((t) => /^graph_(create|edit|register|project)/.test(t));
    assert.deepEqual(writes, [], "no graph write step was called");
    const granted = new Set(GRANTED.map((t) => t.replace(/[^a-zA-Z0-9_]/g, "_")));
    for (const t of tools.filter((t) => t.startsWith("graph_"))) assert.ok(granted.has(t), `graph tool ${t} is one of the granted reads`);
    const art = join(dataDir, "artifacts", res.runId);
    if (!existsSync(join(art, "report.md"))) {
      console.error(`artifacts/${res.runId}: ${existsSync(art) ? readdirSync(art).join(", ") : "(no dir)"}; the agent's file tool calls:`);
      for (const e of events.filter((e) => /^tool:(bash|str_replace)/.test(String(e.stepType ?? "")))) {
        const { type, path, input, output, error } = e as any;
        console.error(`  ${type} ${path}`, JSON.stringify(input ?? output ?? error).slice(0, 600));
      }
    }
    assert.ok(existsSync(join(art, "report.md")), "report.md written");
    const saved = JSON.parse(readFileSync(join(art, "cleanup-report.json"), "utf-8"));
    assert.ok(Array.isArray(saved.findings), "cleanup-report.json parses and carries findings");
    console.log(`✔ assertions: both overfit nodes flagged, ${CLEAN_DOCTRINE} ${clean ? "low" : "not flagged"}, Concepts only, no writes, report.md + cleanup-report.json`);
    // Expectations the mandate should meet, reported rather than asserted.
    const fact = byRef(FACT_TO_LAW);
    if (fact?.["severity"] !== "high") console.log(`  ⚠ ${FACT_TO_LAW} is ${fact?.["severity"]}, expected high (grading rule stated as doctrine, fake precedent)`);
    for (const name of [CONTRACTS, OFFER]) if (byRef(name)) console.log(`  ⚠ ${name} flagged, expected clean: ${JSON.stringify(byRef(name))}`);
    for (const name of NAMES.slice(1)) if (!seen.has(refs[name])) console.log(`  ⚠ ${name} never visited`);

    // ── 6. the refusal: Law is a Concept, but not under Janitor ──────────
    const refused = await strut.run(WORKFLOW, { concept: LAW.name, start: LAW.name });
    assert.equal(refused.status, "error");
    const code = refused.error!.message.split("\n").find((l) => l.startsWith("not_a_janitor:"));
    assert.ok(code, `a line of error.message starts with not_a_janitor: — ${refused.error!.message}`);
    console.log(`✔ ${LAW.name} as a mandate refused: ${code}`);
    console.log("\njanitor smoke: all good");
  } finally {
    await strut?.close();
    await graph.bolt.run(`MATCH (n:Concept) WHERE n.name IN $names DETACH DELETE n`, { names: [...NAMES, ...SEEDED] }).catch(() => {});
    await closeGraphBackends();
    rmSync(dataDir, { recursive: true, force: true });
  }
}

main().catch((e) => {
  console.error("janitor smoke FAILED:", e);
  process.exit(1);
});
