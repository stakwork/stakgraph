/**
 * janitor BUILDER smoke — can the AI builder add a janitor from a plain
 * request, by reading the convention off the graph? The real
 * `createLabStrut` (every lab workflow seeded; the builder's prompt carries
 * the `Workflow Builder` page, which lists `Janitor` and says nothing about
 * how one is made) over a THROWAWAY Neo4j, a planted Law subtree with two
 * near-duplicate pairs, then one chat message:
 *
 *   "Make me a janitor that flags near-duplicate Concepts under Law, and have
 *    it run every day at 6am."
 *
 * Waits for the chat to settle (turn-end callbacks: a detached run wakes it
 * again), then reports the builder's tool calls — did it OPEN the Janitor
 * page before acting? — and checks what it left
 * behind: a new Concept under Janitor with docs, the PARENT_OF edge, an
 * automation on graph-janitor with { concept, start: "Law" }, and — if it ran
 * the workflow — what that run found. Nothing is asserted: a model's route
 * varies; the report says which of the three steps it got right.
 *
 *   ANTHROPIC_API_KEY=… NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=struttest \
 *     STRUT_GRAPH_SEED_ONTOLOGY=1 STRUT_GRAPH_EMBEDDINGS=off NO_DB=true \
 *     npx tsx src/lab/janitor/builder-smoke.ts
 */
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { closeGraphBackends } from "strut";
import { createLabStrut, type LabServices } from "../createLabStrut.js";

const MESSAGE = process.env["JANITOR_BUILDER_MESSAGE"] ?? "Make me a janitor that flags near-duplicate Concepts under Law, and have it run every day at 6am.";
const TIMEOUT_MS = 20 * 60_000;
const FIXTURE: Array<{ name: string; parent: string; description: string; docs: string }> = [
  { name: "Offer and Acceptance", parent: "Law", description: "How a contract is formed by offer and acceptance.", docs: "A contract forms when an offer is met by an acceptance that mirrors its terms. An offer ends on rejection, counter-offer, revocation before acceptance, or lapse of time." },
  { name: "Contract Formation: Offer & Acceptance", parent: "Law", description: "Offer and acceptance in contract formation.", docs: "Formation requires an offer and a matching acceptance; an acceptance that varies the terms is a counter-offer (the mirror-image rule). Offers lapse, can be revoked before acceptance, and end on rejection." },
  { name: "Authority and Citation Discipline", parent: "Law", description: "Supporting legal propositions with authority.", docs: "Support every proposition with the authority that establishes it; prefer binding over persuasive authority and pin-cite the page that supports the point." },
  { name: "Citing Legal Authority", parent: "Law", description: "How to cite authority for a legal claim.", docs: "Back each legal claim with the case or statute that establishes it, preferring binding authority, and cite the specific page or paragraph relied on." },
  { name: "Practice Area: Torts", parent: "Law", description: "Tort law.", docs: "Hub for tort-law Concepts: negligence, intentional torts, strict liability and damages." },
];
const NAMES = ["Law", ...FIXTURE.map((f) => f.name)];
const clip = (v: unknown, n = 220) => {
  const s = typeof v === "string" ? v : JSON.stringify(v);
  return s && s.length > n ? `${s.slice(0, n)}…` : s;
};

async function main() {
  const uri = process.env["NEO4J_URI"];
  if (!uri || /:7687\/?$/.test(uri)) throw new Error("set NEO4J_URI to the THROWAWAY Neo4j (bolt://localhost:7688), never :7687");
  if (!process.env["ANTHROPIC_API_KEY"]) throw new Error("set ANTHROPIC_API_KEY");
  process.env["STRUT_SCHEDULER"] = "0"; // store the automation, never fire it

  // Turn-end callbacks: how a host waits for a chat to settle.
  const posts: any[] = [];
  let wake: () => void = () => {};
  const server = createServer((req, res) => {
    let body = "";
    req.on("data", (d) => (body += d));
    req.on("end", () => {
      try { posts.push(JSON.parse(body)); } catch { /* ignore */ }
      res.end("ok");
      wake();
    });
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const callback = { url: `http://127.0.0.1:${(server.address() as AddressInfo).port}/turn` };

  const dir = mkdtempSync(join(tmpdir(), "janitor-builder-"));
  const strut = await createLabStrut({ workspacePath: dir, serveUi: false, services: {} as LabServices });
  const graph = strut.workspace.graph!;
  const ns = graph.cfg.namespace;
  const before = new Set((await graph.bolt.run(`MATCH (:Concept {name: 'Janitor'})-[:PARENT_OF]->(c:Concept) RETURN c.ref_id AS r`)).map((r) => String(r.r)));
  // Whatever the builder publishes is removed at the end, so the next run starts from the seeded lab only.
  const seeded = new Set((await strut.workspace.listWorkflows()).map((w) => w.name));
  const autosOf = async () => ((await (await strut.app.fetch(new Request("http://lab/automations"))).json()) as any).automations ?? [];
  const autosBefore = new Set((await autosOf()).map((a: any) => a.id));
  try {
    // ── the fixture ──────────────────────────────────────────────────────
    await graph.bolt.run(`MATCH (n:Concept) WHERE n.name IN $names DETACH DELETE n`, { names: NAMES });
    const refs: Record<string, string> = {};
    refs["Law"] = (await graph.nodes.write({ type: "Concept", data: { name: "Law", description: "Legal knowledge.", docs: "Root of the legal-knowledge Concept tree." } }, "create", { namespace: ns })).ref_id;
    for (const f of FIXTURE) refs[f.name] = (await graph.nodes.write({ type: "Concept", data: { name: f.name, description: f.description, docs: f.docs } }, "create", { namespace: ns })).ref_id;
    for (const f of FIXTURE) await graph.edges.write({ edge: "PARENT_OF", source_ref_id: refs[f.parent]!, target_ref_id: refs[f.name]! });
    console.log(`✔ lab strut up (${(await strut.workspace.listWorkflows()).length} workflows seeded), Law subtree planted`);

    // ── the request ──────────────────────────────────────────────────────
    const t0 = Date.now();
    const r = await strut.app.fetch(new Request("http://lab/chat", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ message: MESSAGE, callback }) }));
    const { chatId } = (await r.json()) as { chatId: string };
    console.log(`→ ${MESSAGE}\n  chat ${chatId} (${r.status})`);
    const settled = () => posts.some((p) => p.chatId === chatId && p.settled === true && (p.event === "settled" || p.event === "turn.end"));
    while (!settled() && Date.now() - t0 < TIMEOUT_MS) {
      await new Promise<void>((res) => { wake = res; setTimeout(res, 15_000); });
      const last = posts.at(-1);
      if (last) console.log(`  … ${Math.round((Date.now() - t0) / 1000)}s: ${last.event} turn ${last.turn} (${last.trigger ?? ""}) settled=${last.settled}${last.parked ? " PARKED" : ""}`);
    }
    console.log(`\n${settled() ? "✔ settled" : "✖ timed out"} after ${Math.round((Date.now() - t0) / 1000)}s, ${posts.filter((p) => p.event === "turn.end").length} turn(s)`);

    // ── what the builder did ─────────────────────────────────────────────
    const chat = (await (await strut.app.fetch(new Request(`http://lab/chat/${chatId}`))).json()) as { messages: any[] };
    const calls: Array<{ toolName: string; input: unknown }> = chat.messages
      .filter((m) => m.role === "assistant" && Array.isArray(m.content))
      .flatMap((m) => m.content.filter((p: any) => p.type === "tool-call"));
    // graph_get, or the same step through run_step.
    const opened = calls.findIndex((c) => (c.toolName === "graph_get" || /graph\/graph-get/.test(JSON.stringify(c.input))) && /"name":\s*"Janitor"/.test(JSON.stringify(c.input)));
    const wrote = calls.findIndex((c) => /graph\/create-node|graph\/create-triplet|set_automation|create_workflow/.test(`${c.toolName} ${JSON.stringify(c.input)}`));
    console.log(
      opened < 0
        ? "\n✖ the builder never opened the Janitor page"
        : `\n✔ the builder opened the Janitor page (call ${opened + 1} of ${calls.length})${wrote >= 0 && wrote < opened ? " ⚠ AFTER it had started writing" : ", before writing anything"}`,
    );
    if (calls.some((c) => c.toolName === "create_workflow" || c.toolName === "edit_workflow")) console.log("⚠ the builder published a workflow: the convention says never to");
    console.log("\nbuilder tool calls:");
    for (const m of chat.messages) {
      if (m.role === "user") console.log(`  [user] ${clip(typeof m.content === "string" ? m.content : m.content, 160)}`);
      if (m.role === "tool" && Array.isArray(m.content)) {
        for (const p of m.content) {
          const o = p.output;
          if (String(o?.type ?? "").startsWith("error") || /"error"|^error|failed/i.test(clip(o?.value, 400) ?? "")) console.log(`    ✖ ${p.toolName} → ${clip(o?.value, 240)}`);
        }
      }
      if (m.role !== "assistant" || !Array.isArray(m.content)) continue;
      for (const p of m.content) if (p.type === "tool-call") console.log(`  ${p.toolName}(${clip(p.input)})`);
    }
    const lastTurn = posts.filter((p) => p.chatId === chatId && p.event === "turn.end").at(-1);
    console.log(`\nbuilder's last words:\n${lastTurn?.text ?? "(none)"}\n`);

    // ── what it left behind ──────────────────────────────────────────────
    const kids = (await graph.bolt.run(
      `MATCH (:Concept {name: 'Janitor'})-[:PARENT_OF]->(c:Concept) RETURN c.ref_id AS ref_id, c.name AS name, c.description AS description, c.docs AS docs`,
    )).filter((k) => !before.has(String(k.ref_id)));
    const loose = (await graph.bolt.run(
      `MATCH (c:Concept) WHERE NOT c.name IN $names AND NOT c.name IN ['Workflow Builder', 'Janitor', 'Overfit Concept Janitor'] AND NOT (:Concept {name: 'Janitor'})-[:PARENT_OF]->(c) RETURN c.name AS name`,
      { names: NAMES },
    )).map((x) => x.name);
    console.log(`(1)+(2) new mandate(s) under Janitor: ${kids.length ? "" : "NONE"}`);
    for (const k of kids) console.log(`  ✔ "${k.name}" — ${k.description}\n    docs: ${clip(k.docs, 600)}`);
    if (loose.length) console.log(`  ⚠ Concepts created but NOT under Janitor: ${loose.join(", ")}`);
    const autos = (await (await strut.app.fetch(new Request("http://lab/automations?workflow=graph-janitor"))).json()) as any;
    const list: any[] = autos.automations ?? [];
    console.log(`(3) automations on graph-janitor: ${list.length ? "" : "NONE"}`);
    for (const a of list) console.log(`  ${a.enabled ? "✔" : "✖ disabled"} "${a.name}" ${JSON.stringify(a.trigger)} input=${JSON.stringify(a.input)}${kids.some((k) => k.name === a.input?.concept) ? " (names the new mandate)" : " ⚠ concept is not the new mandate"}${a.input?.start === "Law" ? "" : " ⚠ start is not Law"}`);
    const runs = await strut.store.listRuns("graph-janitor");
    for (const run of runs) {
      const sum = await strut.store.getRunSummary("graph-janitor", run);
      const out = (sum as any)?.output;
      console.log(`run ${run}: ${sum?.status}${sum?.error ? ` — ${clip(sum.error.message, 300)}` : ""}`);
      for (const f of out?.findings ?? []) console.log(`  [${f.severity}] ${f.issue} — ${f.name}: ${clip(f.problem, 200)}`);
    }
    const verdict = kids.length > 0 && list.some((a) => kids.some((k) => k.name === a.input?.concept) && a.input?.start === "Law" && a.enabled);
    console.log(`\n${verdict ? "✔ the builder added a working janitor" : "✖ the builder did NOT finish a janitor"}`);
  } finally {
    for (const a of await autosOf().catch(() => [])) {
      if (autosBefore.has(a.id)) continue;
      const r = await strut.app.fetch(new Request(`http://lab/workflows/${encodeURIComponent(a.workflow)}/automations/${a.id}`, { method: "DELETE" }));
      console.log(`  removed the builder's automation ${a.workflow}/${a.id} (${r.status})`);
    }
    for (const w of await strut.workspace.listWorkflows().catch(() => [])) {
      if (seeded.has(w.name)) continue;
      const r = await strut.app.fetch(new Request(`http://lab/workflows/${encodeURIComponent(w.name)}`, { method: "DELETE" }));
      console.log(`  removed the builder's workflow ${w.name} (${r.status})`);
    }
    await graph.bolt.run(`MATCH (n:Concept) WHERE n.name IN $names DETACH DELETE n`, { names: NAMES }).catch(() => {});
    await graph.bolt.run(`MATCH (:Concept {name: 'Janitor'})-[:PARENT_OF]->(c:Concept) WHERE NOT c.ref_id IN $keep DETACH DELETE c`, { keep: [...before] }).catch(() => {});
    await strut.close();
    await closeGraphBackends();
    server.close();
    rmSync(dir, { recursive: true, force: true });
  }
}

main().then(() => process.exit(0), (e) => {
  console.error("janitor builder smoke FAILED:", e);
  process.exit(1);
});
