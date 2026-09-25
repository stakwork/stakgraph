import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { CONCEPTS_DIR, parseConceptFile, planConceptSeed, seedJanitorConcepts, stampOf } from "./seed.js";

describe("parseConceptFile", () => {
  it("name from the filename, description + parent from front matter, body as docs", () => {
    const text = "---\ndescription: One line.\nparent: Janitor\n---\nThe mandate.\n\nMore.\n";
    const c = parseConceptFile("Legal Overfit.md", text);
    assert.deepEqual(c, {
      name: "Legal Overfit",
      description: "One line.",
      parent: "Janitor",
      docs: "The mandate.\n\nMore.",
      stamp: stampOf("Legal Overfit.md", text),
    });
    assert.match(c.stamp, /^lab\/janitor\/concepts\/Legal Overfit\.md@[0-9a-f]{12}$/);
  });

  it("no front matter, empty body: just a name and a stamp", () => {
    assert.deepEqual(parseConceptFile("X.md", ""), { name: "X", stamp: stampOf("X.md", "") });
    assert.deepEqual(parseConceptFile("X.md", "body only\n"), { name: "X", docs: "body only", stamp: stampOf("X.md", "body only\n") });
  });

  it("the stamp follows the file's bytes", () => {
    assert.notEqual(stampOf("A.md", "a"), stampOf("A.md", "b"));
    assert.notEqual(stampOf("A.md", "a"), stampOf("B.md", "a"));
    assert.equal(stampOf("A.md", "a"), stampOf("A.md", "a"));
  });

  it("the committed root parses as expected", async () => {
    const c = parseConceptFile("Janitor.md", await readFile(join(CONCEPTS_DIR, "Janitor.md"), "utf-8"));
    assert.equal(c.name, "Janitor");
    assert.equal(c.parent, undefined);
    assert.equal(c.description, "Automated agents for cleaning up code, concept trees, or other data.");
    assert.equal(c.docs, c.description);
  });
});

describe("planConceptSeed", () => {
  const stamp = "lab/janitor/concepts/J.md@aaaaaaaaaaaa";
  const older = "lab/janitor/concepts/J.md@bbbbbbbbbbbb";
  it("nothing there → create", () => assert.equal(planConceptSeed(null, stamp), "create"));
  it("same stamp → keep (ours, unchanged: a graph-side edit or mute sticks); deleted → skip", () => {
    assert.equal(planConceptSeed({ ref_id: "r", unique_source_id: stamp }, stamp), "keep");
    assert.equal(planConceptSeed({ ref_id: "r", unique_source_id: stamp, is_deleted: true }, stamp), "skip");
  });
  it("our path, older hash → update; unless deleted", () => {
    assert.equal(planConceptSeed({ ref_id: "r", unique_source_id: older }, stamp), "update");
    assert.equal(planConceptSeed({ ref_id: "r", unique_source_id: older, is_deleted: true }, stamp), "skip");
  });
  it("no stamp or someone else's → skip (theirs)", () => {
    assert.equal(planConceptSeed({ ref_id: "r" }, stamp), "skip");
    assert.equal(planConceptSeed({ ref_id: "r", unique_source_id: null }, stamp), "skip");
    assert.equal(planConceptSeed({ ref_id: "r", unique_source_id: "lab/janitor/concepts/Other.md@aaaaaaaaaaaa" }, stamp), "skip");
    assert.equal(planConceptSeed({ ref_id: "r", unique_source_id: "gitree" }, stamp), "skip");
  });
});

// Live, against a THROWAWAY Neo4j (strut's convention): skipped unless
// STRUT_TEST_NEO4J_URI is set. Seeds the jarvis ontology on first open.
const URI = process.env.STRUT_TEST_NEO4J_URI;
describe("seedJanitorConcepts (live)", { skip: !URI }, () => {
  const tag = `T${Date.now().toString(36)}`;
  const root = `Janitor ${tag}`;
  const kid = `Kid ${tag}`;
  let ws: import("strut").WorkspaceStore;
  let dir: string;
  let dataDir: string;
  const node = async (name: string) => {
    const g = ws.graph!;
    const rows = await g.bolt.run(
      `MATCH (n:Concept {name: $name, namespace: $ns}) RETURN n.ref_id AS ref_id, n.description AS description, n.docs AS docs, n.unique_source_id AS usid, n.is_muted AS muted`,
      { name, ns: g.cfg.namespace },
    );
    return rows[0] ?? null;
  };
  const wsId = `ws-${tag}`;
  const anchors = async (name: string) => {
    const g = ws.graph!;
    const rows = await g.bolt.run(`MATCH (w:HiveWorkspace)-[e:PROCESS]->(c:Concept {name: $c}) RETURN count(e) AS n`, { c: name });
    return Number(rows[0]?.n ?? 0);
  };
  const parentEdges = async () => {
    const g = ws.graph!;
    const rows = await g.bolt.run(`MATCH (p:Concept {name: $p})-[e:PARENT_OF]->(c:Concept {name: $c}) RETURN count(e) AS n`, { p: root, c: kid });
    return Number(rows[0]?.n ?? 0);
  };
  before(async () => {
    const { graphWorkspaceFromEnv } = await import("strut");
    dataDir = await mkdtemp(join(tmpdir(), "janitor-seed-data-"));
    dir = await mkdtemp(join(tmpdir(), "janitor-seed-"));
    ws = (
      await graphWorkspaceFromEnv(
        {
          NEO4J_URI: URI,
          NEO4J_USER: process.env.STRUT_TEST_NEO4J_USER ?? "neo4j",
          NEO4J_PASSWORD: process.env.STRUT_TEST_NEO4J_PASSWORD ?? "struttest",
          STRUT_GRAPH_SEED_ONTOLOGY: "1",
          STRUT_GRAPH_EMBEDDINGS: "off",
        },
        { dataDir },
      )
    ).workspace;
    await writeFile(join(dir, `${root}.md`), "---\ndescription: The root.\n---\nRoot docs.\n");
    await writeFile(join(dir, `${kid}.md`), `---\ndescription: A kid.\nparent: ${root}\n---\nKid docs v1.\n`);
  });
  after(async () => {
    const g = ws?.graph;
    if (g) {
      await g.bolt.run(`MATCH (n:Concept) WHERE n.name IN $names DETACH DELETE n`, { names: [root, kid] });
      await g.bolt.run(`MATCH (w:HiveWorkspace) WHERE w.workspace_id STARTS WITH $id DETACH DELETE w`, { id: wsId });
      await g.bolt.close().catch(() => {});
    }
    await rm(dir, { recursive: true, force: true });
    await rm(dataDir, { recursive: true, force: true });
  });

  it("creates the tree, then does nothing on a reseed", async () => {
    await seedJanitorConcepts(ws, { dir });
    const r = await node(root);
    const k = await node(kid);
    assert.ok(r && k);
    assert.equal(r.description, "The root.");
    assert.equal(k.docs, "Kid docs v1.");
    assert.match(String(k.usid), /^lab\/janitor\/concepts\/Kid .+\.md@[0-9a-f]{12}$/);
    assert.equal(await parentEdges(), 1);

    await seedJanitorConcepts(ws, { dir });
    assert.equal((await node(kid))!.ref_id, k.ref_id);
    assert.equal(await parentEdges(), 1);
    // No workspace node in this graph yet: nothing to anchor to, no error.
    assert.equal(await anchors(root), 0);
  });

  it("a graph-side edit survives an unchanged template; a changed template wins but keeps is_muted", async () => {
    const g = ws.graph!;
    await g.bolt.run(`MATCH (n:Concept {name: $name}) SET n.docs = 'Rewritten in Learn.', n.is_muted = true`, { name: kid });
    await seedJanitorConcepts(ws, { dir });
    assert.equal((await node(kid))!.docs, "Rewritten in Learn.");

    await writeFile(join(dir, `${kid}.md`), `---\ndescription: A kid.\nparent: ${root}\n---\nKid docs v2.\n`);
    await seedJanitorConcepts(ws, { dir });
    const k = await node(kid);
    assert.equal(k!.docs, "Kid docs v2.");
    assert.equal(k!.muted, true);
    assert.equal(await parentEdges(), 1);
  });

  it("a node that lost its stamp is never touched again", async () => {
    const g = ws.graph!;
    await g.bolt.run(`MATCH (n:Concept {name: $name}) SET n.docs = 'Mine now.' REMOVE n.unique_source_id`, { name: kid });
    await writeFile(join(dir, `${kid}.md`), `---\ndescription: A kid.\nparent: ${root}\n---\nKid docs v3.\n`);
    await seedJanitorConcepts(ws, { dir });
    const k = await node(kid);
    assert.equal(k!.docs, "Mine now.");
    assert.equal(k!.usid, null);
  });

  it("anchors an owned root to the one HiveWorkspace node, idempotently; never a child; never with two", async () => {
    const g = ws.graph!;
    await g.nodes.write({ type: "HiveWorkspace", data: { workspace_id: wsId, name: "T", slug: `t-${tag}` } }, "create", { namespace: g.cfg.namespace });
    await seedJanitorConcepts(ws, { dir }); // root is "keep" (unchanged file) → still anchored
    assert.equal(await anchors(root), 1);
    assert.equal(await anchors(kid), 0);
    await seedJanitorConcepts(ws, { dir });
    assert.equal(await anchors(root), 1);

    await g.nodes.write({ type: "HiveWorkspace", data: { workspace_id: `${wsId}-2`, name: "T2" } }, "create", { namespace: g.cfg.namespace });
    await writeFile(join(dir, `${root}.md`), "---\ndescription: The root, v2.\n---\nRoot docs v2.\n");
    await seedJanitorConcepts(ws, { dir }); // updated, but two workspace nodes → no guess
    assert.equal((await node(root))!.description, "The root, v2.");
    assert.equal(await anchors(root), 1);
  });
});
