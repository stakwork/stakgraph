import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { composeNodeKey, closeGraphBackends } from "strut";
import type { NodeSchema } from "strut";
import pages, {
  PRODUCTS_PLACEHOLDER,
  parsePage,
  parseStamp,
  planPage,
  productsPrompt,
  repoInputHash,
  repoPrompt,
  rootInputHash,
  stampOf,
  sumUsage,
  textHash,
} from "../steps/pages.js";

const SCHEMA = { type: "Concept", node_key: "concept-name", parent: "Thing", attributes: {} } as unknown as NodeSchema;

describe("the hashes", () => {
  it("do not depend on order, and change with any input", () => {
    const docs = [
      { name: "README.md", body: "a" },
      { name: "AGENTS.md", body: "b" },
    ];
    const feats = [
      { name: "Chat", description: "talk" },
      { name: "Auth", description: "log in" },
    ];
    const h = repoInputHash(docs, feats);
    assert.match(h, /^[0-9a-f]{12}$/);
    assert.equal(repoInputHash([...docs].reverse(), [...feats].reverse()), h);
    assert.notEqual(repoInputHash([{ name: "README.md", body: "a!" }, docs[1]!], feats), h, "a doc file changed");
    assert.notEqual(repoInputHash(docs, [feats[0]!, { name: "Auth", description: "sign in" }]), h, "a feature's description changed");
    assert.notEqual(repoInputHash(docs, [feats[0]!]), h, "a feature went");
    const kids = [
      { name: "o/a", description: "d", docs: "x" },
      { name: "o/b", description: "e", docs: "y" },
    ];
    assert.equal(rootInputHash([...kids].reverse()), rootInputHash(kids));
    assert.notEqual(rootInputHash([kids[0]!, { ...kids[1]!, docs: "y2" }]), rootInputHash(kids), "a child's docs changed");
  });

  it("the stamp carries the inputs' hash and the text's, and parses back", () => {
    const page = { description: "d", docs: "body" };
    const s = stampOf("acme/web", "0123456789ab", textHash(page));
    assert.equal(s, `lab/concepts/acme/web@0123456789ab.${textHash(page)}`);
    assert.deepEqual(parseStamp(s), { name: "acme/web", inputHash: "0123456789ab", textHash: textHash(page) });
    assert.equal(parseStamp("strut/concepts/Job.md@0123456789ab"), null, "a seeded page's stamp is not ours");
    assert.equal(parseStamp(undefined), null);
  });

  it("planPage: create / foreign / edited / unchanged / rewrite", () => {
    const page = { description: "d", docs: "body" };
    const ours = { ref_id: "r", deleted: false, unique_source_id: stampOf("x", "aaaaaaaaaaaa", textHash(page)), ...page };
    assert.equal(planPage(null, "aaaaaaaaaaaa"), "create");
    assert.equal(planPage({ ...ours, deleted: true }, "aaaaaaaaaaaa"), "create", "a deleted page of ours is written again");
    assert.equal(planPage({ ref_id: "r", deleted: false, ...page }, "aaaaaaaaaaaa"), "foreign", "no stamp: a person's page under our name");
    assert.equal(planPage({ ...ours, docs: "someone changed this" }, "aaaaaaaaaaaa"), "edited");
    assert.equal(planPage(ours, "aaaaaaaaaaaa"), "unchanged");
    assert.equal(planPage(ours, "bbbbbbbbbbbb"), "rewrite");
  });
});

describe("the model's answer", () => {
  it("is a Description line then the page; without the line the first line stands in; fences are stripped", () => {
    assert.deepEqual(parsePage("Description: A workflow engine.\n\n# strut\n\nBody."), { description: "A workflow engine.", docs: "# strut\n\nBody." });
    assert.deepEqual(parsePage("**Description:** A thing.\n\nBody."), { description: "A thing.", docs: "Body." });
    assert.deepEqual(parsePage("```markdown\nDescription: X.\n\nY.\n```"), { description: "X.", docs: "Y." });
    assert.deepEqual(parsePage("# Title\n\nBody."), { description: "Title", docs: "# Title\n\nBody." });
  });

  it("the prompts carry the docs and the features, sorted, and the children's pages", () => {
    const p = repoPrompt("o/r", [{ name: "README.md", body: "readme" }], [{ name: "B", description: "bb" }, { name: "A", description: "aa" }]);
    assert.match(p, /### README\.md\n\nreadme/);
    assert.ok(p.indexOf("- A — aa") < p.indexOf("- B — bb"));
    assert.match(repoPrompt("o/r", [], []), /none of the doc files/);
    assert.match(productsPrompt([{ name: "o/r", description: "d", docs: "x" }]), /## o\/r\n\nd\n\nx/);
  });

  it("sumUsage adds the five fields and ignores the rest", () => {
    assert.deepEqual(sumUsage([{ input: 1, output: 2, total: 3, inputTokens: 99 }, undefined, { input: 1, cache_read: 4 }]), {
      input: 2,
      cache_read: 4,
      cache_write: 0,
      output: 2,
      total: 3,
    });
  });
});

// ── the step over a fake graph: the whole decision flow, offline ────────────

function fakeGraph() {
  const nodes = new Map<string, { ref_id: string; props: Record<string, unknown> }>();
  const keys = new Map<string, string>();
  const edges: Array<{ edge: string; s: string; t: string }> = [];
  let n = 0;
  const graph = {
    namespace: "default",
    schemas: { schema: async () => SCHEMA },
    tx: async (fn: (tx: unknown) => Promise<unknown>) => fn({}),
    nodeEngine: {
      refByKey: async (_tx: unknown, _type: string, key: string) => keys.get(key) ?? null,
      readRaw: async (_tx: unknown, ref: string) => nodes.get(ref) ?? null,
    },
    edgeEngine: {
      liveEdgesOf: async (_tx: unknown, ref: string, edge: string, forward: boolean) => ({
        node: ref,
        edges: edges.filter((e) => e.edge === edge && (forward ? e.s === ref : e.t === ref)).map((e) => ({ other: forward ? e.t : e.s, edge_ref_id: "", props: {} })),
      }),
    },
    nodes: {
      write: async ({ data }: { data: Record<string, unknown> }, mode: string) => {
        const key = composeNodeKey(SCHEMA, data);
        let ref = keys.get(key);
        if (!ref) {
          ref = `ref-${++n}`;
          keys.set(key, ref);
          nodes.set(ref, { ref_id: ref, props: { ...data } });
          return { ref_id: ref, node_key: key, node_type: "Concept", outcome: "created" };
        }
        if (mode === "create") throw new Error(`exists: ${key}`);
        nodes.set(ref, { ref_id: ref, props: { ...nodes.get(ref)!.props, ...data } });
        return { ref_id: ref, node_key: key, node_type: "Concept", outcome: "updated" };
      },
    },
    edges: {
      write: async ({ edge, source_ref_id: s, target_ref_id: t }: { edge: string; source_ref_id: string; target_ref_id: string }) => {
        const created = !edges.some((e) => e.edge === edge && e.s === s && e.t === t);
        if (created) edges.push({ edge, s, t });
        return { created, ref_id: "", edge_key: "", source_ref_id: s, target_ref_id: t };
      },
    },
  };
  const byName = (name: string) => nodes.get(keys.get(composeNodeKey(SCHEMA, { name }))!)!;
  return { graph, nodes, edges, byName };
}

function harness(opts: { files?: Record<string, string>; features?: Array<{ name: string; description: string; ref_id?: string }>; graph?: unknown } = {}) {
  const calls: Array<{ system: string; prompt: string }> = [];
  const files = opts.files ?? {};
  const services = {
    graph: opts.graph,
    storage: { getAllConcepts: async () => opts.features ?? [] },
    github: () => ({
      rest: {
        repos: {
          getContent: async ({ path }: { path: string }) => {
            if (files[path] == null) throw Object.assign(new Error("not found"), { status: 404 });
            return { data: { type: "file", encoding: "base64", content: Buffer.from(files[path]!).toString("base64") } };
          },
        },
      },
    }),
    summarizer: {
      generatePage: async (a: { system: string; prompt: string }) => {
        calls.push(a);
        return { text: `Description: ${a.system} page.\n\nWritten over ${a.prompt.length} chars of prompt.`, usage: { input: 10, output: 5, total: 15 } };
      },
    },
  };
  const cfg = (over: Record<string, unknown> = {}) => pages.input.parse({ owner: "o", repo: "r", repoPageSystem: "repo", productsSystem: "products", ...over });
  const run = (over?: Record<string, unknown>) => pages.run(cfg(over), { services } as never);
  return { run, calls, files };
}

describe("concepts/pages over a fake graph", () => {
  it("skips without a graph", async () => {
    const h = harness();
    const out = await h.run();
    assert.equal(out.repoPage, "skipped");
    assert.equal(out.products, "skipped");
    assert.deepEqual(h.calls, []);
  });

  it("writes the three levels, then nothing until an input changes; a hand edit is left and reported", async () => {
    const g = fakeGraph();
    // Two features as the sweep writes them: `repo` on the node, a ref_id each.
    const f1 = await g.graph.nodes.write({ data: { name: "Chat", description: "talk", repo: "o/r", id: "o/r/chat" } }, "create");
    const f2 = await g.graph.nodes.write({ data: { name: "Auth", description: "log in", repo: "o/r", id: "o/r/auth" } }, "create");
    const features = [
      { name: "Chat", description: "talk", ref_id: f1.ref_id },
      { name: "Auth", description: "log in", ref_id: f2.ref_id },
    ];
    const h = harness({ graph: g.graph, files: { "README.md": "# r\n\nA repo.", "AGENTS.md": "Work like this." }, features });

    // 1. First sweep: the migration — both pages written, every feature parented.
    let out = await h.run();
    assert.equal(out.repoPage, "written");
    assert.equal(out.products, "written");
    assert.equal(out.reparented, 2);
    assert.deepEqual(out.usage, { input: 20, cache_read: 0, cache_write: 0, output: 10, total: 30 });
    assert.equal(h.calls.length, 2);
    assert.match(h.calls[0]!.prompt, /### AGENTS\.md\n\nWork like this\./);
    assert.match(h.calls[0]!.prompt, /- Auth — log in\n- Chat — talk/);
    assert.match(h.calls[1]!.prompt, /## o\/r\n\nrepo page\./, "the root reads its children's pages");
    const repo = g.byName("o/r");
    assert.equal(repo.props["id"], "o/r");
    assert.equal(repo.props["repo"], undefined, "no repo property: invisible to every feature reader");
    assert.equal(parseStamp(repo.props["unique_source_id"])?.name, "o/r");
    const root = g.byName("Products");
    assert.equal(root.props["description"], "products page.");
    assert.deepEqual(
      g.edges.map((e) => `${e.edge}:${e.s}>${e.t}`).sort(),
      [`PARENT_OF:${repo.ref_id}>${f1.ref_id}`, `PARENT_OF:${repo.ref_id}>${f2.ref_id}`, `PARENT_OF:${root.ref_id}>${repo.ref_id}`].sort(),
    );

    // 2. The same inputs: nothing.
    out = await h.run();
    assert.deepEqual([out.repoPage, out.products, out.reparented, h.calls.length], ["unchanged", "unchanged", 0, 2]);

    // 3. A doc file changed at HEAD: the repo page is rewritten, and the root follows because its child's text changed.
    h.files["README.md"] = "# r\n\nA repo, reworded.";
    out = await h.run();
    assert.deepEqual([out.repoPage, out.products, h.calls.length], ["written", "written", 4]);

    // 4. The features' docs changing is NOT an input (the sweep rewrites those every run); a description changing is.
    g.nodes.get(f1.ref_id)!.props["docs"] = "long new docs";
    out = await h.run();
    assert.deepEqual([out.repoPage, out.products], ["unchanged", "unchanged"]);
    features[0]!.description = "chat, with threads";
    out = await h.run();
    assert.deepEqual([out.repoPage, out.products, h.calls.length], ["written", "written", 6]);

    // 5. Someone edited the root in the graph: left alone, reported; the repo page is unaffected.
    g.byName("Products").props["docs"] = "A person's rewrite.";
    out = await h.run();
    assert.deepEqual([out.repoPage, out.products, h.calls.length], ["unchanged", "edited", 6]);
    assert.equal(g.byName("Products").props["docs"], "A person's rewrite.");
    // …and stays so even when the inputs move on.
    h.files["AGENTS.md"] = "Work like THIS.";
    out = await h.run();
    assert.deepEqual([out.repoPage, out.products, h.calls.length], ["written", "edited", 7]);
  });

  it("a page under our name that is not ours is foreign: left alone, still a parent", async () => {
    const g = fakeGraph();
    await g.graph.nodes.write({ data: { name: "Products", description: "Theirs.", docs: "Hand-written." } }, "create");
    const f = await g.graph.nodes.write({ data: { name: "Chat", description: "talk", repo: "o/r" } }, "create");
    const h = harness({ graph: g.graph, files: { "README.md": "r" }, features: [{ name: "Chat", description: "talk", ref_id: f.ref_id }] });
    const out = await h.run();
    assert.deepEqual([out.repoPage, out.products, out.reparented], ["written", "foreign", 1]);
    assert.equal(g.byName("Products").props["docs"], "Hand-written.");
    assert.equal(g.edges.filter((e) => e.s === g.byName("Products").ref_id).length, 1, "the repo page hangs under it all the same");
  });

  it("a repository with no docs and no features gets no page; the root is a placeholder until a child gives it something to say", async () => {
    const g = fakeGraph();
    const h = harness({ graph: g.graph });
    const out = await h.run();
    assert.deepEqual([out.repoPage, out.products, out.reparented, h.calls.length], ["empty", "empty", 0, 0]);
    assert.equal(g.byName("Products").props["description"], PRODUCTS_PLACEHOLDER.description);
    assert.equal(g.byName("o/r"), undefined);
  });

  it("a feature with no ref_id is reported, never parented", async () => {
    const g = fakeGraph();
    const h = harness({ graph: g.graph, files: { "README.md": "r" }, features: [{ name: "Ghost", description: "unkeyed" }] });
    const out = await h.run();
    assert.deepEqual([out.repoPage, out.reparented, out.unparented], ["written", 0, ["Ghost"]]);
  });
});

// ── live: the real writers and readers on a throwaway Neo4j ────────────────
// Runs only when STRUT_TEST_NEO4J_URI is set (a throwaway container: it is written to).
const URI = process.env.STRUT_TEST_NEO4J_URI;
const liveEnv = () => ({
  NEO4J_URI: URI!,
  NEO4J_USER: process.env.STRUT_TEST_NEO4J_USER ?? "neo4j",
  NEO4J_PASSWORD: process.env.STRUT_TEST_NEO4J_PASSWORD ?? "struttest",
  STRUT_GRAPH_SEED_ONTOLOGY: "1",
  STRUT_GRAPH_EMBEDDINGS: "off",
});

describe("concepts/pages (live)", { skip: !URI }, () => {
  const tag = `T${Date.now().toString(36)}`;
  const owner = "lab";
  const repo = `pages-${tag}`;
  const repoName = `${owner}/${repo}`;
  const root = `Products ${tag}`;
  let ws: import("strut").WorkspaceStore;
  let dataDir: string;
  const g = () => ws.graph! as import("strut").Neo4jBackend;
  before(async () => {
    const { graphWorkspaceFromEnv } = await import("strut");
    dataDir = await mkdtemp(join(tmpdir(), "concept-pages-"));
    ws = (await graphWorkspaceFromEnv(liveEnv(), { dataDir })).workspace;
  });
  after(async () => {
    if (ws?.graph) {
      await g().bolt.run(`MATCH (n:Concept) WHERE n.name = $root OR n.name = $repo OR n.repo = $repo DETACH DELETE n`, { root, repo: repoName });
      await closeGraphBackends();
    }
    await rm(dataDir, { recursive: true, force: true });
  });

  it("writes the tree through strut's writers, and a second run writes nothing", async () => {
    // Two features, as the sweep leaves them: `repo` + `id` on the node (the graph's writer stamps node_key + ref_id).
    const f1 = await g().nodes.write({ type: "Concept", data: { name: `Chat ${tag}`, description: "talk", repo: repoName, id: `${repoName}/chat` } }, "create");
    const f2 = await g().nodes.write({ type: "Concept", data: { name: `Auth ${tag}`, description: "log in", repo: repoName, id: `${repoName}/auth` } }, "create");
    const features = [
      { name: `Chat ${tag}`, description: "talk", ref_id: f1.ref_id },
      { name: `Auth ${tag}`, description: "log in", ref_id: f2.ref_id },
    ];
    const h = harness({ graph: g(), files: { "README.md": "# live\n\nA repo." }, features });
    let out = await h.run({ owner, repo, root });
    assert.deepEqual([out.repoPage, out.products, out.reparented, out.unparented], ["written", "written", 2, undefined]);

    const rows = await g().bolt.run(
      `MATCH (p:Concept {name: $root})-[:PARENT_OF]->(r:Concept {name: $repo})-[:PARENT_OF]->(f:Concept)
       RETURN r.unique_source_id AS stamp, r.id AS id, r.repo AS repo, p.unique_source_id AS rootStamp, collect(f.name) AS kids`,
      { root, repo: repoName },
    );
    assert.equal(rows.length, 1, "root → repo page → features, along PARENT_OF");
    const row = rows[0]!;
    assert.equal(parseStamp(row.stamp)?.name, repoName);
    assert.equal(row.id, repoName);
    assert.equal(row.repo, null, "the page carries no repo property");
    assert.equal(parseStamp(row.rootStamp)?.name, root);
    assert.deepEqual((row.kids as string[]).sort(), [`Auth ${tag}`, `Chat ${tag}`]);

    out = await h.run({ owner, repo, root });
    assert.deepEqual([out.repoPage, out.products, out.reparented, h.calls.length], ["unchanged", "unchanged", 0, 2]);

    // A person edits the repo page in the graph: the next sweep leaves it and says so — and the root
    // follows, since a child's text is one of ITS inputs: the overview reflects the person's edit.
    await g().bolt.run(`MATCH (r:Concept {name: $repo}) SET r.docs = 'Edited by hand.'`, { repo: repoName });
    h.files["README.md"] = "# live\n\nReworded.";
    out = await h.run({ owner, repo, root });
    assert.deepEqual([out.repoPage, out.products, h.calls.length], ["edited", "written", 3]);
    assert.match(h.calls[2]!.prompt, /Edited by hand\./);
  });
});
