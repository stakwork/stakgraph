import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { composeNodeKey, closeGraphBackends } from "strut";
import type { NodeSchema } from "strut";
import { normalizeDocPaths } from "../doc-files.js";
import pages, {
  CODEBASE_PLACEHOLDER,
  matchDocNames,
  parsePage,
  parseStamp,
  planPage,
  codebasePrompt,
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

  it("the prompts carry the docs, then the shape, then the features, sorted, and the children's pages", () => {
    const p = repoPrompt(
      "o/r",
      [
        { name: "package.json", body: "{}", kind: "shape" },
        { name: "README.md", body: "readme", kind: "doc" },
      ],
      [
        { name: "B", description: "bb" },
        { name: "A", description: "aa" },
      ],
    );
    assert.match(p, /### README\.md\n\nreadme/);
    assert.ok(p.indexOf("## The repository's own docs") < p.indexOf("## Its shape"), "docs before the shape");
    assert.ok(p.indexOf("### package.json") > p.indexOf("## Its shape"));
    assert.ok(p.indexOf("- A — aa") < p.indexOf("- B — bb"));
    assert.match(repoPrompt("o/r", [], []), /no file that describes it as a whole/);
    assert.match(codebasePrompt([{ name: "o/r", description: "d", docs: "x" }]), /## o\/r\n\nd\n\nx/);
  });

  it("matchDocNames: a file whose stem is one of the names, any case, any extension; never a directory", () => {
    const entries = [
      { name: "Readme.rst", path: "Readme.rst", type: "file" },
      { name: "AGENTS.md", path: "AGENTS.md", type: "file" },
      { name: "readme", path: "docs/readme", type: "file" },
      { name: "README.en.md", path: "README.en.md", type: "file" },
      { name: "CLAUDE", path: "CLAUDE", type: "dir" },
      { name: "index.ts", path: "index.ts", type: "file" },
    ];
    assert.deepEqual(matchDocNames(entries, ["README", "AGENTS", "CLAUDE"]), ["AGENTS.md", "Readme.rst", "docs/readme"]);
  });

  it("normalizeDocPaths: strings the clone holds as files, relative, no `..`, deduped, capped at 8", () => {
    const exists = (p: string) => p !== "missing.md";
    assert.deepEqual(normalizeDocPaths(["./README.md", "/docs/a.md", "docs/a.md", "../x.md", "missing.md", 7, " AGENTS.md "], exists), ["README.md", "docs/a.md", "AGENTS.md"]);
    assert.deepEqual(normalizeDocPaths("README.md", exists), []);
    assert.equal(normalizeDocPaths(Array.from({ length: 12 }, (_, i) => `d${i}.md`), () => true).length, 8);
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

function harness(
  opts: {
    files?: Record<string, string>;
    features?: Array<{ name: string; description: string; ref_id?: string }>;
    graph?: unknown;
    /** The recorded pick, as the bootstrap left it: absent = never recorded. */
    record?: string[];
    /** What the one-time identify pass answers; absent = no such service. */
    identify?: string[];
  } = {},
) {
  const calls: Array<{ system: string; prompt: string }> = [];
  const files = opts.files ?? {};
  const records = new Map<string, string[]>();
  if (opts.record) records.set("o/r", opts.record);
  const identified: string[] = [];
  const notFound = () => Object.assign(new Error("not found"), { status: 404 });
  // The contents API: a path that is a file comes back base64; a directory (the root is "") as its entries.
  const getContent = async ({ path }: { path: string }) => {
    if (files[path] != null) return { data: { type: "file", encoding: "base64", content: Buffer.from(files[path]!).toString("base64") } };
    const prefix = path ? `${path}/` : "";
    const names = new Map<string, string>();
    for (const k of Object.keys(files)) {
      if (!k.startsWith(prefix)) continue;
      const rest = k.slice(prefix.length);
      const [head] = rest.split("/");
      names.set(head!, rest.includes("/") ? "dir" : "file");
    }
    if (!names.size) throw notFound();
    return { data: [...names].map(([name, type]) => ({ name, path: `${prefix}${name}`, type })) };
  };
  const services = {
    graph: opts.graph,
    storage: {
      getAllConcepts: async () => opts.features ?? [],
      getDocFiles: async (repo: string) => records.get(repo) ?? null,
      setDocFiles: async (repo: string, list: string[]) => void records.set(repo, list),
    },
    github: () => ({ rest: { repos: { getContent } } }),
    ...(opts.identify
      ? {
          identifyDocs: async (owner: string, repo: string) => {
            identified.push(`${owner}/${repo}`);
            records.set(`${owner}/${repo}`, opts.identify!);
            return { docFiles: opts.identify!, usage: { input: 100, output: 1, total: 101 } };
          },
        }
      : {}),
    summarizer: {
      generatePage: async (a: { system: string; prompt: string }) => {
        calls.push(a);
        return { text: `Description: ${a.system} page.\n\nWritten over ${a.prompt.length} chars of prompt.`, usage: { input: 10, output: 5, total: 15 } };
      },
    },
  };
  const cfg = (over: Record<string, unknown> = {}) => pages.input.parse({ owner: "o", repo: "r", repoPageSystem: "repo", codebaseSystem: "codebase", ...over });
  const run = (over?: Record<string, unknown>) => pages.run(cfg(over), { services } as never);
  return { run, calls, files, records, identified };
}

describe("concepts/pages over a fake graph", () => {
  it("skips without a graph", async () => {
    const h = harness();
    const out = await h.run();
    assert.equal(out.repoPage, "skipped");
    assert.equal(out.codebase, "skipped");
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

    // 1. First sweep: the migration — both pages written, every feature parented. No recorded pick (a repo
    //    bootstrapped before it existed): the root listing finds the docs, and the shape rides along.
    let out = await h.run();
    assert.equal(out.repoPage, "written");
    assert.equal(out.codebase, "written");
    assert.equal(out.reparented, 2);
    assert.deepEqual(out.docs, { source: "listing", files: ["AGENTS.md", "README.md"] });
    assert.deepEqual(out.usage, { input: 20, cache_read: 0, cache_write: 0, output: 10, total: 30 });
    assert.equal(h.calls.length, 2);
    assert.match(h.calls[0]!.prompt, /### AGENTS\.md\n\nWork like this\./);
    assert.match(h.calls[0]!.prompt, /### \(root listing\)\n\nAGENTS\.md\nREADME\.md/);
    assert.match(h.calls[0]!.prompt, /- Auth — log in\n- Chat — talk/);
    assert.deepEqual(h.identified, [], "the listing found prose: no clone, no agent");
    assert.match(h.calls[1]!.prompt, /## o\/r\n\nrepo page\./, "the root reads its children's pages");
    const repo = g.byName("o/r");
    assert.equal(repo.props["id"], "o/r");
    assert.equal(repo.props["repo"], undefined, "no repo property: invisible to every feature reader");
    assert.equal(parseStamp(repo.props["unique_source_id"])?.name, "o/r");
    const root = g.byName("Codebase");
    assert.equal(root.props["description"], "codebase page.");
    assert.deepEqual(
      g.edges.map((e) => `${e.edge}:${e.s}>${e.t}`).sort(),
      [`PARENT_OF:${repo.ref_id}>${f1.ref_id}`, `PARENT_OF:${repo.ref_id}>${f2.ref_id}`, `PARENT_OF:${root.ref_id}>${repo.ref_id}`].sort(),
    );

    // 2. The same inputs: nothing.
    out = await h.run();
    assert.deepEqual([out.repoPage, out.codebase, out.reparented, h.calls.length], ["unchanged", "unchanged", 0, 2]);

    // 3. A doc file changed at HEAD: the repo page is rewritten, and the root follows because its child's text changed.
    h.files["README.md"] = "# r\n\nA repo, reworded.";
    out = await h.run();
    assert.deepEqual([out.repoPage, out.codebase, h.calls.length], ["written", "written", 4]);

    // 4. The features' docs changing is NOT an input (the sweep rewrites those every run); a description changing is.
    g.nodes.get(f1.ref_id)!.props["docs"] = "long new docs";
    out = await h.run();
    assert.deepEqual([out.repoPage, out.codebase], ["unchanged", "unchanged"]);
    features[0]!.description = "chat, with threads";
    out = await h.run();
    assert.deepEqual([out.repoPage, out.codebase, h.calls.length], ["written", "written", 6]);

    // 5. The repository's shape is an input too: a new top-level folder rewrites the page.
    h.files["kit/package.json"] = "{}";
    out = await h.run();
    assert.deepEqual([out.repoPage, out.codebase, h.calls.length], ["written", "written", 8]);
    assert.match(h.calls[6]!.prompt, /\(root listing\)\n\nAGENTS\.md\nREADME\.md\nkit\//);

    // 6. Someone edited the root in the graph: left alone, reported; the repo page is unaffected.
    g.byName("Codebase").props["docs"] = "A person's rewrite.";
    out = await h.run();
    assert.deepEqual([out.repoPage, out.codebase, h.calls.length], ["unchanged", "edited", 8]);
    assert.equal(g.byName("Codebase").props["docs"], "A person's rewrite.");
    // …and stays so even when the inputs move on.
    h.files["AGENTS.md"] = "Work like THIS.";
    out = await h.run();
    assert.deepEqual([out.repoPage, out.codebase, h.calls.length], ["written", "edited", 9]);
  });

  it("a page under our name that is not ours is foreign: left alone, still a parent", async () => {
    const g = fakeGraph();
    await g.graph.nodes.write({ data: { name: "Codebase", description: "Theirs.", docs: "Hand-written." } }, "create");
    const f = await g.graph.nodes.write({ data: { name: "Chat", description: "talk", repo: "o/r" } }, "create");
    const h = harness({ graph: g.graph, files: { "README.md": "r" }, features: [{ name: "Chat", description: "talk", ref_id: f.ref_id }] });
    const out = await h.run();
    assert.deepEqual([out.repoPage, out.codebase, out.reparented], ["written", "foreign", 1]);
    assert.equal(g.byName("Codebase").props["docs"], "Hand-written.");
    assert.equal(g.edges.filter((e) => e.s === g.byName("Codebase").ref_id).length, 1, "the repo page hangs under it all the same");
  });

  it("a repository the API cannot even list, with no features, gets no page; the root is a placeholder until a child gives it something to say", async () => {
    const g = fakeGraph();
    const h = harness({ graph: g.graph });
    const out = await h.run();
    assert.deepEqual([out.repoPage, out.codebase, out.reparented, h.calls.length], ["empty", "empty", 0, 0]);
    assert.deepEqual(out.docs, { source: "none", files: [] });
    assert.equal(g.byName("Codebase").props["description"], CODEBASE_PLACEHOLDER.description);
    assert.equal(g.byName("o/r"), undefined);
  });

  it("the recorded pick wins over the listing: the bootstrap's doc files are what the page is made from", async () => {
    const g = fakeGraph();
    const h = harness({
      graph: g.graph,
      files: { "README.md": "the readme", "docs/overview.md": "the overview", "docs/changelog.md": "noise", "package.json": "{\"name\":\"r\"}" },
      record: ["docs/overview.md", "gone.md"],
      identify: ["never"],
    });
    const out = await h.run();
    assert.deepEqual(out.docs, { source: "record", files: ["docs/overview.md"] }, "a recorded path that is gone is skipped, not an error");
    const prompt = h.calls[0]!.prompt;
    assert.match(prompt, /### docs\/overview\.md\n\nthe overview/);
    assert.doesNotMatch(prompt, /the readme/, "the README is not read: the pick is the pick");
    assert.match(prompt, /## Its shape[\s\S]*### package\.json\n\n\{"name":"r"\}/, "the manifests always ride along");
    assert.deepEqual(h.identified, []);
  });

  it("no record and nothing README-like: the bootstrap's pick runs alone, once, and its record stops the asking", async () => {
    const g = fakeGraph();
    const h = harness({
      graph: g.graph,
      files: { "src/index.ts": "export {}", "Cargo.toml": "[package]\nname = \"r\"", "notes/guide.md": "the guide" },
      identify: ["notes/guide.md"],
    });
    let out = await h.run();
    assert.deepEqual(out.docs, { source: "identified", files: ["notes/guide.md"] });
    assert.deepEqual(h.identified, ["o/r"]);
    assert.equal(out.usage.input, 120, "what the pass cost is in the step's usage, with the two pages'");
    assert.match(h.calls[0]!.prompt, /### notes\/guide\.md\n\nthe guide/);
    assert.deepEqual(h.records.get("o/r"), ["notes/guide.md"]);
    out = await h.run();
    assert.deepEqual([out.docs.source, out.repoPage, h.identified.length], ["record", "unchanged", 1], "the second sweep reads the record");
  });

  it("the pick that found nothing is still a record: the page comes from the shape, and nobody clones again", async () => {
    const g = fakeGraph();
    const h = harness({ graph: g.graph, files: { "src/index.ts": "export {}", "package.json": "{\"name\":\"bare\"}" }, identify: [] });
    let out = await h.run();
    assert.deepEqual([out.docs.source, out.repoPage, h.identified.length], ["identified", "written", 1]);
    assert.match(h.calls[0]!.prompt, /no file that describes it as a whole[\s\S]*### package\.json/);
    out = await h.run();
    assert.deepEqual([out.docs.source, out.repoPage, h.identified.length], ["record", "unchanged", 1]);
    assert.deepEqual(h.records.get("o/r"), []);
  });

  it("with `identify: false`, or no such service, the page is written from the shape alone", async () => {
    const g = fakeGraph();
    const h = harness({ graph: g.graph, files: { "src/index.ts": "export {}", "go.mod": "module r" }, identify: ["notes/x.md"] });
    const out = await h.run({ identify: false });
    assert.deepEqual([out.docs.source, out.repoPage, h.identified.length], ["none", "written", 0]);
    const h2 = harness({ graph: fakeGraph().graph, files: { "go.mod": "module r" } });
    assert.deepEqual((await h2.run()).docs, { source: "none", files: [] });
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
  const root = `Codebase ${tag}`;
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
    assert.deepEqual([out.repoPage, out.codebase, out.reparented, out.unparented], ["written", "written", 2, undefined]);

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
    assert.deepEqual([out.repoPage, out.codebase, out.reparented, h.calls.length], ["unchanged", "unchanged", 0, 2]);

    // A person edits the repo page in the graph: the next sweep leaves it and says so — and the root
    // follows, since a child's text is one of ITS inputs: the overview reflects the person's edit.
    await g().bolt.run(`MATCH (r:Concept {name: $repo}) SET r.docs = 'Edited by hand.'`, { repo: repoName });
    h.files["README.md"] = "# live\n\nReworded.";
    out = await h.run({ owner, repo, root });
    assert.deepEqual([out.repoPage, out.codebase, h.calls.length], ["edited", "written", 3]);
    assert.match(h.calls[2]!.prompt, /Edited by hand\./);
  });
});
