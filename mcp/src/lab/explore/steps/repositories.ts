import { z, defineStep, openGraphBackend, type StepContext, type StrutCapabilities } from "strut";

/**
 * Which repositories this graph holds parsed, beside the ones a launch
 * checked out — repo_agent's prependRepoInfo (mcp/src/repo/index.ts) as a
 * step. The explore agent is told per repository where to look: the graph
 * (File / Function / Endpoint nodes with descriptions) for an ingested one,
 * the files alone for one that is only on disk.
 *
 * `graphCtx` is strut's `graph/*` preamble, inlined (seeded steps import only
 * "strut"): the same config and options, so the same cached backend.
 */
async function graphCtx(ctx: StepContext<StrutCapabilities>) {
  const secrets = ctx.services?.secrets;
  const uri = (await secrets?.get("NEO4J_URI")) || `bolt://${(await secrets?.get("NEO4J_HOST")) || "localhost:7687"}`;
  const emb = ((await secrets?.get("STRUT_GRAPH_EMBEDDINGS")) ?? "").toLowerCase();
  const ont = ((await secrets?.get("STRUT_GRAPH_SEED_ONTOLOGY")) ?? "").toLowerCase();
  return openGraphBackend(
    {
      uri,
      user: (await secrets?.get("NEO4J_USER")) || "neo4j",
      password: (await secrets?.get("NEO4J_PASSWORD")) || "testtest",
      namespace: (await secrets?.get("STRUT_GRAPH_NAMESPACE")) || "default",
      database: (await secrets?.get("NEO4J_DATABASE")) || undefined,
    },
    { embeddings: !["off", "0", "false"].includes(emb), seedOntology: ["1", "true", "on"].includes(ont) },
  );
}

/** `https://github.com/stakwork/hive(.git)` → `stakwork/hive` (repo_agent's normalizeRepoRef). */
export function repoRef(input: unknown): string {
  const t = String(input ?? "").trim().replace(/\.git$/, "").replace(/\/+$/, "");
  if (!/^https?:\/\//.test(t)) return t;
  const parts = t.split("/").filter(Boolean);
  const repo = parts.pop() || "";
  const owner = parts.pop() || "";
  return owner && repo ? `${owner}/${repo}` : repo;
}

const Graph = z.object({ repo: z.string(), ref_id: z.string().optional() });

/** The note the agent's prompt carries: per repository, where to look. */
export function repoNote(graph: Array<{ repo: string; ref_id?: string }>, checkedOut: string[]): string {
  const key = (r: string) => r.toLowerCase();
  const inGraph = new Map(graph.map((g) => [key(g.repo), g]));
  const both = checkedOut.filter((r) => inGraph.has(key(r)));
  const filesOnly = checkedOut.filter((r) => !inGraph.has(key(r)));
  const graphOnly = graph.filter((g) => !checkedOut.some((r) => key(r) === key(g.repo)));
  const ref = (g?: { ref_id?: string }) => (g?.ref_id ? ` (Repository ${g.ref_id})` : "");
  const lines: string[] = [];
  if (both.length) {
    lines.push(
      `Checked out in your working directory AND parsed into the graph: ${both.map((r) => r + ref(inGraph.get(key(r)))).join(", ")}. ` +
        "Search the graph first, even when a grep would find it: graph/graph-search with `type: \"Function,Class,Datamodel,Endpoint,Request,Page,File\"`, " +
        "then graph/graph-get the hits you will use — a search hit has no file path; the node gives its file, its start and end lines, its body and a written description, so you can cite it without opening the file. " +
        "Open a file only for lines the graph does not give, and fulltext_search for what it does not hold.",
    );
  }
  if (filesOnly.length) {
    lines.push(`Checked out but NOT in the graph: ${filesOnly.join(", ")}. Read these with fulltext_search and the file tool.`);
  }
  if (graphOnly.length) {
    lines.push(
      `Parsed into the graph but not checked out: ${graphOnly.map((g) => g.repo + ref(g)).join(", ")}. ` +
        "Their code is searchable in the graph (graph/graph-search with `type: \"Function,Class,Datamodel,Endpoint,Request,Page,File\"`, then graph/graph-get); their files are not on disk.",
    );
  }
  if (!lines.length) lines.push("No repositories were named and the graph holds none: answer from the graph's other nodes.");
  return lines.join("\n");
}

export default defineStep({
  type: "explore/repositories",
  description:
    "The repositories this deployment's graph holds parsed (its Repository nodes, as owner/name), compared with the ones a launch checked out. " +
    "Output: { graph: [{ repo, ref_id }], checkedOut: [owner/name], note } — `note` says, per repository, whether to search the graph or read the files. " +
    "Never fails: a graph that cannot be read is no repositories, with `error`.",
  input: z.object({
    repos: z.any().optional().describe("The repositories checked out (URLs or owner/name), usually the launch's `repos`"),
  }),
  output: z.object({
    graph: z.array(Graph),
    checkedOut: z.array(z.string()),
    note: z.string(),
    error: z.string().optional(),
  }),
  async run(cfg, ctx: StepContext<StrutCapabilities>) {
    const checkedOut = [...new Set((Array.isArray(cfg.repos) ? cfg.repos : []).map(repoRef).filter(Boolean))];
    let graph: Array<{ repo: string; ref_id?: string }> = [];
    let error: string | undefined;
    try {
      const b = await graphCtx(ctx);
      const rows = await b.bolt.read((tx) =>
        tx.run("MATCH (r:Repository) WHERE r.deleted_at IS NULL RETURN r.ref_id AS ref_id, r.name AS name, r.source_link AS source_link"),
      );
      const seen = new Set<string>();
      for (const rec of rows.records) {
        const repo = repoRef(rec.get("source_link") || rec.get("name"));
        if (!repo || seen.has(repo.toLowerCase())) continue;
        seen.add(repo.toLowerCase());
        const refId = rec.get("ref_id");
        graph.push(refId ? { repo, ref_id: String(refId) } : { repo });
      }
      graph.sort((a, b) => a.repo.localeCompare(b.repo));
    } catch (err) {
      graph = [];
      error = err instanceof Error ? err.message : String(err);
    }
    if (error) {
      // Not "the graph holds none": it may, and the agent can still search it.
      const note =
        "Which repositories the graph holds parsed could not be read; graph/graph-search with `type: \"Function,Class,Datamodel,Endpoint,Request,Page,File\"` may still find their code." +
        (checkedOut.length ? `\nChecked out in your working directory: ${checkedOut.join(", ")}.` : "");
      return { graph, checkedOut, note, error };
    }
    return { graph, checkedOut, note: repoNote(graph, checkedOut) };
  },
});
