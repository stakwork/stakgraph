import { z, defineStep, openGraphBackend, type StepContext, type StrutCapabilities } from "strut";

/**
 * janitor/queue — which nodes this janitor run reviews. Code picks them, not
 * the agent: left to walk the subtree itself, the agent read the start's
 * children every run and the levels below now and then (swarm38, Medicine,
 * 2026-10-14: a depth-2 Concept still naming "the benchmark" was read once
 * in 91 hourly runs).
 *
 * The subtree under `start_ref_id` (PARENT_OF, live nodes and edges, the
 * start included) is sorted by name, and the run takes the next `size`
 * nodes after `cursor` — the name of the last node the previous run
 * reviewed — wrapping to the first name past the end. `next_cursor` is the
 * last name taken; the automation hands it back on its next fire
 * (`cursor: "{{ last.output.next_cursor }}"` — `last` is the latest
 * SUCCESSFUL run, so a failed run's batch is taken again). Every node is
 * reviewed once per ceil(total / size) runs. A name, not a position: a
 * Concept added or deleted between runs never shifts anyone's turn, and a
 * cursor whose node is gone still sorts where it was. No cursor: the
 * first `size` names.
 *
 * The batch comes with each node's `description` and `docs` verbatim, so
 * the agent is handed what it reviews instead of spending its steps finding
 * it. Nothing is reported as accessed: a listing is not a node the caller
 * named (the agent's own reads are its tool calls').
 *
 * `graphCtx` is inlined per file, like plan.ts (seeded steps import only "strut").
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

type Bolt = { bolt: { run(q: string, p?: Record<string, unknown>): Promise<Array<Record<string, unknown>>> } };

export interface QueueNode {
  ref_id: string;
  name: string;
}

export interface Pick<T extends QueueNode> {
  batch: T[];
  /** Index of the batch's first node in name order. */
  from: number;
  /** The batch ran past the last name and went on from the first. */
  wrapped: boolean;
  next_cursor: string;
}

/** Name order, by code point: the same on every run and every machine. */
export const byName = (a: QueueNode, b: QueueNode) => (a.name < b.name ? -1 : a.name > b.name ? 1 : a.ref_id < b.ref_id ? -1 : a.ref_id > b.ref_id ? 1 : 0);

/** Pure: the next `size` of `sorted` (name order) after `cursor`, wrapping; everything when it holds no more. */
export function pickBatch<T extends QueueNode>(sorted: T[], cursor: string | undefined, size: number): Pick<T> {
  const n = Math.max(1, Math.floor(size));
  if (sorted.length === 0) return { batch: [], from: 0, wrapped: false, next_cursor: cursor ?? "" };
  const after = cursor ? sorted.findIndex((x) => x.name > cursor) : 0;
  const from = after < 0 ? 0 : after;
  const take = Math.min(n, sorted.length);
  const batch = Array.from({ length: take }, (_, i) => sorted[(from + i) % sorted.length]!);
  return { batch, from, wrapped: from + take > sorted.length || (!!cursor && after < 0), next_cursor: batch.at(-1)!.name };
}

const LIVE_NODE = (v: string) => `(${v}.deleted_at IS NULL AND (${v}.is_deleted IS NULL OR ${v}.is_deleted <> true))`;
const LIVE_EDGE = (r: string) => `((${r}.is_muted IS NULL OR ${r}.is_muted <> true) AND (${r}.is_deleted IS NULL OR ${r}.is_deleted <> true))`;

/** Every live `node_type` node under `start` along live PARENT_OF edges, the start included: ref_id + name. */
export async function readSubtree(b: Bolt, start: string, nodeType: string, maxDepth: number, maxNodes: number): Promise<QueueNode[]> {
  const rows = await b.bolt.run(
    `MATCH (s:Data_Bank {ref_id: $start})
     MATCH p = (s)-[:PARENT_OF*0..${Math.max(0, Math.floor(maxDepth))}]->(n:Data_Bank)
     WHERE $type IN labels(n) AND ${LIVE_NODE("n")}
       AND all(m IN nodes(p) WHERE ${LIVE_NODE("m")}) AND all(r IN relationships(p) WHERE ${LIVE_EDGE("r")})
     RETURN DISTINCT n.ref_id AS ref_id, n.name AS name
     LIMIT ${Math.max(1, Math.floor(maxNodes))}`,
    // Both bounds are inlined as integers: a JS number goes over bolt as a FLOAT, which LIMIT refuses.
    { start, type: nodeType },
  );
  return rows
    .filter((r) => typeof r["ref_id"] === "string")
    .map((r) => ({ ref_id: String(r["ref_id"]), name: typeof r["name"] === "string" ? (r["name"] as string) : "" }));
}

/** The batch's text, verbatim. */
export async function readText(b: Bolt, ids: string[]): Promise<Map<string, { description: string | null; docs: string | null }>> {
  const out = new Map<string, { description: string | null; docs: string | null }>();
  if (ids.length === 0) return out;
  const rows = await b.bolt.run(`UNWIND $ids AS id MATCH (n:Data_Bank {ref_id: id}) RETURN id, n.description AS description, n.docs AS docs`, { ids });
  const str = (v: unknown) => (v === null || v === undefined ? null : typeof v === "string" ? v : String(v));
  for (const r of rows) out.set(String(r["id"]), { description: str(r["description"]), docs: str(r["docs"]) });
  return out;
}

/** The batch as the agent reads it: one section per node, its text verbatim (fenced, so a heading inside docs stays inside). */
export function renderBatch(batch: Array<QueueNode & { description: string | null; docs: string | null }>): string {
  const fence = (t: string) => {
    let f = "````";
    while (t.includes(f)) f += "`";
    return `${f}\n${t}\n${f}`;
  };
  return batch
    .map((n, i) => [`### ${i + 1}. ${n.name}`, `ref_id: ${n.ref_id}`, "", "description:", fence(n.description ?? ""), "", "docs:", fence(n.docs ?? "")].join("\n"))
    .join("\n\n");
}

const EXAMPLE = `- id: queue
  type: janitor/queue
  config:
    start_ref_id: "{{ start.ref_id }}"
    cursor: "{{ input.cursor }}"    # the automation passes "{{ last.output.next_cursor }}"
    size: 10`;

export default defineStep({
  type: "janitor/queue",
  description:
    "Pick the nodes a graph-janitor run reviews, in rotation: the live `node_type` subtree under `start_ref_id` " +
    "(PARENT_OF, the start included) sorted by name, and the next `size` names after `cursor` (the last name the previous " +
    "run reviewed), wrapping to the first past the end; no cursor = the first `size`. Every node comes round once per " +
    "ceil(total / size) runs. Read over bolt; the batch carries each node's description and docs verbatim. " +
    "Output: { batch: [{ ref_id, name, node_type, description, docs }], ref_ids, visited (the batch without its text), text (the batch rendered for a prompt), next_cursor, from, total, wrapped, truncated }.\n\n" +
    EXAMPLE,
  input: z.object({
    start_ref_id: z.string().min(1).describe("the Concept whose subtree to queue"),
    cursor: z.string().optional().describe("name of the last node the previous run reviewed; empty or absent = start from the first name"),
    size: z.number().int().positive().default(10).describe("nodes per run"),
    node_type: z.string().default("Concept").describe("the only node type queued"),
    max_depth: z.number().int().nonnegative().default(25).describe("PARENT_OF hops below the start"),
    max_nodes: z.number().int().positive().default(5000).describe("cap on the subtree read; past it `truncated` is true"),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const b = await graphCtx(ctx as StepContext<StrutCapabilities>);
    const all = (await readSubtree(b, cfg.start_ref_id, cfg.node_type, cfg.max_depth, cfg.max_nodes + 1)).sort(byName);
    const truncated = all.length > cfg.max_nodes;
    const nodes = truncated ? all.slice(0, cfg.max_nodes) : all;
    const pick = pickBatch(nodes, cfg.cursor || undefined, cfg.size);
    const text = await readText(b, pick.batch.map((n) => n.ref_id));
    const batch = pick.batch.map((n) => ({ ref_id: n.ref_id, name: n.name, node_type: cfg.node_type, ...(text.get(n.ref_id) ?? { description: null, docs: null }) }));
    return {
      batch,
      ref_ids: batch.map((n) => n.ref_id),
      text: renderBatch(batch),
      visited: batch.map(({ ref_id, name, node_type }) => ({ ref_id, name, node_type })),
      next_cursor: pick.next_cursor,
      from: pick.from,
      total: nodes.length,
      wrapped: pick.wrapped,
      truncated,
    };
  },
});
