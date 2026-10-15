import { z, defineStep, openGraphBackend, type StepContext, type StrutCapabilities } from "strut";

/**
 * janitor/queue — which nodes this janitor run reviews. Code picks them, not
 * the agent: left to walk the subtree itself, the agent read the start's
 * children every run and the levels below now and then (swarm38, Medicine,
 * 2026-10-14: a depth-2 Concept still naming "the benchmark" was read once
 * in 91 hourly runs).
 *
 * The queue is the tree under `start_ref_id` (live PARENT_OF edges, every
 * node on the way a live `node_type`, the start included) in TREE ORDER:
 * each node is keyed by its path of names from the start, and paths sort
 * segment by segment, a parent before its children. A node's whole subtree
 * comes before its next sibling, and siblings go by name, so ten in a row
 * are one branch (a parent with its children, or a run of siblings), not
 * ten nodes that share a first letter. A node with several parents in the
 * tree sits at its first path in that order.
 *
 * The run takes the next `size` nodes after `cursor` (the path of the last
 * node the previous run reviewed, its names joined by " › "), wrapping to
 * the first past the end. `next_cursor` is the last path taken; the
 * automation hands it back on its next fire (`cursor: "{{
 * last.output.next_cursor }}"` — `last` is the latest SUCCESSFUL run, so a
 * failed run's batch is taken again). Every node is reviewed once per
 * ceil(total / size) runs. A path, not a position: a Concept added or
 * deleted between runs never shifts anyone's turn, and a cursor whose node
 * is gone still sorts where it was. A node an apply run moves changes its
 * path, so it may come round twice in that pass or wait for the next. No
 * cursor: the first `size` nodes, the start first. A plain name (the cursor
 * before tree order) reads as a one-segment path: unless it is the start's
 * own name, the pass starts over.
 *
 * The batch comes with each node's `description` and `docs` verbatim, so
 * the agent is handed what it reviews instead of spending its steps finding
 * it, and with where the node sits: its path, its parent, its siblings'
 * names. Nothing is reported as accessed: a listing is not a node the caller
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

export interface TreeNode {
  ref_id: string;
  name: string;
  /** ref_ids of its live parents along live PARENT_OF edges; those outside the tree are ignored. */
  parents: string[];
}

export interface QueueNode {
  ref_id: string;
  name: string;
  /** Names from the start down to this node, both included. */
  path: string[];
}

export interface Placed extends QueueNode {
  /** The parent on its path; null for the start. */
  parent: { ref_id: string; name: string } | null;
}

export interface Pick<T extends QueueNode> {
  batch: T[];
  /** Index of the batch's first node in tree order. */
  from: number;
  /** The batch ran past the last node and went on from the first. */
  wrapped: boolean;
  next_cursor: string;
}

/** Name order, by code point: the same on every run and every machine. */
export const byName = (a: { ref_id: string; name: string }, b: { ref_id: string; name: string }) =>
  a.name < b.name ? -1 : a.name > b.name ? 1 : a.ref_id < b.ref_id ? -1 : a.ref_id > b.ref_id ? 1 : 0;

/** Paths segment by segment, each by code point; a path before every path it is a prefix of. */
export function comparePath(a: string[], b: string[]): number {
  for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i] !== b[i]) return a[i]! < b[i]! ? -1 : 1;
  return a.length - b.length;
}

/** Tree order: by path, same-named siblings by ref_id. */
export const byPath = (a: QueueNode, b: QueueNode) => comparePath(a.path, b.path) || (a.ref_id < b.ref_id ? -1 : a.ref_id > b.ref_id ? 1 : 0);

/** The cursor's separator. A name holding it reads as two segments: that cursor's place is close, not exact. */
export const PATH_SEP = " › ";

/**
 * Pure: every node reachable from `start` through the parents of `nodes`, in
 * tree order, each with the path and parent of the first path a depth-first
 * walk (children by name) reaches it by. Nodes the walk never reaches (their
 * parents fell past the read's cap) are left out.
 */
export function placeTree(nodes: TreeNode[], start: string): Placed[] {
  const byId = new Map(nodes.map((n) => [n.ref_id, n]));
  const root = byId.get(start);
  if (!root) return [];
  const children = new Map<string, TreeNode[]>();
  for (const n of nodes) {
    if (n.ref_id === start) continue;
    for (const p of new Set(n.parents)) {
      if (!byId.has(p)) continue;
      const kids = children.get(p);
      if (kids) kids.push(n);
      else children.set(p, [n]);
    }
  }
  for (const kids of children.values()) kids.sort(byName);
  const placed: Placed[] = [];
  const seen = new Set<string>();
  // Depth-first with an explicit stack, children pushed last-first: the first time a node is popped is its first path.
  const stack: Array<{ node: TreeNode; path: string[]; parent: TreeNode | null }> = [{ node: root, path: [root.name], parent: null }];
  while (stack.length) {
    const { node, path, parent } = stack.pop()!;
    if (seen.has(node.ref_id)) continue;
    seen.add(node.ref_id);
    placed.push({ ref_id: node.ref_id, name: node.name, path, parent: parent && { ref_id: parent.ref_id, name: parent.name } });
    const kids = children.get(node.ref_id) ?? [];
    for (let i = kids.length - 1; i >= 0; i--) if (!seen.has(kids[i]!.ref_id)) stack.push({ node: kids[i]!, path: [...path, kids[i]!.name], parent: node });
  }
  // Walk order already, but for same-named siblings, whose subtrees byPath interleaves: pickBatch needs the list sorted by path.
  return placed.sort(byPath);
}

/** Pure: names of the other children `parent` has among `nodes`, in name order; none without a parent (the start). */
export function siblingsOf(nodes: TreeNode[], parent: string | undefined, self: string): string[] {
  if (!parent) return [];
  return nodes.filter((n) => n.ref_id !== self && n.parents.includes(parent)).sort(byName).map((n) => n.name);
}

/** Pure: the next `size` of `sorted` (tree order) after the path `cursor`, wrapping; everything when it holds no more. */
export function pickBatch<T extends QueueNode>(sorted: T[], cursor: string | undefined, size: number): Pick<T> {
  const n = Math.max(1, Math.floor(size));
  if (sorted.length === 0) return { batch: [], from: 0, wrapped: false, next_cursor: cursor ?? "" };
  const at = cursor ? cursor.split(PATH_SEP) : undefined;
  const after = at ? sorted.findIndex((x) => comparePath(x.path, at) > 0) : 0;
  const from = after < 0 ? 0 : after;
  const take = Math.min(n, sorted.length);
  const batch = Array.from({ length: take }, (_, i) => sorted[(from + i) % sorted.length]!);
  return { batch, from, wrapped: from + take > sorted.length || (!!cursor && after < 0), next_cursor: batch.at(-1)!.path.join(PATH_SEP) };
}

const LIVE_NODE = (v: string) => `(${v}.deleted_at IS NULL AND (${v}.is_deleted IS NULL OR ${v}.is_deleted <> true))`;
const LIVE_EDGE = (r: string) => `((${r}.is_muted IS NULL OR ${r}.is_muted <> true) AND (${r}.is_deleted IS NULL OR ${r}.is_deleted <> true))`;

/**
 * Every live `node_type` node under `start` along live PARENT_OF edges
 * through live `node_type` nodes only, the start included: ref_id, name and
 * its live parents (placeTree keeps those inside the tree).
 */
export async function readSubtree(b: Bolt, start: string, nodeType: string, maxDepth: number, maxNodes: number): Promise<TreeNode[]> {
  const rows = await b.bolt.run(
    `MATCH (s:Data_Bank {ref_id: $start})
     MATCH p = (s)-[:PARENT_OF*0..${Math.max(0, Math.floor(maxDepth))}]->(n:Data_Bank)
     WHERE all(m IN nodes(p) WHERE $type IN labels(m) AND ${LIVE_NODE("m")}) AND all(r IN relationships(p) WHERE ${LIVE_EDGE("r")})
     WITH DISTINCT n
     LIMIT ${Math.max(1, Math.floor(maxNodes))}
     OPTIONAL MATCH (up:Data_Bank)-[e:PARENT_OF]->(n)
     WHERE ${LIVE_NODE("up")} AND ${LIVE_EDGE("e")}
     RETURN n.ref_id AS ref_id, n.name AS name, collect(up.ref_id) AS parents`,
    // Both bounds are inlined as integers: a JS number goes over bolt as a FLOAT, which LIMIT refuses.
    { start, type: nodeType },
  );
  return rows
    .filter((r) => typeof r["ref_id"] === "string")
    .map((r) => ({
      ref_id: String(r["ref_id"]),
      name: typeof r["name"] === "string" ? (r["name"] as string) : "",
      parents: Array.isArray(r["parents"]) ? (r["parents"] as unknown[]).filter((x): x is string => typeof x === "string") : [],
    }));
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

/** Siblings listed per node; past it the line says how many more (a parent with hundreds of children would swamp the prompt). */
export const MAX_SIBLINGS = 50;

/** The batch as the agent reads it: one section per node, where it sits, then its text verbatim (fenced, so a heading inside docs stays inside). */
export function renderBatch(batch: Array<Placed & { siblings: string[]; description: string | null; docs: string | null }>): string {
  const fence = (t: string) => {
    let f = "````";
    while (t.includes(f)) f += "`";
    return `${f}\n${t}\n${f}`;
  };
  const siblings = (s: string[]) =>
    s.length === 0 ? "none" : s.slice(0, MAX_SIBLINGS).join(", ") + (s.length > MAX_SIBLINGS ? ` … and ${s.length - MAX_SIBLINGS} more` : "");
  return batch
    .map((n, i) =>
      [
        `### ${i + 1}. ${n.name}`,
        `ref_id: ${n.ref_id}`,
        `path: ${n.path.join(PATH_SEP)}`,
        `parent: ${n.parent ? `${n.parent.name} (ref_id ${n.parent.ref_id})` : "none — the start of the sweep"}`,
        `siblings: ${siblings(n.siblings)}`,
        "",
        "description:",
        fence(n.description ?? ""),
        "",
        "docs:",
        fence(n.docs ?? ""),
      ].join("\n"),
    )
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
    "Pick the nodes a graph-janitor run reviews, in rotation: the live `node_type` tree under `start_ref_id` " +
    "(PARENT_OF, the start included) in tree order — by path of names from the start, a parent before its children, " +
    "siblings by name — and the next `size` nodes after `cursor` (the path of the last node the previous run reviewed, " +
    "names joined by \" › \"), wrapping to the first past the end; no cursor = the first `size`. Every node comes round " +
    "once per ceil(total / size) runs, and a batch is one branch. Read over bolt; the batch carries each node's path, " +
    "parent, siblings' names, description and docs verbatim. " +
    "Output: { batch: [{ ref_id, name, node_type, path, parent, siblings, description, docs }], ref_ids, visited (the batch without its text), text (the batch rendered for a prompt), next_cursor, from, total, wrapped, truncated }.\n\n" +
    EXAMPLE,
  input: z.object({
    start_ref_id: z.string().min(1).describe("the Concept whose subtree to queue"),
    cursor: z.string().optional().describe("path of the last node the previous run reviewed (names joined by \" › \"); empty or absent = start from the first node"),
    size: z.number().int().positive().default(10).describe("nodes per run"),
    node_type: z.string().default("Concept").describe("the only node type queued, and the only one walked through"),
    max_depth: z.number().int().nonnegative().default(25).describe("PARENT_OF hops below the start"),
    max_nodes: z.number().int().positive().default(5000).describe("cap on the subtree read; past it `truncated` is true"),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const b = await graphCtx(ctx as StepContext<StrutCapabilities>);
    const read = await readSubtree(b, cfg.start_ref_id, cfg.node_type, cfg.max_depth, cfg.max_nodes + 1);
    const truncated = read.length > cfg.max_nodes;
    const all = placeTree(read, cfg.start_ref_id);
    const nodes = truncated ? all.slice(0, cfg.max_nodes) : all;
    const pick = pickBatch(nodes, cfg.cursor || undefined, cfg.size);
    const text = await readText(b, pick.batch.map((n) => n.ref_id));
    const batch = pick.batch.map((n) => ({
      ...n,
      node_type: cfg.node_type,
      siblings: siblingsOf(read, n.parent?.ref_id, n.ref_id),
      ...(text.get(n.ref_id) ?? { description: null, docs: null }),
    }));
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
