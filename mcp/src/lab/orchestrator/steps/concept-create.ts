import { z, defineStep, composeNodeKey, withAccessedNodes, type GraphBackend, type StepContext, type StrutCapabilities } from "strut";

/**
 * concept/create — a NEW Concept on this swarm's graph as ONE gated step
 * the orchestrator proposes (orchestrator/seed.ts: `CONCEPT_WRITES`), in
 * place of the batch-triplet recipe the prompt handed the model every turn.
 * It composes what hive's own approver did (hive
 * src/lib/proposals/handleApproval.ts, `approveConceptCreate` +
 * `writeConceptAnchorEdges`). Lab-seeded, not strut's: strut knows no
 * HiveWorkspace, no gitree slug and no kind taxonomy (lab/concept-anchor.ts).
 *
 * One call, in order:
 *   1. the gitree address: `id = [<repo>/]<slug>` of the name (gitree's
 *      `generateSlug`, mcp/src/gitree/store/utils.ts) — stakgraph resolves
 *      a Concept by it and hive's Learn link is built from it, so it is
 *      always answered;
 *   2. the parent, resolved BEFORE anything is written (a bad parent never
 *      leaves a half-made Concept): by ref_id, else by exact name (a
 *      Concept's node_key IS its name); it must be a Concept and not the
 *      new one itself;
 *   3. the node, `graph/create-node`'s create-or-merge: an existing name is
 *      a Warning with the existing ref_id and its docs LEFT ALONE (changing
 *      a Concept is graph/edit-node's), never a duplicate;
 *   4. `<parent> -PARENT_OF-> <concept>` when a parent was given
 *      (idempotent; a failure rolls back a node THIS call created, as hive
 *      did, never one a merge landed on);
 *   5. the workspace anchor, best-effort: `HiveWorkspace -<EDGE>-> <concept>`
 *      to the ONE workspace node hive mirrors into this namespace, EDGE by
 *      `kind` (hive's table, jarvis migration 111), HAS_CONCEPT by default.
 *      No workspace node (a standalone strut), more than one (never guess)
 *      or no Neo4j: `anchored: false`, one log line, the step still
 *      succeeds; a failed edge write is a Warning with a message.
 *
 * A seeded step: imports only "strut", and the graph is `ctx.services.graph`
 * — the one createStrut wired — never opened here (strut's step lint).
 */

const STEP = "concept/create";
const TYPE = "Concept";
const PARENT_EDGE = "PARENT_OF";
const WORKSPACE_TYPE = "HiveWorkspace";

export const CONCEPT_KINDS = ["preference", "best_practice", "gotcha", "process", "general"] as const;
export type ConceptKind = (typeof CONCEPT_KINDS)[number];

/** hive's `CONCEPT_KIND_EDGE` (jarvis migration 111): the kind → the edge
 *  from the workspace node. `general`, and no kind at all, is HAS_CONCEPT. */
export const KIND_EDGE: Record<ConceptKind, string> = {
  preference: "PREFERENCE",
  best_practice: "BEST_PRACTICE",
  gotcha: "GOTCHA",
  process: "PROCESS",
  general: "HAS_CONCEPT",
};

/** gitree's `generateSlug`, as hive mirrors it. */
export function conceptSlug(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

type Found = { ref_id: string; node_type?: string; name: string; id?: string };
type Envelope = Awaited<ReturnType<GraphBackend["reader"]["getNode"]>>;

function graphCtx(ctx?: StepContext<StrutCapabilities>): GraphBackend {
  const g = ctx?.services?.graph;
  if (!g) throw new Error("no graph on the services bag (ctx.services.graph): run inside a strut with a graph-backed workspace");
  return g;
}

const str = (v: unknown): string | undefined => (typeof v === "string" && v.trim() ? v : undefined);
const msg = (e: unknown): string => (e instanceof Error ? e.message : String(e));

/** The `code` of a GraphValidationError / GraphReadError, duck-typed on `name`. */
function graphErrorCode(e: unknown): string | undefined {
  const n = (e as { name?: unknown } | null)?.name;
  return n === "GraphValidationError" || n === "GraphReadError" ? String((e as { code: unknown }).code) : undefined;
}

function describe(e: unknown): string {
  const code = graphErrorCode(e);
  return code ? `${code}: ${msg(e)}` : msg(e);
}

function found(n: Envelope): Found | null {
  if (!n) return null;
  const p = n.properties ?? {};
  return { ref_id: n.ref_id, node_type: n.node_type, name: str(p["name"]) ?? n.name ?? "", id: str(p["id"]) };
}

/** The parent: a ref_id first, else the Concept under that exact name. */
async function findParent(b: GraphBackend, schema: { type: string; node_key: string; exact_key?: boolean }, parent: string, namespace: string): Promise<Found | null> {
  const byRef = found(await b.reader.getNode(parent));
  if (byRef) return byRef;
  return found(await b.reader.getNodeByKey(schema.type, composeNodeKey(schema, { name: parent }), namespace));
}

/** The ONE workspace node hive mirrors into this namespace (lab
 *  concept-anchor.ts `findWorkspace`); null, with a log line, when there is
 *  none, more than one, no schema for it, or no Neo4j to ask. */
async function findWorkspace(b: GraphBackend, namespace: string): Promise<{ ref_id: string; label: string } | null> {
  if (b.kind !== "neo4j") {
    console.log(`[${STEP}] a ${b.kind} graph holds no ${WORKSPACE_TYPE}: not anchored`);
    return null;
  }
  const schema = await b.schemas.schema(WORKSPACE_TYPE);
  if (!schema) {
    console.log(`[${STEP}] no ${WORKSPACE_TYPE} schema in the graph: not anchored`);
    return null;
  }
  const rows = await b.bolt.run(
    `MATCH (w:\`${schema.type}\`)
     WHERE coalesce(w.namespace, 'default') = $ns AND w.deleted_at IS NULL AND NOT coalesce(w.is_deleted, false)
     RETURN w.ref_id AS ref_id, coalesce(w.slug, w.name, w.ref_id) AS label
     LIMIT 2`,
    { ns: namespace },
  );
  if (rows.length === 0) {
    console.log(`[${STEP}] no ${WORKSPACE_TYPE} node in namespace "${namespace}" (standalone strut?): not anchored`);
    return null;
  }
  if (rows.length > 1) {
    console.warn(`[${STEP}] more than one ${WORKSPACE_TYPE} node in namespace "${namespace}": not anchored`);
    return null;
  }
  return { ref_id: String(rows[0]!["ref_id"]), label: String(rows[0]!["label"]) };
}

export default defineStep({
  type: STEP,
  description:
    "Create a NEW Concept in this workspace's knowledge graph: one call writes the Concept node, places it under its " +
    "parent Concept (PARENT_OF) and anchors it to the workspace (HiveWorkspace -<kind's edge>-> Concept). " +
    "A Concept is a reusable unit of knowledge — a preference, a best practice, a gotcha, a process — with a name, a " +
    "one-line description and a markdown body in `docs`. REUSE first: graph/graph-search (type Concept) for the name; " +
    "a Concept that already exists under that name is returned as a Warning with its ref_id and its docs untouched — " +
    "to change a Concept use graph/edit-node on its ref_id. Never write the deprecated `documentation` field.",
  input: z.object({
    name: z
      .string()
      .describe("The Concept's name — its key in the graph (a Concept's node_key is its name) and what its slug is made from. A name that already exists merges into that Concept instead of creating a duplicate."),
    docs: z
      .string()
      .describe("The Concept's documentation: markdown, whole as it should read. This is the body stakgraph indexes and hive's Learn page shows. Never `documentation` (deprecated, unindexed)."),
    description: z.string().optional().describe("One line: what this Concept is, as lists and search hits show it."),
    parent: z
      .string()
      .optional()
      .describe("The parent Concept: its ref_id (preferred, from graph/graph-search) or its exact name. The new Concept is placed under it (PARENT_OF). Omit for a top-level Concept."),
    kind: z
      .enum(CONCEPT_KINDS)
      .optional()
      .describe("What kind of knowledge this is; picks the edge from the workspace node: preference → PREFERENCE, best_practice → BEST_PRACTICE, gotcha → GOTCHA, process → PROCESS, general (or omitted) → HAS_CONCEPT."),
    repo: z
      .string()
      .optional()
      .describe("`owner/repo` when the knowledge is about one repository's code: prefixes the Concept's id (`owner/repo/<slug>`) and is kept on the node. Omit for workspace-wide knowledge."),
    namespace: z
      .string()
      .optional()
      .describe("Namespace (data partition) to write into — must be registered. Not an access-control boundary."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const name = cfg.name.trim();
    if (!name) return `${STEP} failed — name is required`;
    if (!cfg.docs.trim()) return `${STEP} failed — docs is required: the Concept's body, markdown`;
    const slug = conceptSlug(name);
    if (!slug) return `${STEP} failed — the name must contain a letter or a digit (its slug would be empty)`;
    const repo = cfg.repo?.trim() || undefined;
    const id = repo ? `${repo}/${slug}` : slug;
    const description = cfg.description?.trim() || undefined;
    const parentArg = cfg.parent?.trim() || undefined;
    const edge = KIND_EDGE[cfg.kind ?? "general"];

    try {
      const b = graphCtx(ctx as StepContext<StrutCapabilities>);
      const namespace = await b.reader.resolveNamespace(cfg.namespace);
      const schema = await b.schemas.schema(TYPE);
      if (!schema) return `${STEP} failed — no "${TYPE}" schema in this graph`;

      // The parent first: nothing is written until it is known to be a Concept, and another one.
      let parent: Found | undefined;
      if (parentArg) {
        const p = await findParent(b, schema, parentArg, namespace);
        if (!p) return `${STEP} failed — parent Concept not found: "${parentArg}" is neither a ref_id nor the exact name of a Concept (graph/graph-search, type Concept, finds one)`;
        if (p.node_type !== schema.type) return `${STEP} failed — the parent "${parentArg}" is a ${p.node_type ?? "node"}, not a Concept`;
        if (p.name === name || (p.id !== undefined && p.id === id)) return `${STEP} failed — a concept cannot be its own parent`;
        parent = p;
      }

      const r = await b.nodes.write(
        { type: TYPE, data: { id, name, docs: cfg.docs, ...(description ? { description } : {}), ...(repo ? { repo } : {}) } },
        "create",
        { namespace },
      );
      // Merge-by-name landed on the parent itself (same key, another spelling).
      if (parent && r.ref_id === parent.ref_id) return `${STEP} failed — a concept cannot be its own parent: "${name}" is the parent itself (ref_id ${r.ref_id})`;
      const existed = r.outcome === "existing";
      const messages: string[] = [];
      if (existed) messages.push(`Concept "${name}" already exists (ref_id ${r.ref_id}): its docs were left as they are — change a Concept with graph/edit-node on its ref_id`);
      if (r.outcome === "restored") messages.push("Concept restored (it had been deleted)");

      if (parent) {
        try {
          await b.edges.write({ edge: PARENT_EDGE, source_ref_id: parent.ref_id, target_ref_id: r.ref_id });
        } catch (e) {
          // hive's compensating rollback: only a node THIS call made, never one a merge landed on.
          let rolledBack = "";
          if (r.outcome === "created") {
            try {
              await b.nodes.softDelete(r.ref_id);
              rolledBack = "; the new Concept was removed again";
            } catch (e2) {
              rolledBack = `; the new Concept (ref_id ${r.ref_id}) could not be removed: ${msg(e2)}`;
            }
          }
          return `${STEP} failed — could not link ${parent.name} -${PARENT_EDGE}-> ${name}: ${describe(e)}${rolledBack}`;
        }
      }

      // The anchor, best-effort: never fails the step.
      let ws: { ref_id: string; label: string } | null = null;
      let anchored = false;
      let anchorFailed = false;
      try {
        ws = await findWorkspace(b, namespace);
        if (ws) {
          await b.edges.write({ edge, source_ref_id: ws.ref_id, target_ref_id: r.ref_id });
          anchored = true;
        }
      } catch (e) {
        anchorFailed = true;
        console.warn(`[${STEP}] could not anchor "${name}" to the ${WORKSPACE_TYPE}:`, msg(e));
        messages.push(`not anchored to the workspace (${WORKSPACE_TYPE} -${edge}-> ${name} failed: ${describe(e)})`);
      }

      return withAccessedNodes(
        {
          status: existed || anchorFailed ? "Warning" : "Success",
          ref_id: r.ref_id,
          id,
          name,
          ...(parent ? { parent_ref_id: parent.ref_id } : {}),
          anchored,
          ...(messages.length ? { messages } : {}),
        },
        [
          { ref_id: r.ref_id, node_type: schema.type, name },
          parent ? { ref_id: parent.ref_id, node_type: schema.type, name: parent.name } : null,
          ws ? { ref_id: ws.ref_id, node_type: WORKSPACE_TYPE, name: ws.label } : null,
        ],
      );
    } catch (e) {
      return `${STEP} failed — ${describe(e)}`;
    }
  },
});
