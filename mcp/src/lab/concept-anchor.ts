import type { ConceptSeedOutcome, GraphBackend, Neo4jBackend } from "strut";

/**
 * The anchor hive writes for every Concept a person approves (jarvis
 * migration 111), `HiveWorkspace -PROCESS-> <Concept>`, written here for
 * every Concept a boot's `seedConcepts` call (strut's) OWNS — created,
 * updated or kept: strut's pages (`Workflow Builder`, `Job`) and the lab's
 * under them — to the ONE workspace node hive's mirror writes into this
 * graph, so hive's graph walker reaches a seeded page from the workspace
 * node. Nothing in strut or the lab reads it. Strut knows no HiveWorkspace:
 * anchoring is the host's, over the outcomes the seed returns.
 *
 * A page someone took over (`skip`: no stamp, or another's) is theirs, never
 * anchored. Whether a page has a parent does not matter. No workspace node (a
 * standalone strut), or more than one (never guess): no edge, one log line.
 * Neo4j only: hive's mirror writes into no other graph.
 */
const WORKSPACE_TYPE = "HiveWorkspace";
const ANCHOR_EDGE = "PROCESS";

export async function anchorConcepts(graph: GraphBackend | undefined, outcomes: ConceptSeedOutcome[]): Promise<void> {
  if (graph?.kind !== "neo4j") return;
  const owned = outcomes.filter((o) => o.ref_id && (o.action === "create" || o.action === "update" || o.action === "keep"));
  if (!owned.length) return;
  let ws: { ref_id: string; label: string } | null;
  try {
    ws = await findWorkspace(graph);
  } catch (err) {
    console.warn(`[concepts] could not look up the ${WORKSPACE_TYPE} node:`, err instanceof Error ? err.message : err);
    return;
  }
  if (!ws) return;
  for (const c of owned) {
    try {
      const e = await graph.edges.write({ edge: ANCHOR_EDGE, source_ref_id: ws.ref_id, target_ref_id: c.ref_id! });
      if (e.created) console.log(`[concepts] anchored ${ws.label} -${ANCHOR_EDGE}-> ${c.name}`);
    } catch (err) {
      console.warn(`[concepts] could not anchor "${c.name}" to ${ws.label}:`, err instanceof Error ? err.message : err);
    }
  }
}

/** The ONE workspace node hive mirrors into this graph; null (with a log line) when there is none or more than one. */
async function findWorkspace(graph: Neo4jBackend): Promise<{ ref_id: string; label: string } | null> {
  const schema = await graph.schemas.schema(WORKSPACE_TYPE);
  if (!schema) {
    console.log(`[concepts] no ${WORKSPACE_TYPE} schema in the graph: Concepts not anchored`);
    return null;
  }
  const rows = await graph.bolt.run(
    `MATCH (w:\`${schema.type}\`)
     WHERE coalesce(w.namespace, 'default') = $ns AND w.deleted_at IS NULL AND NOT coalesce(w.is_deleted, false)
     RETURN w.ref_id AS ref_id, coalesce(w.slug, w.name, w.ref_id) AS label
     LIMIT 2`,
    { ns: graph.namespace },
  );
  if (rows.length === 0) {
    console.log(`[concepts] no ${WORKSPACE_TYPE} node in the graph (standalone strut?): Concepts not anchored`);
    return null;
  }
  if (rows.length > 1) {
    console.warn(`[concepts] more than one ${WORKSPACE_TYPE} node in namespace "${graph.namespace}": Concepts not anchored`);
    return null;
  }
  return { ref_id: String(rows[0]!.ref_id), label: String(rows[0]!.label) };
}
