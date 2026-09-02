/**
 * Helpers for graph tests that need a live Neo4j.
 *
 * Tests are opt-in: they run only when `VEIN_TEST_NEO4J_URI` is set, and
 * they WIPE that database between cases — point it at a throwaway
 * container, never at a jarvis instance. Example:
 *
 *   docker run -d --name vein-neo4j-test -p 7688:7687 \
 *     -e NEO4J_AUTH=neo4j/veintest neo4j:5
 *   VEIN_TEST_NEO4J_URI=bolt://localhost:7688 VEIN_TEST_NEO4J_PASSWORD=veintest \
 *     npm run test:graph
 */
import { Bolt, type GraphConfig } from "./bolt.js";

export function testGraphConfig(): GraphConfig | null {
  const uri = process.env["VEIN_TEST_NEO4J_URI"];
  if (!uri) return null;
  return {
    uri,
    user: process.env["VEIN_TEST_NEO4J_USER"] ?? "neo4j",
    password: process.env["VEIN_TEST_NEO4J_PASSWORD"] ?? "",
    namespace: process.env["VEIN_TEST_NEO4J_NAMESPACE"] ?? "default",
  };
}

/** Drop every node, relationship, constraint, and index. */
export async function wipeGraph(bolt: Bolt): Promise<void> {
  await bolt.run(`MATCH (n) DETACH DELETE n`);
  const constraints = await bolt.run(`SHOW CONSTRAINTS YIELD name RETURN name`);
  for (const c of constraints) await bolt.run(`DROP CONSTRAINT \`${c["name"]}\` IF EXISTS`);
  const indexes = await bolt.run(`SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP' RETURN name`);
  for (const i of indexes) await bolt.run(`DROP INDEX \`${i["name"]}\` IF EXISTS`);
}

export interface GraphSnapshot {
  nodes: Array<{ labels: string[]; properties: Record<string, unknown> }>;
  rels: Array<{ type: string; properties: Record<string, unknown>; from: unknown; to: unknown }>;
  constraints: unknown[];
  indexes: unknown[];
}

function canon(v: unknown): string {
  return JSON.stringify(v, (_k, x) => (x && typeof x === "object" && !Array.isArray(x)
    ? Object.fromEntries(Object.entries(x as Record<string, unknown>).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)))
    : x));
}

function sortByCanon<T>(xs: T[]): T[] {
  return [...xs].sort((a, b) => (canon(a) < canon(b) ? -1 : canon(a) > canon(b) ? 1 : 0));
}

/**
 * Canonical, order-independent picture of the whole database: every node
 * (labels + all properties), every relationship (type + properties + the
 * `type`/`ref_id`/`node_key` of its endpoints), every constraint and index.
 * Deep-equal two snapshots to prove a run was a no-op.
 */
export async function graphSnapshot(bolt: Bolt): Promise<GraphSnapshot> {
  const nodes = await bolt.run(`MATCH (n) RETURN labels(n) AS labels, properties(n) AS properties`);
  const rels = await bolt.run(
    `MATCH (a)-[r]->(b)
     RETURN type(r) AS type, properties(r) AS properties,
            {type: a.type, ref_id: a.ref_id, node_key: a.node_key} AS from,
            {type: b.type, ref_id: b.ref_id, node_key: b.node_key} AS to`,
  );
  const constraints = await bolt.run(
    `SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties
     RETURN name, type, entityType, labelsOrTypes, properties`,
  );
  const indexes = await bolt.run(
    `SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, options
     WHERE type <> 'LOOKUP'
     RETURN name, type, entityType, labelsOrTypes, properties, options`,
  );
  return {
    nodes: sortByCanon(nodes.map((r) => ({ labels: ([...(r["labels"] as string[])]).sort(), properties: r["properties"] as Record<string, unknown> }))),
    rels: sortByCanon(rels.map((r) => ({ type: r["type"] as string, properties: r["properties"] as Record<string, unknown>, from: r["from"], to: r["to"] }))),
    constraints: sortByCanon(constraints),
    indexes: sortByCanon(indexes),
  };
}

/** Names of all constraints / non-lookup indexes. */
export async function schemaObjectNames(bolt: Bolt): Promise<{ constraints: string[]; indexes: string[] }> {
  const c = await bolt.run(`SHOW CONSTRAINTS YIELD name RETURN name`);
  const i = await bolt.run(`SHOW INDEXES YIELD name, type WHERE type <> 'LOOKUP' RETURN name`);
  return { constraints: c.map((r) => r["name"] as string).sort(), indexes: i.map((r) => r["name"] as string).sort() };
}
