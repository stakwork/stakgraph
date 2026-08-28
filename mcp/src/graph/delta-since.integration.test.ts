/**
 * Integration test for the `$since` delta filter in `listQueryForLabel()`
 * (exercised through the real read path: db.nodes_by_type, as used by the
 * `GET /graph` endpoint).
 *
 * Seeds one legacy-seconds node (7-decimal string — the old Rust ingest
 * format) and one new epoch-ms Integer node (what nowEpochMs() writes), then
 * asserts:
 *   1. a millisecond `$since` cursor older than both returns BOTH (the
 *      regression: the old `toFloat(...) >= $since` comparison silently
 *      dropped every legacy-seconds node once the frontend sent ms cursors);
 *   2. a ms cursor between the two returns only the ms node (legacy seconds
 *      must not over-match a newer ms cursor);
 *   3. mixed-format nodes sort by *normalized* ms (ORDER BY);
 *   4. returned Integer timestamps are coerced to plain numbers (no
 *      `{low, high}` leak).
 *
 * Runs only against a live Neo4j at bolt://${NEO4J_HOST} (defaults
 * localhost:7687 / neo4j / testtest, matching createNeo4jDriver). Skips when
 * NO_DB=true or when unreachable, so `npm run test:node` stays DB-free.
 * Run standalone with:
 *   npx tsx --test --test-timeout=60000 src/graph/delta-since.integration.test.ts
 */
import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import neo4j from "neo4j-driver";

import { db } from "./neo4j.js";
import { toReturnNode } from "./utils.js";

const noDb = process.env.NO_DB === "true" || process.env.NO_DB === "1";
const host = process.env.NEO4J_HOST || "localhost:7687";
const user = process.env.NEO4J_USER || "neo4j";
const pswd = process.env.NEO4J_PASSWORD || "testtest";
const uri = `bolt://${host}`;

// --- reachability probe -----------------------------------------------------
let reachable = false;
const probeDriver = neo4j.driver(uri, neo4j.auth.basic(user, pswd), {
  connectionTimeout: 3000,
});
try {
  if (!noDb) {
    await probeDriver.verifyConnectivity();
    reachable = true;
    console.log(`===> delta-since integration test using ${uri}`);
  }
} catch {
  console.log(`===> Neo4j unreachable at ${uri} — skipping integration tests`);
}
await probeDriver.close().catch(() => {});

const skipReason: string | false = noDb
  ? "NO_DB=true"
  : reachable
    ? false
    : `Neo4j unreachable at ${uri}`;

// --- fixtures ---------------------------------------------------------------
const NOW_MS = Date.now();
const RUN = `t${NOW_MS}`;
const LEGACY_KEY = `test-delta-legacy-${RUN}`;
const MS_KEY = `test-delta-ms-${RUN}`;
// legacy: epoch-seconds as a 7-decimal string (old Rust ingest format),
// stored 2h ago
const LEGACY_TS = ((NOW_MS - 2 * 3600_000) / 1000).toFixed(7);
// new: epoch-ms Integer (nowEpochMs format), stored 1h ago
const MS_TS = neo4j.int(NOW_MS - 1 * 3600_000);

let driver: neo4j.Driver | null = null;
async function run(cypher: string, params: Record<string, unknown> = {}) {
  const session = driver!.session();
  try {
    return await session.run(cypher, params);
  } finally {
    await session.close();
  }
}

before(async () => {
  if (!reachable) return;
  driver = neo4j.driver(uri, neo4j.auth.basic(user, pswd));
  await run(
    `MERGE (f:Data_Bank:Hint {node_key: $key})
     ON CREATE SET f.ref_id = $ref_id, f.name = $name,
                   f.date_added_to_graph = $ts`,
    { key: LEGACY_KEY, ref_id: `ref-${LEGACY_KEY}`, name: "delta-legacy", ts: LEGACY_TS },
  );
  await run(
    `MERGE (f:Data_Bank:Hint {node_key: $key})
     ON CREATE SET f.ref_id = $ref_id, f.name = $name,
                   f.date_added_to_graph = $ts`,
    { key: MS_KEY, ref_id: `ref-${MS_KEY}`, name: "delta-ms", ts: MS_TS },
  );
});

after(async () => {
  if (!reachable) return;
  await run(`MATCH (n) WHERE n.node_key IN [$lk, $mk] DETACH DELETE n`, {
    lk: LEGACY_KEY,
    mk: MS_KEY,
  });
  await driver?.close();
});

async function fetchNodes(sinceMs: number) {
  return db.nodes_by_type("Hint", undefined, 50000, sinceMs);
}

describe("delta filter: ms $since over mixed legacy-seconds / new-ms stored values", () => {
  it(
    "a ms cursor older than both returns the legacy-seconds AND new-ms nodes",
    { skip: skipReason },
    async () => {
      const nodes = await fetchNodes(NOW_MS - 3 * 3600_000);
      const keys = new Set(nodes.map((n) => n.properties.node_key));
      assert.ok(
        keys.has(LEGACY_KEY),
        "legacy-seconds node dropped by the ms $since filter",
      );
      assert.ok(keys.has(MS_KEY), "new-ms node dropped by the ms $since filter");
    },
  );

  it(
    "a ms cursor between the two returns only the new-ms node",
    { skip: skipReason },
    async () => {
      const nodes = await fetchNodes(NOW_MS - 1.5 * 3600_000); // between -2h and -1h
      const keys = new Set(nodes.map((n) => n.properties.node_key));
      assert.ok(keys.has(MS_KEY), "ms node should match a cursor older than it");
      assert.ok(
        !keys.has(LEGACY_KEY),
        "legacy node must not over-match a newer ms cursor",
      );
    },
  );

  it(
    "orders mixed-format nodes by normalized ms (DESC)",
    { skip: skipReason },
    async () => {
      const nodes = await fetchNodes(NOW_MS - 3 * 3600_000);
      const msIdx = nodes.findIndex((n) => n.properties.node_key === MS_KEY);
      const legacyIdx = nodes.findIndex(
        (n) => n.properties.node_key === LEGACY_KEY,
      );
      assert.ok(msIdx !== -1 && legacyIdx !== -1);
      assert.ok(
        msIdx < legacyIdx,
        `expected ms node (idx ${msIdx}) before legacy node (idx ${legacyIdx})`,
      );
    },
  );

  it(
    "returned Integer timestamps are coerced to plain numbers (no {low, high} leak)",
    { skip: skipReason },
    async () => {
      const nodes = await fetchNodes(NOW_MS - 3 * 3600_000);
      const msNode = nodes.find((n) => n.properties.node_key === MS_KEY)!;
      assert.ok(msNode, "ms node not found");
      const ts = msNode.properties.date_added_to_graph as unknown;
      assert.equal(typeof ts, "number", `expected number, got ${typeof ts}`);
      assert.equal(ts, MS_TS.toNumber());
      // The shaped API response (toReturnNode) must never carry a raw
      // Integer object for this field. (Node.identity is also an Integer
      // internally, but it is not part of the ReturnNode wire format.)
      const json = JSON.stringify(toReturnNode(msNode));
      assert.ok(!json.includes('"low"'), `leaked Integer object: ${json}`);
    },
  );
});
