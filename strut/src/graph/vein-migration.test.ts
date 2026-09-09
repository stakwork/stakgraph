import { describe, it, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Bolt } from "./bolt.js";
import { openGraphBackend, type GraphBackend } from "./backend.js";
import { seedStrutDomain, STRUT_MIGRATION_ID, DOMAIN_VECTOR_INDEX, DOMAIN_FULLTEXT_INDEX_V2 } from "./schema-seed.js";
import { STRUT_SCHEMAS, STRUT_DOMAIN_LABEL, searchableAttributes, vectorIndexedPairs, vectorIndexName, embeddingColumn } from "./strut-schemas.js";
import { graphSnapshot, testGraphConfig, wipeGraph, type GraphSnapshot } from "./test-util.js";
import { Neo4jWorkspaceStore } from "./workspace-store.js";
import {
  LEGACY_DOMAIN,
  LEGACY_DOMAIN_LABEL,
  VEIN_MIGRATION_ID,
  VeinMigrationCollision,
  legacyTypes,
  migrateVeinToStrut,
} from "./vein-migration.js";

const cfg = testGraphConfig();

describe("vein-migration (pure)", () => {
  it("derives the nine renames from the library", () => {
    const types = legacyTypes();
    assert.equal(types.length, STRUT_SCHEMAS.length);
    for (const t of types) {
      assert.ok(t.type.startsWith("Strut") && t.legacy.startsWith("Vein"), `${t.type} / ${t.legacy}`);
      assert.equal(t.legacy.slice(4), t.type.slice(5));
      assert.ok(t.nodeKeySpec.startsWith(t.keyPrefix), `${t.nodeKeySpec} starts with ${t.keyPrefix}`);
    }
    const run = types.find((t) => t.type === "StrutRun")!;
    assert.deepEqual(run, {
      type: "StrutRun",
      legacy: "VeinRun",
      nodeKeySpec: "strutrun-run_id",
      keyPrefix: "strutrun-",
      legacyKeyPrefix: "veinrun-",
    });
  });
});

// ── live ────────────────────────────────────────────────────────────────────

const VECTOR_OPTIONS =
  "OPTIONS { indexConfig: { `vector.dimensions`: 384, `vector.similarity_function`: 'cosine' } }";

/** Drop every constraint, then every index, that mentions one of `labels`. */
async function dropSchemaObjectsOn(bolt: Bolt, labels: string[]): Promise<void> {
  const list = labels.map((l) => `'${l}'`).join(", ");
  const cs = await bolt.run(
    `SHOW CONSTRAINTS YIELD name, labelsOrTypes WHERE any(l IN labelsOrTypes WHERE l IN [${list}]) RETURN name`,
  );
  for (const c of cs) await bolt.run(`DROP CONSTRAINT \`${c["name"]}\` IF EXISTS`);
  const is = await bolt.run(
    `SHOW INDEXES YIELD name, type, labelsOrTypes
     WHERE type <> 'LOOKUP' AND any(l IN labelsOrTypes WHERE l IN [${list}]) RETURN name`,
  );
  for (const i of is) await bolt.run(`DROP INDEX \`${i["name"]}\` IF EXISTS`);
}

/**
 * Turn a seeded + populated Strut database back into exactly what the
 * pre-rename code wrote: labels, node_keys, Schema nodes, constraint and
 * index names, ledger row. The rename (#1664) was purely mechanical, so the
 * inverse is too — which is what makes the round-trip assertion exact.
 */
async function demigrate(bolt: Bolt): Promise<void> {
  const types = legacyTypes();
  for (const t of types) {
    await bolt.run(
      `MATCH (n:\`${t.type}\`)
       SET n:\`${t.legacy}\`:\`${LEGACY_DOMAIN_LABEL}\`
       SET n.node_key = $old + substring(n.node_key, size($new))
       REMOVE n:\`${t.type}\`:\`${STRUT_DOMAIN_LABEL}\``,
      { old: t.legacyKeyPrefix, new: t.keyPrefix },
    );
    await bolt.run(
      `MATCH (s:Schema {type: $type}) SET s.type = $legacy, s.domain = $domain, s.node_key = $spec`,
      { type: t.type, legacy: t.legacy, domain: LEGACY_DOMAIN, spec: t.legacyKeyPrefix + t.nodeKeySpec.slice(t.keyPrefix.length) },
    );
  }
  await dropSchemaObjectsOn(bolt, [...types.map((t) => t.type), STRUT_DOMAIN_LABEL]);
  for (const t of types) {
    await bolt.run(
      `CREATE CONSTRAINT unique_${t.legacy.toLowerCase()}_node_key IF NOT EXISTS
       FOR (n:\`${t.legacy}\`) REQUIRE (n.node_key, n.namespace) IS UNIQUE`,
    );
    await bolt.run(`CREATE INDEX IF NOT EXISTS FOR (n:\`${t.legacy}\`) ON (n.node_key)`);
  }
  const legacyDomainVector = DOMAIN_VECTOR_INDEX.replace("strut", "vein");
  const legacyDomainFulltext = DOMAIN_FULLTEXT_INDEX_V2.replace("strut", "vein");
  await bolt.run(
    `CREATE VECTOR INDEX \`${legacyDomainVector}\` IF NOT EXISTS FOR (n:\`${LEGACY_DOMAIN_LABEL}\`) ON n.text_embeddings ${VECTOR_OPTIONS}`,
  );
  for (const { type, prop } of vectorIndexedPairs()) {
    const legacy = types.find((t) => t.type === type)!.legacy;
    await bolt.run(
      `CREATE VECTOR INDEX \`${vectorIndexName(type, prop).replace(/^strut/, "vein")}\` IF NOT EXISTS
       FOR (n:\`${legacy}\`) ON n.${embeddingColumn(prop)} ${VECTOR_OPTIONS}`,
    );
  }
  const props = [...searchableAttributes(), "node_key"].map((p) => `n.\`${p}\``).join(", ");
  await bolt.run(
    `CREATE FULLTEXT INDEX \`${legacyDomainFulltext}\` IF NOT EXISTS
     FOR (n:\`${LEGACY_DOMAIN_LABEL}\`) ON EACH [${props}]
     OPTIONS { indexConfig: { \`fulltext.analyzer\`: 'english' } }`,
  );
  await bolt.run(`MATCH (m:Migration {migration_id: $id}) DELETE m`, { id: STRUT_MIGRATION_ID });
  await bolt.run(`CREATE (:Migration {migration_id: 'vein_domain_seed_v1', executed_at: timestamp()})`);
}

/** Snapshot minus what legitimately differs across a round trip: the
 *  ledger rows, and Neo4j's auto-generated names for unnamed objects. */
function normalize(s: GraphSnapshot) {
  const anon = (name: unknown) => (typeof name === "string" && /^(index|constraint)_[0-9a-f]+$/.test(name) ? "<auto>" : name);
  const byJson = (a: unknown, b: unknown) => (JSON.stringify(a) < JSON.stringify(b) ? -1 : 1);
  return {
    nodes: s.nodes.filter((n) => !n.labels.includes("Migration")),
    rels: s.rels,
    constraints: (s.constraints as Array<Record<string, unknown>>).map((c) => ({ ...c, name: anon(c["name"]) })).sort(byJson),
    indexes: (s.indexes as Array<Record<string, unknown>>).map((i) => ({ ...i, name: anon(i["name"]) })).sort(byJson),
  };
}

const LEGACY_STEP = `import { z, defineStep } from "vein";
export default defineStep({ type: "legacy/step", input: z.object({}), output: z.any(), async run() { return 1; } });
`;

describe("migrateVeinToStrut (live Neo4j)", { skip: cfg ? false : "STRUT_TEST_NEO4J_URI not set" }, () => {
  let backend: GraphBackend;
  let scratch: string;
  const bolt = () => backend.bolt;
  const countLegacy = async () => {
    const [r] = await bolt().run(
      `OPTIONAL MATCH (n:\`${LEGACY_DOMAIN_LABEL}\`) WITH count(n) AS nodes
       OPTIONAL MATCH (s:Schema) WHERE s.type STARTS WITH 'Vein' RETURN nodes, count(s) AS schemas`,
    );
    return { nodes: Number(r!["nodes"]), schemas: Number(r!["schemas"]) };
  };
  const countStrut = async () =>
    Number((await bolt().run(`MATCH (n:\`${STRUT_DOMAIN_LABEL}\`) RETURN count(n) AS c`))[0]!["c"]);

  before(async () => {
    backend = await openGraphBackend(cfg!, { embeddings: false, skipBoot: true });
    scratch = await mkdtemp(join(tmpdir(), "strut-vein-mig-"));
  });
  after(async () => {
    await backend?.close();
    await rm(scratch, { recursive: true, force: true });
  });
  beforeEach(async () => {
    await wipeGraph(bolt());
  });

  it("round-trips a pre-rename database exactly, and the store sees the same workflows and steps", async () => {
    await seedStrutDomain(bolt());
    const ws = new Neo4jWorkspaceStore(backend, { materializeDir: join(scratch, "steps") });
    await ws.publishWorkflow("wf", "v1", { steps: [{ id: "a", type: "log", config: { message: "x" } }] }, "first");
    await ws.publishStep("legacy/step", LEGACY_STEP, "old");
    const before = {
      workflows: await ws.listWorkflows(),
      steps: await ws.listSteps(),
      versions: await ws.listStepVersions("legacy/step"),
      hash: await ws.getWorkflowHash("wf"),
    };
    const golden = normalize(await graphSnapshot(bolt()));

    await demigrate(bolt());
    assert.deepEqual(await countLegacy(), { nodes: 4, schemas: 9 });
    assert.equal(await countStrut(), 0);
    assert.deepEqual((await ws.listWorkflows()), [], "sanity: the renamed store cannot see legacy rows");

    const r = await migrateVeinToStrut(bolt());
    assert.equal(r.status, "migrated");
    assert.deepEqual(r.schemasRenamed.sort(), STRUT_SCHEMAS.map((s) => s.type).sort());
    assert.deepEqual(r.schemasDropped, []);
    assert.deepEqual(r.relabeled, { StrutWorkflow: 1, StrutWorkflowVersion: 1, StrutStep: 1, StrutStepVersion: 1 });
    assert.equal(r.droppedConstraints.length, 9);
    // 9 node_key range indexes + domain vector + domain fulltext + 4 per-stem vector.
    assert.equal(r.droppedIndexes.length, 15);
    assert.ok(r.droppedIndexes.includes("domain_vein_vector_index"));
    assert.ok(r.droppedIndexes.includes("veinstep_input_vector_index"));
    assert.equal(r.strays, 0);
    assert.deepEqual(await countLegacy(), { nodes: 0, schemas: 0 });

    const seed = await seedStrutDomain(bolt());
    assert.deepEqual(seed.createdSchemas, [], "renamed Schema nodes are found, not recreated");
    assert.deepEqual(seed.reconciled, {}, "renamed Schema nodes lack nothing");
    assert.deepEqual(normalize(await graphSnapshot(bolt())), golden);

    const fresh = new Neo4jWorkspaceStore(backend, { materializeDir: join(scratch, "steps2") });
    assert.deepEqual(await fresh.listWorkflows(), before.workflows);
    assert.deepEqual(await fresh.listSteps(), before.steps);
    assert.deepEqual(await fresh.listStepVersions("legacy/step"), before.versions);
    assert.equal(await fresh.getWorkflowHash("wf"), before.hash);
    assert.equal((await fresh.getStepSource("legacy/step"))?.code, LEGACY_STEP, "source bytes untouched");

    const ledger = await bolt().run(`MATCH (m:Migration) RETURN m.migration_id AS id ORDER BY id`);
    assert.deepEqual(ledger.map((x) => x["id"]), [STRUT_MIGRATION_ID, VEIN_MIGRATION_ID, "vein_domain_seed_v1"].sort());

    const snap = await graphSnapshot(bolt());
    assert.equal((await migrateVeinToStrut(bolt())).status, "already_done");
    assert.deepEqual(await graphSnapshot(bolt()), snap, "second run is a no-op");
  });

  it("refuses to run over a node_key collision and changes nothing", async () => {
    await seedStrutDomain(bolt());
    const ws = new Neo4jWorkspaceStore(backend, { materializeDir: join(scratch, "steps") });
    await ws.publishStep("s", LEGACY_STEP.replace("legacy/step", "s"));
    await bolt().run(
      `CREATE (:VeinStep:Node:Data_Bank:\`${LEGACY_DOMAIN_LABEL}\`
        {node_key: 'veinstep-s', namespace: $ns, ref_id: 'legacy-ref', step_type: 's'})`,
      { ns: cfg!.namespace },
    );
    const snap = await graphSnapshot(bolt());
    await assert.rejects(migrateVeinToStrut(bolt()), (e: unknown) => {
      assert.ok(e instanceof VeinMigrationCollision);
      assert.deepEqual(e.collisions, [{ legacy: "veinstep-s", key: "strutstep-s", labels: e.collisions[0]!.labels }]);
      assert.ok(e.collisions[0]!.labels.includes("StrutStep"));
      assert.match(e.message, /veinstep-s → strutstep-s/);
      return true;
    });
    assert.deepEqual(await graphSnapshot(bolt()), snap);
    assert.equal(Number((await bolt().run(`MATCH (m:Migration {migration_id: $id}) RETURN count(m) AS c`, { id: VEIN_MIGRATION_ID }))[0]!["c"]), 0);
  });

  it("drops a legacy Schema node whose Strut twin already exists", async () => {
    await seedStrutDomain(bolt());
    const [twin] = await bolt().run(`MATCH (s:Schema {type: 'StrutRun'}) RETURN s.ref_id AS ref_id`);
    await bolt().run(
      `MATCH (t:Schema {type: 'Thing'})
       CREATE (s:Schema {type: 'VeinRun', domain: 'Vein', parent: 'Thing', node_key: 'veinrun-run_id', ref_id: 'legacy-schema'})
       CREATE (s)-[:CHILD_OF {ref_id: 'legacy-edge'}]->(t)`,
    );
    const r = await migrateVeinToStrut(bolt());
    assert.equal(r.status, "migrated");
    assert.deepEqual(r.schemasDropped, ["StrutRun"]);
    assert.deepEqual(r.schemasRenamed, []);
    const rows = await bolt().run(`MATCH (s:Schema) WHERE toLower(s.type) = 'strutrun' RETURN s.ref_id AS ref_id`);
    assert.deepEqual(rows.map((x) => x["ref_id"]), [twin!["ref_id"]]);
    assert.deepEqual(await countLegacy(), { nodes: 0, schemas: 0 });
  });

  it("boot path: a fresh database is a no-op that stamps the ledger, and the seed still runs after it", async () => {
    await backend.close();
    backend = await openGraphBackend(cfg!, { embeddings: false });
    assert.equal(backend.veinMigration?.status, "nothing_to_do");
    assert.equal(backend.seed?.mode, "standalone");
    assert.equal((await migrateVeinToStrut(bolt())).status, "already_done");
  });
});
