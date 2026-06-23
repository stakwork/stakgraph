/**
 * Feature -> Concept migration test harness.
 *
 * Seeds a Neo4j database with LEGACY gitree data (using the old `:Feature` /
 * `:FeaturesMetadata` labels and `featureId` / `relatedFeatures` Clue
 * properties + old indexes), then runs `GraphStorage.initialize()` (which
 * triggers `migrateFeatureToConcept()`), and finally asserts the data was
 * correctly relabeled and that the renamed query methods work end-to-end.
 *
 * Run against a LOCAL / DISPOSABLE Neo4j (the migration is global — every
 * `:Feature` node in the DB is relabeled). The easiest way:
 *
 *   cd mcp && docker compose -f neo4j.yaml up -d      # wait until healthy
 *   NEO4J_HOST=localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=testtest \
 *     npx tsx scripts/test-concept-migration.ts
 *
 * Or just use the wrapper:  ./scripts/test-concept-migration.sh
 *
 * Env:
 *   KEEP=1   leave the seeded test data in the DB for manual inspection.
 */
import neo4j from "neo4j-driver";
import { createNeo4jDriver } from "../src/utils/neo4jRetry.js";
import { GraphStorage } from "../src/gitree/store/index.js";

const REPO = "migtest/concept-migration";
const CONCEPT_ID = `${REPO}/auth-system`;
const CLUE_ID = `${REPO}/clue-jwt`;
const FILE_REF = "migtest-file-auth-ts";
const FN_REF = "migtest-fn-verify-token";
const now = Math.floor(Date.now() / 1000);

let passed = 0;
let failed = 0;
function check(name: string, cond: boolean, detail = "") {
  if (cond) {
    passed++;
    console.log(`  ✅ ${name}`);
  } else {
    failed++;
    console.log(`  ❌ ${name}${detail ? ` — ${detail}` : ""}`);
  }
}

const OLD_INDEXES = ["feature_id_index", "feature_repo_index", "clue_feature_index"];

async function cleanup(driver: neo4j.Driver) {
  const s = driver.session();
  try {
    await s.run(
      `MATCH (n) WHERE n.id STARTS WITH $p OR n.repo = $repo OR n.ref_id IN $refs DETACH DELETE n`,
      { p: `${REPO}/`, repo: REPO, refs: [FILE_REF, FN_REF] }
    );
    for (const idx of OLD_INDEXES) {
      await s.run(`DROP INDEX ${idx} IF EXISTS`).catch(() => {});
    }
  } finally {
    await s.close();
  }
}

async function seed(driver: neo4j.Driver) {
  const s = driver.session();
  try {
    // Recreate the OLD indexes so we can verify they get dropped.
    await s.run(`CREATE INDEX feature_id_index IF NOT EXISTS FOR (f:Feature) ON (f.id)`);
    await s.run(`CREATE INDEX feature_repo_index IF NOT EXISTS FOR (f:Feature) ON (f.repo)`);
    await s.run(`CREATE INDEX clue_feature_index IF NOT EXISTS FOR (c:Clue) ON (c.featureId)`);

    await s.run(
      `
      // PR + Commit (unchanged by migration)
      CREATE (pr:Data_Bank:PullRequest {
        id: $repo + '/pr-1', number: 1, repo: $repo, name: $repo + '/pr-1',
        title: 'Add JWT auth', summary: 'Adds verifyToken', url: 'https://github.com/' + $repo + '/pull/1',
        files: ['src/auth.ts'], date: $now
      })
      CREATE (cm:Data_Bank:Commit {
        id: $repo + '/commit-abc1234', sha: 'abc1234def', repo: $repo, name: $repo + '/commit-abc1234',
        message: 'jwt', summary: 'jwt', author: 'dev', url: 'x', files: ['src/auth.ts'], date: $now
      })

      // LEGACY Feature node (the thing being migrated)
      CREATE (f:Data_Bank:Feature {
        id: $conceptId, repo: $repo, ref_id: 'ref-concept-1', name: 'Authentication System',
        description: 'JWT based auth', prNumbers: [1], commitShas: ['abc1234def'],
        docs: 'Auth uses verifyToken to validate JWTs.', date: $now, namespace: 'default'
      })

      // LEGACY FeaturesMetadata node
      CREATE (m:Data_Bank:FeaturesMetadata {
        namespace: 'default', repo: $repo, lastProcessedPR: 1, ref_id: 'ref-meta-1'
      })

      // LEGACY Clue node with featureId + relatedFeatures
      CREATE (clue:Data_Bank:Clue {
        id: $clueId, repo: $repo, featureId: $conceptId, relatedFeatures: [$conceptId],
        type: 'utility', title: 'JWT utils', content: 'token helpers', entities: '{}',
        files: ['src/auth.ts'], keywords: ['jwt'], relatedClues: [], dependsOn: [],
        createdAt: $now, updatedAt: $now, namespace: 'default'
      })

      // File + contained Function (for MODIFIES / provenance)
      CREATE (file:Data_Bank:File { ref_id: $fileRef, name: 'auth.ts', file: $repo + '/src/auth.ts' })
      CREATE (fn:Data_Bank:Function { ref_id: $fnRef, name: 'verifyToken', file: $repo + '/src/auth.ts', start: 1, end: 20 })

      // Relationships that must survive the relabel
      CREATE (pr)-[:TOUCHES]->(f)
      CREATE (cm)-[:TOUCHES]->(f)
      CREATE (clue)-[:RELEVANT_TO]->(f)
      CREATE (f)-[:MODIFIES { importance: 0.9 }]->(file)
      CREATE (file)-[:CONTAINS]->(fn)
      `,
      { repo: REPO, conceptId: CONCEPT_ID, clueId: CLUE_ID, fileRef: FILE_REF, fnRef: FN_REF, now }
    );
  } finally {
    await s.close();
  }
}

async function indexNames(driver: neo4j.Driver): Promise<string[]> {
  const s = driver.session();
  try {
    const r = await s.run(`SHOW INDEXES YIELD name RETURN name`);
    return r.records.map((rec) => rec.get("name"));
  } finally {
    await s.close();
  }
}

async function scalar(driver: neo4j.Driver, query: string, params: any = {}): Promise<number> {
  const s = driver.session();
  try {
    const r = await s.run(query, params);
    const v = r.records[0]?.get(0);
    return v?.toNumber ? v.toNumber() : v ?? 0;
  } finally {
    await s.close();
  }
}

async function main() {
  const driver = createNeo4jDriver();

  console.log(`\n=== Feature → Concept migration test ===`);
  console.log(`Neo4j: bolt://${process.env.NEO4J_HOST || "localhost:7687"}\n`);

  console.log("1) Cleanup any prior test data...");
  await cleanup(driver);

  console.log("2) Seed LEGACY :Feature data + old indexes...");
  await seed(driver);

  // Sanity: legacy data is present before migration
  const preFeatures = await scalar(driver, `MATCH (f:Feature {id:$id}) RETURN count(f)`, { id: CONCEPT_ID });
  const preMeta = await scalar(driver, `MATCH (m:FeaturesMetadata {repo:$r}) RETURN count(m)`, { r: REPO });
  console.log(`   seeded: ${preFeatures} :Feature, ${preMeta} :FeaturesMetadata`);
  check("legacy :Feature seeded", preFeatures === 1);
  check("legacy :FeaturesMetadata seeded", preMeta === 1);

  console.log("\n3) Run GraphStorage.initialize() (triggers migration)...");
  const storage = new GraphStorage();
  await storage.initialize();

  console.log("\n4) Assert label / property migration:");
  check(
    "no :Feature nodes remain (global)",
    (await scalar(driver, `MATCH (f:Feature) RETURN count(f)`)) === 0
  );
  check(
    "no :FeaturesMetadata nodes remain (global)",
    (await scalar(driver, `MATCH (m:FeaturesMetadata) RETURN count(m)`)) === 0
  );
  check(
    ":Concept node exists with same id",
    (await scalar(driver, `MATCH (c:Concept {id:$id}) RETURN count(c)`, { id: CONCEPT_ID })) === 1
  );
  check(
    ":Concept kept :Data_Bank label",
    (await scalar(driver, `MATCH (c:Concept:Data_Bank {id:$id}) RETURN count(c)`, { id: CONCEPT_ID })) === 1
  );
  check(
    ":ConceptsMetadata exists for repo",
    (await scalar(driver, `MATCH (m:ConceptsMetadata {repo:$r}) RETURN count(m)`, { r: REPO })) === 1
  );
  check(
    "Clue.conceptId set, Clue.featureId removed",
    (await scalar(
      driver,
      `MATCH (c:Clue {id:$id}) WHERE c.conceptId = $cid AND c.featureId IS NULL RETURN count(c)`,
      { id: CLUE_ID, cid: CONCEPT_ID }
    )) === 1
  );
  check(
    "Clue.relatedConcepts set, Clue.relatedFeatures removed",
    (await scalar(
      driver,
      `MATCH (c:Clue {id:$id}) WHERE $cid IN c.relatedConcepts AND c.relatedFeatures IS NULL RETURN count(c)`,
      { id: CLUE_ID, cid: CONCEPT_ID }
    )) === 1
  );

  console.log("\n5) Assert relationships survived relabel:");
  check(
    "PullRequest -[:TOUCHES]-> :Concept",
    (await scalar(driver, `MATCH (:PullRequest)-[:TOUCHES]->(:Concept {id:$id}) RETURN count(*)`, { id: CONCEPT_ID })) === 1
  );
  check(
    "Clue -[:RELEVANT_TO]-> :Concept",
    (await scalar(driver, `MATCH (:Clue {id:$cid})-[:RELEVANT_TO]->(:Concept {id:$id}) RETURN count(*)`, { cid: CLUE_ID, id: CONCEPT_ID })) === 1
  );
  check(
    ":Concept -[:MODIFIES]-> :File",
    (await scalar(driver, `MATCH (:Concept {id:$id})-[:MODIFIES]->(:File {ref_id:$f}) RETURN count(*)`, { id: CONCEPT_ID, f: FILE_REF })) === 1
  );

  console.log("\n6) Assert indexes migrated:");
  const idx = await indexNames(driver);
  for (const old of OLD_INDEXES) {
    check(`old index dropped: ${old}`, !idx.includes(old));
  }
  for (const neo of ["concept_id_index", "concept_repo_index", "clue_concept_index"]) {
    check(`new index present: ${neo}`, idx.includes(neo));
  }

  console.log("\n7) Exercise renamed GraphStorage query methods:");
  const all = await storage.getAllConcepts(REPO);
  check("getAllConcepts() returns seeded concept", all.some((c) => c.id === CONCEPT_ID), `got ${all.length}`);

  const one = await storage.getConcept(CONCEPT_ID);
  check("getConcept() returns concept w/ name", one?.name === "Authentication System");

  const clues = await storage.getCluesForConcept(CONCEPT_ID);
  check("getCluesForConcept() returns clue", clues.some((c) => c.id === CLUE_ID), `got ${clues.length}`);

  const files = await storage.getFilesForConcept(CONCEPT_ID);
  check("getFilesForConcept() returns modified file", files.some((f) => f.ref_id === FILE_REF), `got ${files.length}`);

  const prov = await storage.getProvenanceForConcepts([CONCEPT_ID]);
  const provFiles = prov[0]?.files ?? [];
  const entityNames = provFiles.flatMap((f: any) => f.entities.map((e: any) => e.name));
  check("getProvenanceForConcepts() returns concept", prov[0]?.conceptId === CONCEPT_ID);
  check("provenance includes matched entity (verifyToken)", entityNames.includes("verifyToken"), `entities: ${entityNames.join(",")}`);

  const lastPR = await storage.getLastProcessedPR(REPO);
  check("getLastProcessedPR() reads migrated :ConceptsMetadata", lastPR === 1, `got ${lastPR}`);

  console.log("\n8) Idempotency — run initialize() again (should be a no-op migration):");
  await storage.initialize();
  check("still 0 :Feature after 2nd init", (await scalar(driver, `MATCH (f:Feature) RETURN count(f)`)) === 0);
  check("still 1 :Concept after 2nd init", (await scalar(driver, `MATCH (c:Concept {id:$id}) RETURN count(c)`, { id: CONCEPT_ID })) === 1);

  console.log("\n9) Sample queries you can run yourself in cypher-shell:");
  console.log(`   MATCH (c:Concept {repo:"${REPO}"}) RETURN c.id, c.name, c.description;`);
  console.log(`   MATCH (cl:Clue {repo:"${REPO}"}) RETURN cl.id, cl.conceptId, cl.relatedConcepts;`);
  console.log(`   MATCH (c:Concept)-[m:MODIFIES]->(f:File) WHERE c.repo="${REPO}" RETURN c.name, f.file, m.importance;`);

  // Cleanup
  if (process.env.KEEP === "1") {
    console.log(`\n(KEEP=1) Leaving seeded data in DB under repo "${REPO}".`);
  } else {
    console.log("\n10) Cleanup test data...");
    await cleanup(driver);
  }

  await storage.close().catch(() => {});
  await driver.close().catch(() => {});

  console.log(`\n=== Result: ${passed} passed, ${failed} failed ===\n`);
  process.exit(failed === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error("Test harness error:", e);
  process.exit(1);
});
