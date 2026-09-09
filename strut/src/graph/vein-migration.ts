/**
 * One-shot rename of the pre-rename graph domain: `Vein*` → `Strut*`.
 *
 * Before the package was renamed (stakgraph #1664) strut wrote its nodes
 * under `VeinWorkflow`, `VeinStep`, … with `Domain_vein` and node_keys like
 * `veinstep-<type>`. Nothing about those rows changed except the names, so
 * this pass rewrites the names in place and leaves everything else —
 * `ref_id`s, edges, `content_hash`es, embeddings, `Data_Bank` — untouched:
 *
 *   1. guard: refuse (throw) if any rewritten `(node_key, namespace)` is
 *      already held by another node — a database that was booted on the
 *      renamed code and re-published into needs a human, not a merge policy;
 *   2. `:Schema` meta-nodes: `type`, `domain`, `node_key` spec renamed in
 *      place (edge-schema relationships hang off the same nodes). A legacy
 *      Schema whose Strut twin already exists (the seed ran on the renamed
 *      code before this migration existed) is dropped instead — two Schema
 *      nodes for one type would break jarvis;
 *   3. data nodes, per type, in batches: add the `Strut*` + `Domain_strut`
 *      labels, rewrite the node_key prefix, remove the legacy labels;
 *   4. drop every constraint and index on a legacy label — the seed that
 *      runs next recreates them under the new names (jarvis rebuilds its
 *      own `domain_<x>_attribute_index` from `Schema.domain` by itself);
 *   5. stamp the `Migration` ledger so this never scans again.
 *
 * Runs at boot before `seedStrutDomain` (backend.ts). Idempotent at every
 * step, so a crash mid-way is repaired by the next boot. Step sources that
 * `import "vein"` are deliberately NOT rewritten (the content hash is the
 * version's identity); strut-resolve-hook.ts keeps `"vein"` as an alias.
 */
import { Bolt } from "./bolt.js";
import { schemaStatement } from "./schema-seed.js";
import { STRUT_DOMAIN, STRUT_DOMAIN_LABEL, STRUT_SCHEMAS } from "./strut-schemas.js";

export const VEIN_MIGRATION_ID = "strut_rename_vein_v1";
export const LEGACY_DOMAIN = "Vein";
export const LEGACY_DOMAIN_LABEL = "Domain_vein";
const NEW_PREFIX = "Strut";
const LEGACY_PREFIX = "Vein";
const BATCH = 1000;

export interface LegacyType {
  /** Current label, e.g. `StrutRun`. */
  type: string;
  /** Pre-rename label, e.g. `VeinRun`. */
  legacy: string;
  /** node_key spec from the library, e.g. `strutrun-run_id`. */
  nodeKeySpec: string;
  /** node_key value prefixes: `strutrun-` / `veinrun-`. */
  keyPrefix: string;
  legacyKeyPrefix: string;
}

/** The nine renames, derived from the live library so they cannot drift. */
export function legacyTypes(): LegacyType[] {
  return STRUT_SCHEMAS.map((s) => {
    if (!s.type.startsWith(NEW_PREFIX)) throw new Error(`legacyTypes: ${s.type} is not a ${NEW_PREFIX}* type`);
    const legacy = LEGACY_PREFIX + s.type.slice(NEW_PREFIX.length);
    return {
      type: s.type,
      legacy,
      nodeKeySpec: s.node_key,
      keyPrefix: `${s.type.toLowerCase()}-`,
      legacyKeyPrefix: `${legacy.toLowerCase()}-`,
    };
  });
}

export interface VeinMigrationReport {
  /** `already_done` = ledger row present, nothing scanned; `nothing_to_do` =
   *  scanned, found no legacy names, ledger stamped. */
  status: "migrated" | "already_done" | "nothing_to_do";
  /** Legacy Schema nodes renamed in place (new type names). */
  schemasRenamed: string[];
  /** Legacy Schema nodes dropped because a Strut twin already existed. */
  schemasDropped: string[];
  /** Data nodes relabeled, per new type. */
  relabeled: Record<string, number>;
  droppedConstraints: string[];
  droppedIndexes: string[];
  /** `Domain_vein` nodes left alone because they carry no legacy type label. */
  strays: number;
}

export class VeinMigrationCollision extends Error {
  constructor(readonly collisions: Array<{ legacy: string; key: string; labels: string[] }>) {
    const sample = collisions
      .slice(0, 5)
      .map((c) => `${c.legacy} → ${c.key} (held by ${c.labels.join(":")})`)
      .join("; ");
    super(
      `migrateVeinToStrut: ${collisions.length} node_key collision(s) — a Strut node already holds a key a legacy ` +
        `node would be renamed to; resolve by hand before booting: ${sample}`,
    );
    this.name = "VeinMigrationCollision";
  }
}

export async function migrateVeinToStrut(bolt: Bolt): Promise<VeinMigrationReport> {
  const report: VeinMigrationReport = {
    status: "nothing_to_do",
    schemasRenamed: [],
    schemasDropped: [],
    relabeled: {},
    droppedConstraints: [],
    droppedIndexes: [],
    strays: 0,
  };
  const ledger = await bolt.run(`MATCH (m:Migration {migration_id: $id}) RETURN count(m) AS c`, { id: VEIN_MIGRATION_ID });
  if (Number(ledger[0]?.["c"] ?? 0) > 0) {
    report.status = "already_done";
    return report;
  }

  const types = legacyTypes();
  const legacyLabels = [...types.map((t) => t.legacy), LEGACY_DOMAIN_LABEL];

  // 1. Collision guard — read-only, before anything is touched.
  const collisions: VeinMigrationCollision["collisions"] = [];
  for (const t of types) {
    const rows = await bolt.run(
      `MATCH (v:\`${t.legacy}\`) WHERE v.node_key STARTS WITH $old
       WITH v, $new + substring(v.node_key, size($old)) AS k
       MATCH (s:Node {node_key: k, namespace: v.namespace})
       RETURN v.node_key AS legacy, k AS key, labels(s) AS labels LIMIT 25`,
      { old: t.legacyKeyPrefix, new: t.keyPrefix },
    );
    for (const r of rows) collisions.push({ legacy: r["legacy"] as string, key: r["key"] as string, labels: r["labels"] as string[] });
  }
  if (collisions.length > 0) throw new VeinMigrationCollision(collisions);

  let touched = false;

  // 2. Schema meta-nodes.
  for (const t of types) {
    const twin = await bolt.run(`MATCH (s:Schema) WHERE toLower(s.type) = toLower($t) RETURN count(s) AS c`, { t: t.type });
    const hasTwin = Number(twin[0]?.["c"] ?? 0) > 0;
    const rows = hasTwin
      ? await bolt.run(`MATCH (s:Schema {type: $legacy}) DETACH DELETE s RETURN count(s) AS c`, { legacy: t.legacy })
      : await bolt.run(
          `MATCH (s:Schema {type: $legacy})
           SET s.type = $type, s.domain = $domain, s.node_key = $spec
           RETURN count(s) AS c`,
          { legacy: t.legacy, type: t.type, domain: STRUT_DOMAIN, spec: t.nodeKeySpec },
        );
    if (Number(rows[0]?.["c"] ?? 0) > 0) {
      touched = true;
      (hasTwin ? report.schemasDropped : report.schemasRenamed).push(t.type);
    }
  }

  // 3. Data nodes, batched by label so a large run/tool-call history does
  //    not become one giant transaction.
  for (const t of types) {
    let total = 0;
    for (;;) {
      const rows = await bolt.run(
        `MATCH (v:\`${t.legacy}\`) WITH v LIMIT ${BATCH}
         SET v:\`${t.type}\`:\`${STRUT_DOMAIN_LABEL}\`
         SET v.node_key = CASE WHEN v.node_key STARTS WITH $old
                               THEN $new + substring(v.node_key, size($old))
                               ELSE v.node_key END
         REMOVE v:\`${t.legacy}\`:\`${LEGACY_DOMAIN_LABEL}\`
         RETURN count(v) AS c`,
        { old: t.legacyKeyPrefix, new: t.keyPrefix },
      );
      const c = Number(rows[0]?.["c"] ?? 0);
      total += c;
      if (c < BATCH) break;
    }
    if (total > 0) {
      touched = true;
      report.relabeled[t.type] = total;
    }
  }
  const strays = await bolt.run(`MATCH (n:\`${LEGACY_DOMAIN_LABEL}\`) RETURN count(n) AS c`);
  report.strays = Number(strays[0]?.["c"] ?? 0);

  // 4. Schema objects on legacy labels. Constraints first: dropping one
  //    drops its backing index, so the index listing must come after.
  const labelList = legacyLabels.map((l) => `'${l}'`).join(", ");
  const constraints = await bolt.run(
    `SHOW CONSTRAINTS YIELD name, labelsOrTypes
     WHERE any(l IN labelsOrTypes WHERE l IN [${labelList}])
     RETURN name ORDER BY name`,
  );
  for (const r of constraints) {
    await bolt.run(`DROP CONSTRAINT \`${r["name"]}\` IF EXISTS`);
    report.droppedConstraints.push(r["name"] as string);
  }
  const indexes = await bolt.run(
    `SHOW INDEXES YIELD name, type, labelsOrTypes
     WHERE type <> 'LOOKUP' AND any(l IN labelsOrTypes WHERE l IN [${labelList}])
     RETURN name ORDER BY name`,
  );
  for (const r of indexes) {
    await bolt.run(`DROP INDEX \`${r["name"]}\` IF EXISTS`);
    report.droppedIndexes.push(r["name"] as string);
  }
  if (constraints.length + indexes.length > 0) touched = true;

  // 5. Ledger.
  await schemaStatement(
    bolt,
    `CREATE CONSTRAINT migration_id_unique IF NOT EXISTS
     FOR (m:Migration) REQUIRE m.migration_id IS UNIQUE`,
  );
  await bolt.run(`MERGE (m:Migration {migration_id: $id}) ON CREATE SET m.executed_at = timestamp()`, {
    id: VEIN_MIGRATION_ID,
  });

  report.status = touched ? "migrated" : "nothing_to_do";
  return report;
}
