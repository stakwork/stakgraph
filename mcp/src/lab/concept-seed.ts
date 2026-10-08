import { readFile, readdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join } from "node:path";
import yaml from "js-yaml";
import { composeNodeKey, type GraphBackend, type WorkspaceStore } from "strut";
import { jarvisMutate } from "../repo/toolsJarvis.js";

/**
 * Concept files → `Concept` nodes. Graph DATA the lab ships in git.
 *
 * A SET is one directory of `<Name>.md` files and the source prefix its
 * nodes are stamped with (`builder/concepts`, `janitor/concepts`). A file is
 * one Concept: front matter `description` (one line) and optional `parent`
 * (another Concept's name — a file of any set in the same call, or a node
 * already in the graph), body = `docs`.
 *
 * Reconciled like the workflow seeders (SEED_OPTS), on the node instead of a
 * version: every seeded Concept carries `unique_source_id =
 * <prefix><file>@<sha256[0..12]>` of the file it came from.
 *   - no node under that name → create (the writer's create mode);
 *   - the same stamp → keep: nothing written, so an edit made in the graph
 *     (docs rewritten in Learn, `is_muted` set) stays until the committed
 *     file changes; a deleted node stays deleted;
 *   - our path with an older hash, node not deleted → upsert description +
 *     docs (a changed template wins, as for workflows; `+=` merge, so
 *     `is_muted` and every other attribute survive);
 *   - no stamp, or someone else's → skip, never touched: a Concept a person
 *     made under that name, or took over by clearing the stamp, is theirs.
 * A file that moves between sets changes its prefix and reads as someone
 * else's: keep files where they are.
 *
 * The seeded graph depends on the committed files ONLY, never on what was
 * seeded before — two graphs seeded at different times end up the same.
 * That holds for the edges as for the nodes, over the Concepts the seed OWNS
 * (create / update / keep):
 *   - `PARENT_OF`: each owned Concept's `parent` is merged after the nodes
 *     (idempotent by edge key), so a parent may come from another set. A
 *     `PARENT_OF` between two owned Concepts that no file declares is
 *     REMOVED — a parent a file used to name, or one added by hand. An edge
 *     with anyone else's Concept at either end is never touched.
 *   - the anchor: EVERY owned Concept is anchored the way hive anchors every
 *     Concept it creates (jarvis migration 111), `HiveWorkspace -PROCESS->
 *     <Concept>`, to the one workspace node hive's mirror writes into this
 *     graph. Whether a Concept has a parent does not matter, so gaining one
 *     changes nothing. No workspace node (a standalone strut), or more than
 *     one (never guess): no edge, one log line.
 * To own a seeded Concept's edges, take the Concept over: clear its stamp.
 *
 * RETIRED files. Deleting a file from git removes nothing from a graph it
 * was seeded into, so a set lists the files it USED to ship (`retired`): a
 * node under that name that still carries that file's stamp is deleted
 * through jarvis (`retireConcept`): `deleted_at` set, hidden from every
 * read, its edges removed. One whose stamp was cleared or replaced is
 * someone's: left alone. A node counts as deleted when `deleted_at` is set
 * or `is_deleted` is true (either flag, until jarvis stops writing
 * `is_deleted`).
 *
 * What ships here is seeded into EVERY workspace. A Concept that only makes
 * sense for one kind of workspace (a mandate about evals, a legal topic) is
 * that workspace's to add, never a file here.
 *
 * Skipped, with one log line, on a filesystem workspace: no graph.
 */
const TYPE = "Concept";
const WORKSPACE_TYPE = "HiveWorkspace";
const ANCHOR_EDGE = "PROCESS";

export interface ConceptSet {
  dir: string;
  /** What `unique_source_id` starts with, e.g. `lab/janitor/concepts/`. */
  prefix: string;
  /** Files this set used to ship (`<Name>.md`): their nodes are retired. */
  retired?: string[];
}

export interface ConceptFile {
  /** The file's basename without `.md` — the Concept's `name` (its identity). */
  name: string;
  description?: string;
  parent?: string;
  docs?: string;
  /** What `unique_source_id` carries: `<prefix><file>@<hash>`. */
  stamp: string;
}

export function stampOf(file: string, text: string, prefix: string): string {
  return `${prefix}${file}@${createHash("sha256").update(text).digest("hex").slice(0, 12)}`;
}

/** Pure: one `<Name>.md` (optional YAML front matter, body = docs) → its Concept. */
export function parseConceptFile(file: string, text: string, prefix: string): ConceptFile {
  const name = file.replace(/\.md$/i, "");
  let body = text;
  let meta: Record<string, unknown> = {};
  const m = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/.exec(text);
  if (m) {
    body = text.slice(m[0].length);
    const loaded = yaml.load(m[1]);
    if (loaded && typeof loaded === "object" && !Array.isArray(loaded)) meta = loaded as Record<string, unknown>;
  }
  const str = (v: unknown): string | undefined => (typeof v === "string" && v.trim() ? v.trim() : undefined);
  const description = str(meta.description);
  const parent = str(meta.parent);
  const docs = body.trim() || undefined;
  return {
    name,
    ...(description ? { description } : {}),
    ...(parent ? { parent } : {}),
    ...(docs ? { docs } : {}),
    stamp: stampOf(file, text, prefix),
  };
}

/** create / update write the node; keep is ours and unchanged; skip is not ours to touch. */
export type SeedAction = "create" | "update" | "keep" | "skip";

export interface ExistingConcept {
  ref_id: string;
  unique_source_id?: unknown;
  /** `deleted_at` set or `is_deleted` true. */
  deleted?: boolean;
}

/** Pure: the reconcile rule in the module comment. */
export function planConceptSeed(existing: ExistingConcept | null, stamp: string): SeedAction {
  if (!existing) return "create";
  if (existing.deleted) return "skip";
  const usid = existing.unique_source_id;
  if (usid === stamp) return "keep";
  const path = stamp.slice(0, stamp.lastIndexOf("@") + 1);
  if (typeof usid === "string" && usid.startsWith(path)) return "update";
  return "skip";
}

export async function seedConcepts(workspace: WorkspaceStore, sets: ConceptSet[]): Promise<void> {
  const graph = workspace.graph;
  if (!graph) {
    console.log("[concept-seed] filesystem workspace: Concept files not seeded (no graph)");
    return;
  }
  const refs = new Map<string, string>(); // every node under a seeded name, for parenting
  const owned = new Set<string>(); // the ones the seed wrote or keeps (create / update / keep)
  const parsed: ConceptFile[] = [];
  for (const { dir, prefix } of sets) {
    let files: string[];
    try {
      files = (await readdir(dir)).filter((f) => f.endsWith(".md")).sort();
    } catch (err) {
      console.warn(`[concept-seed] could not read ${dir}:`, err instanceof Error ? err.message : err);
      continue;
    }
    for (const file of files) {
      try {
        const c = parseConceptFile(file, await readFile(join(dir, file), "utf-8"), prefix);
        parsed.push(c);
        const existing = await findConcept(graph, c.name);
        const action = planConceptSeed(existing, c.stamp);
        if (action === "keep" || action === "skip") {
          refs.set(c.name, existing!.ref_id);
          if (action === "keep") owned.add(c.name);
          continue;
        }
        const data = {
          name: c.name,
          ...(c.description ? { description: c.description } : {}),
          ...(c.docs ? { docs: c.docs } : {}),
          unique_source_id: c.stamp,
        };
        const r = await graph.nodes.write({ type: TYPE, data }, action === "create" ? "create" : "upsert", {
          namespace: graph.cfg.namespace,
        });
        refs.set(c.name, r.ref_id);
        owned.add(c.name);
        console.log(`[concept-seed] ${action === "create" ? "seeded" : "updated"} Concept: ${c.name} (${r.outcome})`);
      } catch (err) {
        console.warn(`[concept-seed] could not seed Concept from "${file}":`, err instanceof Error ? err.message : err);
      }
    }
  }
  for (const { prefix, retired } of sets) {
    for (const file of retired ?? []) {
      const name = file.replace(/\.md$/i, "");
      try {
        const existing = await findConcept(graph, name);
        const usid = existing?.unique_source_id;
        if (!existing || existing.deleted || typeof usid !== "string" || !usid.startsWith(`${prefix}${file}@`)) continue;
        await retireConcept(graph, existing.ref_id);
        console.log(`[concept-seed] retired Concept: ${name}`);
      } catch (err) {
        console.warn(`[concept-seed] could not retire Concept "${name}":`, err instanceof Error ? err.message : err);
      }
    }
  }
  const declared = new Set<string>(); // `<parent ref_id>><child ref_id>`, as the files have it
  for (const c of parsed) {
    if (!c.parent || !owned.has(c.name)) continue;
    try {
      const child = refs.get(c.name);
      const parent = refs.get(c.parent) ?? (await findConcept(graph, c.parent))?.ref_id;
      if (!child || !parent) {
        console.warn(`[concept-seed] no Concept "${c.parent}" to parent "${c.name}" under`);
        continue;
      }
      declared.add(`${parent}>${child}`); // before the write: a failed write must not read as "undeclared"
      const e = await graph.edges.write({ edge: "PARENT_OF", source_ref_id: parent, target_ref_id: child });
      if (e.created) console.log(`[concept-seed] linked ${c.parent} -PARENT_OF-> ${c.name}`);
    } catch (err) {
      console.warn(`[concept-seed] could not parent "${c.name}" under "${c.parent}":`, err instanceof Error ? err.message : err);
    }
  }
  const ours = parsed.filter((c) => owned.has(c.name));
  if (!ours.length) return;
  // Between two Concepts we own, the files are the whole truth.
  try {
    const ids = ours.map((c) => refs.get(c.name)!);
    const rows = await graph.bolt.run(
      `MATCH (p:Data_Bank)-[e:PARENT_OF]->(c:Data_Bank)
       WHERE p.ref_id IN $ids AND c.ref_id IN $ids
       RETURN p.ref_id AS parent, c.ref_id AS child, p.name AS parent_name, c.name AS child_name`,
      { ids },
    );
    for (const r of rows) {
      if (declared.has(`${r.parent}>${r.child}`)) continue;
      await graph.bolt.run(`MATCH (p:Data_Bank {ref_id: $parent})-[e:PARENT_OF]->(c:Data_Bank {ref_id: $child}) DELETE e`, {
        parent: r.parent,
        child: r.child,
      });
      console.log(`[concept-seed] unlinked ${r.parent_name} -PARENT_OF-> ${r.child_name} (no file declares it)`);
    }
  } catch (err) {
    console.warn("[concept-seed] could not reconcile PARENT_OF edges:", err instanceof Error ? err.message : err);
  }
  let ws: { ref_id: string; label: string } | null;
  try {
    ws = await findWorkspace(graph);
  } catch (err) {
    console.warn(`[concept-seed] could not look up the ${WORKSPACE_TYPE} node:`, err instanceof Error ? err.message : err);
    return;
  }
  if (!ws) return;
  for (const c of ours) {
    try {
      const e = await graph.edges.write({ edge: ANCHOR_EDGE, source_ref_id: ws.ref_id, target_ref_id: refs.get(c.name)! });
      if (e.created) console.log(`[concept-seed] anchored ${ws.label} -${ANCHOR_EDGE}-> ${c.name}`);
    } catch (err) {
      console.warn(`[concept-seed] could not anchor "${c.name}" to ${ws.label}:`, err instanceof Error ? err.message : err);
    }
  }
}

/**
 * Delete one retired Concept the way every other node delete goes: jarvis
 * `DELETE /v2/nodes/<ref_id>/single` sets `deleted_at` (and `is_deleted`)
 * and removes the node's edges. Already deleted (409) is done.
 *
 * Without JARVIS_URL (a standalone strut: its own Neo4j, no jarvis process)
 * there is nothing to call, so the same delete runs here in one statement:
 * the first `deleted_at` kept, and only edges whose ends are both
 * `:Data_Bank` in this namespace removed.
 */
export async function retireConcept(graph: GraphBackend, refId: string): Promise<void> {
  const ns = graph.cfg.namespace;
  const jarvisUrl = process.env.JARVIS_URL?.replace(/\/+$/, "");
  if (!jarvisUrl) {
    await graph.bolt.run(
      `MATCH (n:Data_Bank {ref_id: $ref_id, namespace: $ns})
       WHERE n.deleted_at IS NULL AND NOT coalesce(n.is_deleted, false)
       SET n.deleted_at = coalesce(n.deleted_at, $now), n.is_deleted = true
       WITH n
       OPTIONAL MATCH (n)-[r]-(m:Data_Bank {namespace: $ns})
       DELETE r`,
      { ref_id: refId, ns, now: Date.now() },
    );
    return;
  }
  // Seeded Concepts carry no owner, so jarvis lets only an admin delete them.
  const headers = { "X-Api-Token": process.env.API_TOKEN ?? "", "X-Is-Admin": "true" };
  const url = `${jarvisUrl}/v2/nodes/${encodeURIComponent(refId)}/single?${new URLSearchParams({ namespace: ns })}`;
  const res = await jarvisMutate("delete", url, headers);
  if (res.ok || (res.status === 409 && res.text.includes("ALREADY_DELETED"))) return;
  throw new Error(`jarvis delete failed — HTTP ${res.status}: ${res.text}`);
}

/** The node under this name in the backend's namespace, deleted or not. */
async function findConcept(graph: GraphBackend, name: string): Promise<ExistingConcept | null> {
  const schema = await graph.nodes.resolver.schema(TYPE);
  if (!schema) throw new Error(`no "${TYPE}" schema in the graph (seed the ontology: STRUT_GRAPH_SEED_ONTOLOGY=1)`);
  const key = composeNodeKey(schema, { name });
  const rows = await graph.bolt.run(
    `MATCH (n:\`${schema.type}\` {node_key: $key, namespace: $ns})
     RETURN n.ref_id AS ref_id, n.unique_source_id AS unique_source_id,
            (n.deleted_at IS NOT NULL OR coalesce(n.is_deleted, false)) AS deleted
     LIMIT 1`,
    { key, ns: graph.cfg.namespace },
  );
  const r = rows[0];
  if (!r) return null;
  return { ref_id: String(r.ref_id), unique_source_id: r.unique_source_id, deleted: r.deleted === true };
}

/** The ONE workspace node hive mirrors into this graph; null (with a log line) when there is none or more than one. */
async function findWorkspace(graph: GraphBackend): Promise<{ ref_id: string; label: string } | null> {
  const schema = await graph.nodes.resolver.schema(WORKSPACE_TYPE);
  if (!schema) {
    console.log(`[concept-seed] no ${WORKSPACE_TYPE} schema in the graph: Concepts not anchored`);
    return null;
  }
  const rows = await graph.bolt.run(
    `MATCH (w:\`${schema.type}\`)
     WHERE coalesce(w.namespace, 'default') = $ns AND w.deleted_at IS NULL AND NOT coalesce(w.is_deleted, false)
     RETURN w.ref_id AS ref_id, coalesce(w.slug, w.name, w.ref_id) AS label
     LIMIT 2`,
    { ns: graph.cfg.namespace },
  );
  if (rows.length === 0) {
    console.log(`[concept-seed] no ${WORKSPACE_TYPE} node in the graph (standalone strut?): Concepts not anchored`);
    return null;
  }
  if (rows.length > 1) {
    console.warn(`[concept-seed] more than one ${WORKSPACE_TYPE} node in namespace "${graph.cfg.namespace}": Concepts not anchored`);
    return null;
  }
  return { ref_id: String(rows[0]!.ref_id), label: String(rows[0]!.label) };
}
