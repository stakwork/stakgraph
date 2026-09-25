import { readFile, readdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import yaml from "js-yaml";
import { composeNodeKey, type GraphBackend, type WorkspaceStore } from "strut";

/**
 * janitor — the `Janitor` Concept tree. Graph DATA, not a workflow.
 *
 * A janitor is an automated agent that cleans something up. Its mandate —
 * what "dirty" means and where to start — lives in the knowledge graph as a
 * `Concept` under the root `Janitor`, one child per janitor, so ONE seeded
 * engine workflow can read its mandates from whatever graph it runs on, and
 * a workspace grows its own janitors by adding Concepts (hive's Learn page,
 * Jamie, strut's graph/* steps). This seeder ships the stock tree from
 * `concepts/<Name>.md`: front matter `description` (one line) and optional
 * `parent` (another Concept's name — a sibling file or a node already in the
 * graph), body = `docs`. The root has no parent.
 *
 * Reconciled like the workflow seeders (SEED_OPTS), on the node instead of a
 * version: every seeded Concept carries `unique_source_id =
 * lab/janitor/concepts/<file>@<sha256[0..12]>` of the file it came from.
 *   - no node under that name → create (the writer's create mode);
 *   - the same stamp → keep: nothing written, so an edit made in the graph
 *     (docs rewritten in Learn, `is_muted` set) stays until the committed
 *     file changes; a deleted node stays deleted;
 *   - our path with an older hash, node not deleted → upsert description +
 *     docs (a changed template wins, as for workflows; `+=` merge, so
 *     `is_muted` and every other attribute survive);
 *   - no stamp, or someone else's → skip, never touched: a Concept a person
 *     made under that name, or took over by clearing the stamp, is theirs.
 * `PARENT_OF` edges are merged after the nodes (idempotent by edge key).
 *
 * A root (a file with no `parent`) is a top-level process Concept of the
 * workspace, so it is ANCHORED the way hive anchors every workspace-level
 * Concept it creates (jarvis migration 111): `HiveWorkspace -PROCESS->
 * <root>`, to the one workspace node hive's mirror writes into this graph.
 * That edge — not the tree's shape — is what marks a root, so a freshly
 * seeded `Janitor` with no children yet is already findable. No workspace
 * node (a standalone strut), or more than one (never guess): no edge, one
 * log line. Only seeded roots we own are anchored (create / update / keep).
 * Skipped, with one log line, on a filesystem workspace: no graph, and the
 * janitor engine cannot run there either.
 */
const HERE = dirname(fileURLToPath(import.meta.url));
export const CONCEPTS_DIR = join(HERE, "concepts");
const SOURCE_PREFIX = "lab/janitor/concepts/";
const TYPE = "Concept";
const WORKSPACE_TYPE = "HiveWorkspace";
const ANCHOR_EDGE = "PROCESS";

export interface ConceptFile {
  /** The file's basename without `.md` — the Concept's `name` (its identity). */
  name: string;
  description?: string;
  parent?: string;
  docs?: string;
  /** What `unique_source_id` carries: `lab/janitor/concepts/<file>@<hash>`. */
  stamp: string;
}

export function stampOf(file: string, text: string): string {
  return `${SOURCE_PREFIX}${file}@${createHash("sha256").update(text).digest("hex").slice(0, 12)}`;
}

/** Pure: one `<Name>.md` (optional YAML front matter, body = docs) → its Concept. */
export function parseConceptFile(file: string, text: string): ConceptFile {
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
    stamp: stampOf(file, text),
  };
}

/** create / update write the node; keep is ours and unchanged; skip is not ours to touch. */
export type SeedAction = "create" | "update" | "keep" | "skip";

export interface ExistingConcept {
  ref_id: string;
  unique_source_id?: unknown;
  is_deleted?: boolean;
}

/** Pure: the reconcile rule in the module comment. */
export function planConceptSeed(existing: ExistingConcept | null, stamp: string): SeedAction {
  if (!existing) return "create";
  if (existing.is_deleted) return "skip";
  const usid = existing.unique_source_id;
  if (usid === stamp) return "keep";
  const path = stamp.slice(0, stamp.lastIndexOf("@") + 1);
  if (typeof usid === "string" && usid.startsWith(path)) return "update";
  return "skip";
}

export async function seedJanitorConcepts(workspace: WorkspaceStore, opts: { dir?: string } = {}): Promise<void> {
  const graph = workspace.graph;
  if (!graph) {
    console.log("[janitor] filesystem workspace: Janitor Concept tree not seeded (no graph)");
    return;
  }
  const dir = opts.dir ?? CONCEPTS_DIR;
  let files: string[];
  try {
    files = (await readdir(dir)).filter((f) => f.endsWith(".md")).sort();
  } catch (err) {
    console.warn(`[janitor] could not read ${dir}:`, err instanceof Error ? err.message : err);
    return;
  }
  const refs = new Map<string, string>(); // every node under a seeded name, for parenting
  const owned = new Set<string>(); // the ones the seed wrote or keeps (create / update / keep)
  const parsed: ConceptFile[] = [];
  for (const file of files) {
    try {
      const c = parseConceptFile(file, await readFile(join(dir, file), "utf-8"));
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
      console.log(`[janitor] ${action === "create" ? "seeded" : "updated"} Concept: ${c.name} (${r.outcome})`);
    } catch (err) {
      console.warn(`[janitor] could not seed Concept from "${file}":`, err instanceof Error ? err.message : err);
    }
  }
  for (const c of parsed) {
    if (!c.parent) continue;
    try {
      const child = refs.get(c.name);
      const parent = refs.get(c.parent) ?? (await findConcept(graph, c.parent))?.ref_id;
      if (!child || !parent) {
        console.warn(`[janitor] no Concept "${c.parent}" to parent "${c.name}" under`);
        continue;
      }
      const e = await graph.edges.write({ edge: "PARENT_OF", source_ref_id: parent, target_ref_id: child });
      if (e.created) console.log(`[janitor] linked ${c.parent} -PARENT_OF-> ${c.name}`);
    } catch (err) {
      console.warn(`[janitor] could not parent "${c.name}" under "${c.parent}":`, err instanceof Error ? err.message : err);
    }
  }
  const roots = parsed.filter((c) => !c.parent && owned.has(c.name));
  if (!roots.length) return;
  let ws: { ref_id: string; label: string } | null;
  try {
    ws = await findWorkspace(graph);
  } catch (err) {
    console.warn(`[janitor] could not look up the ${WORKSPACE_TYPE} node:`, err instanceof Error ? err.message : err);
    return;
  }
  if (!ws) return;
  for (const c of roots) {
    try {
      const e = await graph.edges.write({ edge: ANCHOR_EDGE, source_ref_id: ws.ref_id, target_ref_id: refs.get(c.name)! });
      if (e.created) console.log(`[janitor] anchored ${ws.label} -${ANCHOR_EDGE}-> ${c.name}`);
    } catch (err) {
      console.warn(`[janitor] could not anchor "${c.name}" to ${ws.label}:`, err instanceof Error ? err.message : err);
    }
  }
}

/** The node under this name in the backend's namespace, soft-deleted or not. */
async function findConcept(graph: GraphBackend, name: string): Promise<ExistingConcept | null> {
  const schema = await graph.nodes.resolver.schema(TYPE);
  if (!schema) throw new Error(`no "${TYPE}" schema in the graph (seed the ontology: STRUT_GRAPH_SEED_ONTOLOGY=1)`);
  const key = composeNodeKey(schema, { name });
  const rows = await graph.bolt.run(
    `MATCH (n:\`${schema.type}\` {node_key: $key, namespace: $ns})
     RETURN n.ref_id AS ref_id, n.unique_source_id AS unique_source_id, coalesce(n.is_deleted, false) AS is_deleted
     LIMIT 1`,
    { key, ns: graph.cfg.namespace },
  );
  const r = rows[0];
  if (!r) return null;
  return { ref_id: String(r.ref_id), unique_source_id: r.unique_source_id, is_deleted: r.is_deleted === true };
}

/** The ONE workspace node hive mirrors into this graph; null (with a log line) when there is none or more than one. */
async function findWorkspace(graph: GraphBackend): Promise<{ ref_id: string; label: string } | null> {
  const schema = await graph.nodes.resolver.schema(WORKSPACE_TYPE);
  if (!schema) {
    console.log(`[janitor] no ${WORKSPACE_TYPE} schema in the graph: roots not anchored`);
    return null;
  }
  const rows = await graph.bolt.run(
    `MATCH (w:\`${schema.type}\`)
     WHERE coalesce(w.namespace, 'default') = $ns AND NOT coalesce(w.is_deleted, false)
     RETURN w.ref_id AS ref_id, coalesce(w.slug, w.name, w.ref_id) AS label
     LIMIT 2`,
    { ns: graph.cfg.namespace },
  );
  if (rows.length === 0) {
    console.log(`[janitor] no ${WORKSPACE_TYPE} node in the graph (standalone strut?): roots not anchored`);
    return null;
  }
  if (rows.length > 1) {
    console.warn(`[janitor] more than one ${WORKSPACE_TYPE} node in namespace "${graph.cfg.namespace}": roots not anchored`);
    return null;
  }
  return { ref_id: String(rows[0]!.ref_id), label: String(rows[0]!.label) };
}
