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
 *   - the same stamp → nothing, so an edit made in the graph (docs rewritten
 *     in Learn, `is_muted` set, the node deleted) stays until the committed
 *     file changes;
 *   - our path with an older hash, node not deleted → upsert description +
 *     docs (a changed template wins, as for workflows; `+=` merge, so
 *     `is_muted` and every other attribute survive);
 *   - no stamp, or someone else's → never touched: a Concept a person made
 *     under that name, or took over by clearing the stamp, is theirs.
 * `PARENT_OF` edges are merged after the nodes (idempotent by edge key).
 * Skipped, with one log line, on a filesystem workspace: no graph, and the
 * janitor engine cannot run there either.
 */
const HERE = dirname(fileURLToPath(import.meta.url));
export const CONCEPTS_DIR = join(HERE, "concepts");
const SOURCE_PREFIX = "lab/janitor/concepts/";
const TYPE = "Concept";

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

export type SeedAction = "create" | "update" | "skip";

export interface ExistingConcept {
  ref_id: string;
  unique_source_id?: unknown;
  is_deleted?: boolean;
}

/** Pure: the reconcile rule in the module comment. */
export function planConceptSeed(existing: ExistingConcept | null, stamp: string): SeedAction {
  if (!existing) return "create";
  const usid = existing.unique_source_id;
  if (usid === stamp) return "skip";
  const path = stamp.slice(0, stamp.lastIndexOf("@") + 1);
  if (typeof usid === "string" && usid.startsWith(path) && !existing.is_deleted) return "update";
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
  const refs = new Map<string, string>();
  const parsed: ConceptFile[] = [];
  for (const file of files) {
    try {
      const c = parseConceptFile(file, await readFile(join(dir, file), "utf-8"));
      parsed.push(c);
      const existing = await findConcept(graph, c.name);
      const action = planConceptSeed(existing, c.stamp);
      if (action === "skip") {
        refs.set(c.name, existing!.ref_id);
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
