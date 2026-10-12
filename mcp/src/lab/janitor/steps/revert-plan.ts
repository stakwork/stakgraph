import { z, defineStep, openGraphBackend, withAccessedNodes, type StepContext, type StrutCapabilities } from "strut";

/**
 * janitor/revert-plan — the inverse of an apply run's `applied.json`, as a
 * plan the same apply branch writes: for every change that run applied
 * (or the ones `changes` names), an edit that puts the replaced value back
 * (a value that was absent is deleted again) or a move back under the
 * replaced parent. Each is planned only when the node STILL holds what the
 * janitor wrote — a node someone edited or moved since is skipped
 * (`moved_on`), never overwritten. Writes `plan.json` into this run's
 * artifacts. The revert run's own readback writes its own `applied.json`,
 * so a revert can itself be reverted.
 *
 * `graphCtx` is strut's `graph/*` preamble, inlined (seeded steps import
 * only "strut").
 */
async function graphCtx(ctx: StepContext<StrutCapabilities>) {
  const secrets = ctx.services?.secrets;
  const uri = (await secrets?.get("NEO4J_URI")) || `bolt://${(await secrets?.get("NEO4J_HOST")) || "localhost:7687"}`;
  const emb = ((await secrets?.get("STRUT_GRAPH_EMBEDDINGS")) ?? "").toLowerCase();
  const ont = ((await secrets?.get("STRUT_GRAPH_SEED_ONTOLOGY")) ?? "").toLowerCase();
  return openGraphBackend(
    {
      uri,
      user: (await secrets?.get("NEO4J_USER")) || "neo4j",
      password: (await secrets?.get("NEO4J_PASSWORD")) || "testtest",
      namespace: (await secrets?.get("STRUT_GRAPH_NAMESPACE")) || "default",
      database: (await secrets?.get("NEO4J_DATABASE")) || undefined,
    },
    { embeddings: !["off", "0", "false"].includes(emb), seedOntology: ["1", "true", "on"].includes(ont) },
  );
}

export interface LiveState {
  ref_id: string;
  name: string;
  node_type?: string;
  properties: Record<string, unknown>;
  parents: string[];
}

export async function readState(b: { bolt: { run(q: string, p?: Record<string, unknown>): Promise<Array<Record<string, unknown>>> } }, ref_ids: string[]): Promise<Map<string, LiveState>> {
  const ids = [...new Set(ref_ids.filter((r) => typeof r === "string" && r.length > 0))];
  const out = new Map<string, LiveState>();
  if (ids.length === 0) return out;
  const rows = await b.bolt.run(
    `UNWIND $ids AS id
     MATCH (n:Data_Bank {ref_id: id})
     WHERE n.deleted_at IS NULL AND NOT coalesce(n.is_deleted, false)
     OPTIONAL MATCH (p:Data_Bank)-[:PARENT_OF]->(n)
     RETURN id, labels(n) AS labels, properties(n) AS props, collect(DISTINCT p.ref_id) AS parents`,
    { ids },
  );
  for (const r of rows) {
    const props = (r["props"] ?? {}) as Record<string, unknown>;
    out.set(String(r["id"]), {
      ref_id: String(r["id"]),
      name: typeof props["name"] === "string" ? (props["name"] as string) : "",
      node_type: ((r["labels"] ?? []) as unknown[]).map(String).find((l) => l !== "Data_Bank"),
      properties: props,
      parents: ((r["parents"] ?? []) as unknown[]).filter((p) => typeof p === "string") as string[],
    });
  }
  return out;
}

/** An applied change as janitor/readback logged it. */
export interface Applied {
  id: string;
  kind: "edit" | "move";
  ref_id: string;
  name: string;
  finding?: { issue: string; severity: string; problem: string };
  property?: string;
  old_value?: string | null;
  new_value?: string | null;
  from_ref_id?: string;
  to_ref_id?: string;
  [k: string]: unknown;
}

export interface Change {
  id: string;
  kind: "edit" | "move";
  ref_id: string;
  name: string;
  finding?: { issue: string; severity: string; problem: string };
  /** The change this one undoes. */
  reverts: string;
  property?: string;
  old_value?: string | null;
  new_value?: string | null;
  node_data: Record<string, string>;
  properties_to_be_deleted: string[];
  edge_type?: string;
  direction?: "forward" | "reverse";
  from_ref_id?: string;
  to_ref_id?: string;
}

export interface Skipped {
  id: string;
  ref_id: string;
  name: string;
  reason: string;
  detail?: string;
}

function valueOf(node: LiveState, property: string): string | null {
  const v = node.properties[property];
  if (v === null || v === undefined) return null;
  return typeof v === "string" ? v : String(v);
}

/** Pure: the inverse of each applied change, when the node still holds what the janitor wrote. */
export function revertPlan(applied: Applied[], live: Map<string, LiveState>): { edits: Change[]; moves: Change[]; skipped: Skipped[] } {
  const out = { edits: [] as Change[], moves: [] as Change[], skipped: [] as Skipped[] };
  for (const c of applied) {
    const skip = (reason: string, detail?: string) => out.skipped.push({ id: c.id, ref_id: c.ref_id, name: c.name, reason, ...(detail ? { detail } : {}) });
    const node = live.get(c.ref_id);
    if (!node) {
      skip("not_found", "no such node in the graph now (or deleted)");
      continue;
    }
    if (c.kind === "edit") {
      if (!c.property) {
        skip("invalid", "an edit without a property");
        continue;
      }
      const now = valueOf(node, c.property);
      if (now !== (c.new_value ?? null)) {
        skip("moved_on", `${c.property} no longer reads what the janitor wrote — someone changed it since`);
        continue;
      }
      const back = c.old_value ?? null;
      out.edits.push({
        id: c.id,
        kind: "edit",
        ref_id: c.ref_id,
        name: node.name || c.name,
        ...(c.finding ? { finding: c.finding } : {}),
        reverts: c.id,
        property: c.property,
        old_value: now,
        new_value: back,
        node_data: back === null ? {} : { [c.property]: back },
        properties_to_be_deleted: back === null ? [c.property] : [],
      });
    } else if (c.kind === "move") {
      if (!c.from_ref_id || !c.to_ref_id) {
        skip("invalid", "a move without both parents");
        continue;
      }
      if (!node.parents.includes(c.to_ref_id)) {
        skip("moved_on", "no longer hangs where the janitor put it — someone moved it since");
        continue;
      }
      out.moves.push({
        id: c.id,
        kind: "move",
        ref_id: c.ref_id,
        name: node.name || c.name,
        ...(c.finding ? { finding: c.finding } : {}),
        reverts: c.id,
        edge_type: "PARENT_OF",
        direction: "reverse",
        from_ref_id: c.to_ref_id,
        to_ref_id: c.from_ref_id,
        node_data: {},
        properties_to_be_deleted: [],
      });
    } else {
      skip("invalid", `unknown change kind "${String((c as { kind?: unknown }).kind)}"`);
    }
  }
  return out;
}

/** `changes` as the launch gave it: a list of ids, or one comma-separated string; null = every change. */
export function changeIds(v: unknown): Set<string> | null {
  if (v === null || v === undefined) return null;
  const list = Array.isArray(v) ? v : typeof v === "string" ? v.split(",") : [];
  const ids = list.map((x) => String(x).trim()).filter(Boolean);
  return ids.length ? new Set(ids) : null;
}

const EXAMPLE = `- id: plan
  type: janitor/revert-plan
  config:
    run_id: "{{ input.run_id }}"
    changes: "{{ input.changes }}"`;

export default defineStep({
  type: "janitor/revert-plan",
  description:
    "Plan the undo of a graph-janitor run: reads that run's `applied.json` (its undo log — every change it applied with " +
    "the value or parent it replaced) and emits the inverse as a plan in the same shape janitor/plan writes (`edits` / `moves` " +
    "for the foreach bodies over graph/edit-node and graph/move-node, `skipped` with a reason). A change is reverted only " +
    "when the node STILL holds what the janitor wrote; one someone changed since is skipped (`moved_on`), never " +
    "overwritten. `changes` picks some of the run's changes by id (e1, m2, …). Fails `no_applied_log:` when the run has no " +
    "log (not an apply run, or it applied nothing). Writes `plan.json` (`out`) into this run's artifacts. " +
    "Output: { edits, moves, skipped, revert_of, source, counts }.\n\n" +
    EXAMPLE,
  input: z.object({
    run_id: z.string().min(1).describe("the graph-janitor run whose applied changes to undo"),
    changes: z.any().optional().describe("ids of the changes to undo (e1, m2, …), a list or comma-separated; omit for all"),
    file: z.string().default("applied.json").describe("the log's name in that run's artifacts"),
    out: z.string().default("plan.json").describe("where in this run's artifacts the plan is written"),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const c = ctx as StepContext<StrutCapabilities>;
    const art = c.services?.artifacts;
    if (!art) throw new Error("janitor/revert-plan needs the artifacts capability (where the apply run's log lives)");
    let text: string;
    try {
      text = Buffer.from(await art.read(cfg.run_id, cfg.file)).toString();
    } catch {
      throw new Error(`no_applied_log: run ${cfg.run_id} has no ${cfg.file} — not an apply run of graph-janitor, or it applied nothing`);
    }
    const log = JSON.parse(text) as { workflow?: string; written_at?: string; revert_of?: string; applied?: Applied[] };
    const all = Array.isArray(log.applied) ? log.applied : [];
    const ids = changeIds(cfg.changes);
    const wanted = ids ? all.filter((x) => ids.has(x.id)) : all;
    const unknown = ids ? [...ids].filter((id) => !all.some((x) => x.id === id)) : [];
    const live = wanted.length ? await readState(await graphCtx(c), wanted.map((x) => x.ref_id)) : new Map<string, LiveState>();
    const plan = revertPlan(wanted, live);
    for (const id of unknown) plan.skipped.push({ id, ref_id: "", name: "", reason: "unknown_change", detail: `run ${cfg.run_id} applied no change "${id}"` });
    const out = {
      ...plan,
      revert_of: cfg.run_id,
      source: { run_id: cfg.run_id, workflow: log.workflow ?? null, written_at: log.written_at ?? null, ...(log.revert_of ? { revert_of: log.revert_of } : {}), applied: all.length },
      counts: { requested: wanted.length, edits: plan.edits.length, moves: plan.moves.length, skipped: plan.skipped.length },
    };
    await art.write(ctx.runId, cfg.out, JSON.stringify(out, null, 2));
    return withAccessedNodes(
      out,
      [...live.values()].map((n) => ({ ref_id: n.ref_id, node_type: n.node_type, name: n.name })),
    );
  },
});
