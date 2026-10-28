import { z, defineStep, withAccessedNodes, type StepContext, type StrutCapabilities, type Neo4jBackend } from "strut";

/**
 * janitor/readback — after the apply branch wrote a plan's changes, says
 * what actually happened, from the graph itself. Graph steps return error
 * strings instead of throwing, so each write's result is read: a string is
 * `failed`; a success is re-read live and is `applied` only when the node
 * now holds the planned value (an edit) or hangs under the planned parent
 * (a move), else `not_verified`. Writes `applied.json` — the run's UNDO
 * LOG: every applied change with the value or parent it replaced, what
 * graph-janitor-revert reads — appends the outcome and the revert
 * instruction to the run's report, and FAILS the step (`apply_failed:`)
 * when anything failed or could not be verified, so the run is only green
 * when every write matches what was read back. Artifacts are kept, so the
 * log outlives the run.
 *
 * `graphCtx` is the deployment's graph as strut wires it onto the services
 * bag (`ctx.services.graph`, what the `graph/*` steps read through) —
 * never opened from env here and never closed: it is the process's one
 * connection (2026-10-28, swarm38: a step that opened the graph and closed
 * it took the whole deployment down with "Pool is closed").
 */
function graphCtx(ctx: StepContext<StrutCapabilities>): Neo4jBackend {
  const g = ctx.services?.graph;
  if (!g) throw new Error("no_graph: no graph on the services bag (ctx.services.graph)");
  if (g.kind !== "neo4j") throw new Error(`no_graph: these reads are Cypher over bolt; this graph is ${g.kind}`);
  return g;
}

/** What a verdict needs of a node now: its properties and its parents. */
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

/** A planned change as janitor/plan (or janitor/revert-plan) emits it. */
export interface Change {
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

export interface Verdicts {
  applied: Array<Change & { result: unknown }>;
  failed: Array<Change & { error: string }>;
  not_verified: Array<Change & { expected: string | null; found: string | null }>;
}

function valueOf(node: LiveState, property: string): string | null {
  const v = node.properties[property];
  if (v === null || v === undefined) return null;
  return typeof v === "string" ? v : String(v);
}

/** Pure: each planned change against its write's result and the node as read back. */
export function verdicts(plan: { edits?: Change[]; moves?: Change[] }, edits: unknown[], moves: unknown[], live: Map<string, LiveState>): Verdicts {
  const v: Verdicts = { applied: [], failed: [], not_verified: [] };
  const errorOf = (r: unknown): string | null => {
    if (r === undefined || r === null) return "no result — the write never ran";
    if (typeof r === "string") return r;
    if (typeof r === "object" && "status" in r) {
      const s = String((r as { status?: unknown }).status ?? "");
      return s === "Success" || s === "Warning" ? null : `status ${s}: ${JSON.stringify(r).slice(0, 300)}`;
    }
    return `unexpected result: ${JSON.stringify(r).slice(0, 300)}`;
  };
  (plan.edits ?? []).forEach((c, i) => {
    const r = edits[i];
    const err = errorOf(r);
    if (err) return void v.failed.push({ ...c, error: err });
    const node = live.get(c.ref_id);
    const expected = c.new_value ?? null;
    const found = node && c.property ? valueOf(node, c.property) : null;
    if (!node) return void v.not_verified.push({ ...c, expected, found: null });
    if (found === expected) v.applied.push({ ...c, result: r });
    else v.not_verified.push({ ...c, expected, found });
  });
  (plan.moves ?? []).forEach((c, i) => {
    const r = moves[i];
    const err = errorOf(r);
    if (err) return void v.failed.push({ ...c, error: err });
    const node = live.get(c.ref_id);
    const expected = c.to_ref_id ?? null;
    const found = node ? node.parents.join(",") || null : null;
    if (node && expected && node.parents.includes(expected) && (!c.from_ref_id || !node.parents.includes(c.from_ref_id))) v.applied.push({ ...c, result: r });
    else v.not_verified.push({ ...c, expected, found });
  });
  return v;
}

const short = (s: string | null | undefined, n = 60) => (s === null || s === undefined ? "∅" : JSON.stringify(s.length > n ? `${s.slice(0, n)}…` : s));

/** The section appended to the run's report. */
export function reportSection(runId: string, v: Verdicts, opts: { revert_workflow: string; revert_of?: string; file: string }): string {
  const lines: string[] = [];
  lines.push(`## ${opts.revert_of ? `Reverted (run ${runId}, undoing ${opts.revert_of})` : `Applied (run ${runId})`}`, "");
  if (v.applied.length === 0) lines.push("Nothing was written.");
  for (const c of v.applied) {
    lines.push(
      c.kind === "edit"
        ? `- ${c.id} · ${c.name || c.ref_id} · ${c.property}: ${short(c.old_value)} → ${short(c.new_value)}`
        : `- ${c.id} · ${c.name || c.ref_id} · moved from ${c.from_ref_id} to ${c.to_ref_id}`,
    );
  }
  if (v.failed.length) {
    lines.push("", "### Failed", "");
    for (const c of v.failed) lines.push(`- ${c.id} · ${c.name || c.ref_id} · ${c.error}`);
  }
  if (v.not_verified.length) {
    lines.push("", "### Not verified (the write reported success, the node reads back differently)", "");
    for (const c of v.not_verified) lines.push(`- ${c.id} · ${c.name || c.ref_id} · expected ${short(c.expected)}, found ${short(c.found)}`);
  }
  if (v.applied.length) {
    lines.push(
      "",
      `Every applied change and what it replaced is in \`/artifacts/${runId}/${opts.file}\`. To undo them, run \`${opts.revert_workflow}\` with input ` +
        `\`{ "run_id": "${runId}" }\` — or \`{ "run_id": "${runId}", "changes": [${v.applied.map((c) => `"${c.id}"`).join(", ")}] }\` to pick some.`,
    );
  }
  return lines.join("\n") + "\n";
}

const EXAMPLE = `- id: readback
  type: janitor/readback
  depends: moves
  config:
    plan: "{{ plan }}"
    edits: "{{ edits }}"          # the foreach over graph/edit-node, one result per plan.edits entry
    moves: "{{ moves }}"          # the foreach over graph/move-node, one per plan.moves entry
    workflow: graph-janitor`;

export default defineStep({
  type: "janitor/readback",
  description:
    "After a janitor plan's writes: read each write's result (graph steps return error strings, not exceptions), re-read every " +
    "touched node live and sort each change into `applied` (the node now holds the planned value / hangs under the planned " +
    "parent), `failed` (the step's error text) or `not_verified` (success reported, read-back differs). Writes `applied.json` " +
    "(`file`) into the run's artifacts — the undo log graph-janitor-revert reads: every applied change with the value or " +
    "parent it replaced — appends the outcome and the revert instruction to `report` (created if absent), and FAILS with " +
    "`apply_failed:` when anything failed or could not be verified. Output: { applied, failed, not_verified, applied_json, counts }.\n\n" +
    EXAMPLE,
  input: z.object({
    plan: z.any().describe("the plan the writes followed: { edits, moves }"),
    edits: z.array(z.any()).default([]).describe("the foreach results over graph/edit-node, aligned with plan.edits"),
    moves: z.array(z.any()).default([]).describe("the foreach results over graph/move-node, aligned with plan.moves"),
    workflow: z.string().default("graph-janitor").describe("recorded in the log"),
    revert_of: z.string().optional().describe("when this run reverts another: that run's id, recorded in the log"),
    file: z.string().default("applied.json"),
    report: z.string().default("report.md").describe("the report the outcome is appended to"),
    revert_workflow: z.string().default("graph-janitor-revert").describe("named in the revert instruction"),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const c = ctx as StepContext<StrutCapabilities>;
    const plan = (cfg.plan ?? {}) as { edits?: Change[]; moves?: Change[] };
    const changes = [...(plan.edits ?? []), ...(plan.moves ?? [])];
    const live = changes.length ? await readState(graphCtx(c), changes.map((x) => x.ref_id)) : new Map<string, LiveState>();
    const v = verdicts(plan, cfg.edits, cfg.moves, live);
    const total = changes.length;
    const log = {
      run_id: ctx.runId,
      workflow: cfg.workflow,
      ...(cfg.revert_of ? { revert_of: cfg.revert_of } : {}),
      written_at: new Date().toISOString(),
      counts: { planned: total, applied: v.applied.length, failed: v.failed.length, not_verified: v.not_verified.length },
      applied: v.applied,
      failed: v.failed,
      not_verified: v.not_verified,
    };
    const art = c.services?.artifacts;
    if (art) {
      await art.write(ctx.runId, cfg.file, JSON.stringify(log, null, 2));
      let report = "";
      try {
        report = Buffer.from(await art.read(ctx.runId, cfg.report)).toString();
      } catch {
        report = "";
      }
      const section = reportSection(ctx.runId, v, { revert_workflow: cfg.revert_workflow, revert_of: cfg.revert_of, file: cfg.file });
      await art.write(ctx.runId, cfg.report, (report ? report.replace(/\s*$/, "\n\n") : "") + section);
    }
    const out = withAccessedNodes(
      { applied: v.applied, failed: v.failed, not_verified: v.not_verified, applied_json: `/artifacts/${ctx.runId}/${cfg.file}`, counts: log.counts },
      [...live.values()].map((n) => ({ ref_id: n.ref_id, node_type: n.node_type, name: n.name })),
    );
    if (v.failed.length || v.not_verified.length) {
      throw new Error(
        `apply_failed: ${v.failed.length} of ${total} changes failed and ${v.not_verified.length} could not be verified (${v.applied.length} applied) — every verdict is in /artifacts/${ctx.runId}/${cfg.file}`,
      );
    }
    return out;
  },
});
