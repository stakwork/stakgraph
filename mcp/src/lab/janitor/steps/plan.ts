import { z, defineStep, openGraphBackend, withAccessedNodes, type StepContext, type StrutCapabilities } from "strut";

/**
 * janitor/plan — the gate between what the janitor agent PROPOSES and what
 * code WRITES. The agent is read-only and returns findings; each may carry
 * a typed `action` (set_property | move | none). This step re-reads every
 * target node live, checks each action against the rules below, and sorts
 * it into `edits` / `moves` (what the apply branch writes, in the exact
 * shape graph/edit-node and graph/move-node take), `skipped` (with a
 * reason) or `manual` (no action: a person decides). It writes `plan.json`
 * into the run's artifacts and runs in BOTH modes, so a dry run shows what
 * an apply run would have written.
 *
 * Never applied, whatever the agent says:
 *   - the Janitor node (`parent`), every child of it (the mandates — one
 *     mandate must never rewrite another: the next run would obey text an
 *     LLM wrote) and anything hanging under it live, plus `protect`;
 *   - a node the git seed owns: `unique_source_id` starting with
 *     `seeded_prefix` (concept-seed.ts stamps every Concept it ships; the
 *     next file change would overwrite a janitor edit anyway — fixes to
 *     those belong in git). Strut's read steps strip that property from
 *     every response, so it is read here over bolt;
 *   - a node not in `visited` or not of `node_type`, a property outside
 *     `properties` (no renames: `name` is the identity key and automations
 *     address Concepts by it), a value that no longer matches what the
 *     agent read (`stale`), a move with no or several current parents.
 * `max_changes` caps a run's writes, most severe first.
 *
 * Two shapes of edit. `set_property` carries the whole new value (and the
 * whole old one, as read). `replace_text` carries only the fragment that
 * changes — `find`, verbatim, and `replace` — and THIS step resolves it
 * against the live value: `find` must occur exactly once (that is its
 * stale check: zero = the text moved on, several = ambiguous), and the
 * planned change is an ordinary edit with the full old and new values, so
 * the write, the readback and the undo log never see a fragment.
 *
 * `graphCtx` is strut's `graph/*` preamble, inlined (seeded steps import
 * only "strut"): the same config and options, so the same cached backend.
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

/** A node as the graph holds it NOW — with the system properties the read steps strip. */
export interface LiveNode {
  ref_id: string;
  labels: string[];
  name: string;
  properties: Record<string, unknown>;
  /** ref_ids of the nodes with a PARENT_OF edge INTO this one. */
  parents: string[];
  deleted: boolean;
  /** `unique_source_id`, the seed's stamp when the git seed owns the node. */
  stamp: string | null;
}

/** One bolt round trip for every ref_id: labels, every property, current parents. */
export async function readLive(b: { bolt: { run(q: string, p?: Record<string, unknown>): Promise<Array<Record<string, unknown>>> } }, ref_ids: string[]): Promise<Map<string, LiveNode>> {
  const ids = [...new Set(ref_ids.filter((r) => typeof r === "string" && r.length > 0))];
  const out = new Map<string, LiveNode>();
  if (ids.length === 0) return out;
  const rows = await b.bolt.run(
    `UNWIND $ids AS id
     MATCH (n:Data_Bank {ref_id: id})
     OPTIONAL MATCH (p:Data_Bank)-[:PARENT_OF]->(n)
     RETURN id, labels(n) AS labels, properties(n) AS props, collect(DISTINCT p.ref_id) AS parents`,
    { ids },
  );
  for (const r of rows) {
    const props = (r["props"] ?? {}) as Record<string, unknown>;
    const stamp = props["unique_source_id"];
    out.set(String(r["id"]), {
      ref_id: String(r["id"]),
      labels: ((r["labels"] ?? []) as unknown[]).map(String),
      name: typeof props["name"] === "string" ? (props["name"] as string) : "",
      properties: props,
      parents: ((r["parents"] ?? []) as unknown[]).filter((p) => typeof p === "string") as string[],
      deleted: props["deleted_at"] != null || props["is_deleted"] === true,
      stamp: typeof stamp === "string" && stamp ? stamp : null,
    });
  }
  return out;
}

/** The property as a string to compare: absent → null. */
export function liveValue(node: LiveNode, property: string): string | null {
  const v = node.properties[property];
  if (v === null || v === undefined) return null;
  return typeof v === "string" ? v : String(v);
}

export type Severity = "high" | "medium" | "low";
const RANK: Record<string, number> = { high: 3, medium: 2, low: 1 };

export interface Action {
  kind?: "set_property" | "replace_text" | "move" | "none" | string;
  property?: string;
  old_value?: string | null;
  new_value?: string | null;
  /** replace_text: the fragment to replace, verbatim as read; must occur exactly once. */
  find?: string;
  /** replace_text: what takes its place ("" removes it). */
  replace?: string;
  new_parent_ref_id?: string;
}

export interface Finding {
  ref_id: string;
  name?: string;
  issue?: string;
  problem?: string;
  evidence?: string;
  suggestion?: string;
  severity?: Severity | string;
  action?: Action;
}

/** One planned write, in the shape the foreach bodies hand to graph/edit-node or graph/move-node. */
export interface Change {
  /** `e1`, `m2`… — the handle a revert names. */
  id: string;
  kind: "edit" | "move";
  ref_id: string;
  name: string;
  finding: { issue: string; severity: string; problem: string };
  // edit
  property?: string;
  old_value?: string | null;
  new_value?: string | null;
  /** When the edit came in as replace_text: the fragment, for the record. */
  via?: { find: string; replace: string };
  node_data: Record<string, string>;
  properties_to_be_deleted: string[];
  // move
  edge_type?: string;
  direction?: "forward" | "reverse";
  from_ref_id?: string;
  to_ref_id?: string;
}

export interface Skipped {
  ref_id: string;
  name: string;
  issue: string;
  reason: string;
  detail?: string;
}

export interface Manual {
  ref_id: string;
  name: string;
  issue: string;
  suggestion: string;
}

export interface Plan {
  edits: Change[];
  moves: Change[];
  skipped: Skipped[];
  manual: Manual[];
}

export interface PlanOptions {
  visited: Array<{ ref_id: string }>;
  /** ref_ids never written: the Janitor node, its children, the mandate, `protect`. */
  protect: Iterable<string>;
  /** The Janitor node: anything hanging under it live is protected too (past the 50 children a read lists). */
  parent_ref_id?: string;
  node_type: string;
  properties: Iterable<string>;
  min_severity: string;
  issues?: string[];
  max_changes: number;
  seeded_prefix: string;
}

/** Pure: the rules in the module comment, over findings and the live nodes they name. */
export function planChanges(findings: Finding[], live: Map<string, LiveNode>, opts: PlanOptions): Plan {
  const visited = new Set(opts.visited.map((v) => v.ref_id));
  const protect = new Set(opts.protect);
  const allowed = new Set(opts.properties);
  const issues = opts.issues && opts.issues.length ? new Set(opts.issues) : null;
  const minRank = RANK[opts.min_severity] ?? 1;
  const plan: Plan = { edits: [], moves: [], skipped: [], manual: [] };
  const seen = new Set<string>();
  let n = 0;
  const isProtected = (ref: string, node?: LiveNode) => protect.has(ref) || (!!opts.parent_ref_id && !!node && node.parents.includes(opts.parent_ref_id));
  // Most severe first, stable: the cap keeps what matters most.
  const ordered = findings
    .map((f, i) => ({ f, i }))
    .sort((a, b) => (RANK[String(b.f.severity)] ?? 0) - (RANK[String(a.f.severity)] ?? 0) || a.i - b.i)
    .map((x) => x.f);
  for (const f of ordered) {
    const ref = typeof f.ref_id === "string" ? f.ref_id : "";
    const name = typeof f.name === "string" ? f.name : "";
    const issue = typeof f.issue === "string" ? f.issue : "";
    const skip = (reason: string, detail?: string) => plan.skipped.push({ ref_id: ref, name, issue, reason, ...(detail ? { detail } : {}) });
    const a = f.action;
    if (!a || !a.kind || a.kind === "none") {
      plan.manual.push({ ref_id: ref, name, issue, suggestion: typeof f.suggestion === "string" ? f.suggestion : "" });
      continue;
    }
    if (a.kind !== "set_property" && a.kind !== "replace_text" && a.kind !== "move") {
      skip("invalid_action", `unknown action kind "${String(a.kind)}"`);
      continue;
    }
    if (!ref) {
      skip("invalid_action", "a finding without a ref_id");
      continue;
    }
    const node = live.get(ref);
    if (isProtected(ref, node)) {
      skip("protected", "the Janitor node, a mandate, or something hanging under Janitor");
      continue;
    }
    if (!visited.has(ref)) {
      skip("not_visited", "the finding names a node the agent did not record as inspected");
      continue;
    }
    if ((RANK[String(f.severity)] ?? 1) < minRank) {
      skip("below_min_severity", `${String(f.severity)} < ${opts.min_severity}`);
      continue;
    }
    if (issues && !issues.has(issue)) {
      skip("issue_not_applied", `"${issue}" is not among the issues this run applies`);
      continue;
    }
    if (!node || node.deleted) {
      skip("not_found", "no such node in the graph now (or deleted)");
      continue;
    }
    if (!node.labels.includes(opts.node_type)) {
      skip("wrong_type", `not a ${opts.node_type}: ${node.labels.filter((l) => l !== "Data_Bank").join(", ") || "no type label"}`);
      continue;
    }
    if (node.stamp && node.stamp.startsWith(opts.seeded_prefix)) {
      skip("git_seeded", `${node.stamp} — the git seed owns it; change the file`);
      continue;
    }
    const finding = { issue, severity: String(f.severity ?? ""), problem: typeof f.problem === "string" ? f.problem : "" };
    if (a.kind === "set_property" || a.kind === "replace_text") {
      const prop = typeof a.property === "string" ? a.property.trim() : "";
      if (!prop) {
        skip("invalid_action", `${a.kind} without a property`);
        continue;
      }
      if (!allowed.has(prop)) {
        skip("property_not_allowed", `${prop} is not among ${[...allowed].join(", ") || "(none)"}`);
        continue;
      }
      const cur = liveValue(node, prop);
      let nv: string;
      let via: Change["via"];
      if (a.kind === "replace_text") {
        const find = typeof a.find === "string" ? a.find : "";
        const rep = typeof a.replace === "string" ? a.replace : "";
        if (!find) {
          skip("invalid_action", "replace_text without a find text");
          continue;
        }
        if (cur === null) {
          skip("find_not_once", `${prop} is empty on the node now — nothing to find`);
          continue;
        }
        const n = cur.split(find).length - 1;
        if (n !== 1) {
          skip("find_not_once", n === 0 ? `the find text is not in ${prop} now (it changed since the agent read it?)` : `the find text occurs ${n} times in ${prop} — ambiguous`);
          continue;
        }
        nv = cur.replace(find, rep);
        via = { find, replace: rep };
        if (!nv.trim()) {
          skip("empty_value", `replace_text would leave ${prop} empty`);
          continue;
        }
        if (nv === cur) {
          skip("no_change", `replace equals find — ${prop} would read the same`);
          continue;
        }
      } else {
        nv = typeof a.new_value === "string" ? a.new_value : "";
        if (!nv.trim()) {
          skip("empty_value", `set_property ${prop} without a new_value`);
          continue;
        }
        if (cur === nv) {
          skip("no_change", `${prop} already reads that`);
          continue;
        }
        const ov = a.old_value;
        const evidence = typeof f.evidence === "string" ? f.evidence.trim() : "";
        const fresh =
          (typeof ov === "string" && ov === cur) ||
          (cur === null && (ov === null || ov === undefined || ov === "")) ||
          (cur !== null && evidence.length > 0 && cur.includes(evidence));
        if (!fresh) {
          skip("stale", `${prop} no longer reads what the agent read (old_value and evidence both miss the live value)`);
          continue;
        }
      }
      const key = `${ref}:${prop}`;
      if (seen.has(key)) {
        skip("duplicate", `an earlier finding already changes ${prop} here`);
        continue;
      }
      if (n >= opts.max_changes) {
        skip("over_cap", `more than ${opts.max_changes} changes this run`);
        continue;
      }
      seen.add(key);
      n++;
      plan.edits.push({
        id: `e${plan.edits.length + 1}`,
        kind: "edit",
        ref_id: ref,
        name: node.name || name,
        finding,
        property: prop,
        old_value: cur,
        new_value: nv,
        ...(via ? { via } : {}),
        node_data: { [prop]: nv },
        properties_to_be_deleted: [],
      });
    } else {
      const to = typeof a.new_parent_ref_id === "string" ? a.new_parent_ref_id : "";
      if (!to) {
        skip("invalid_action", "move without a new_parent_ref_id");
        continue;
      }
      if (to === ref) {
        skip("invalid_action", "a node cannot hang under itself");
        continue;
      }
      const target = live.get(to);
      if (isProtected(to, target)) {
        skip("protected", "the new parent is the Janitor node or a mandate — that would make a mandate");
        continue;
      }
      if (!visited.has(to)) {
        skip("not_visited", "the new parent is a node the agent did not record as inspected");
        continue;
      }
      if (!target || target.deleted) {
        skip("not_found", "no such new parent in the graph now (or deleted)");
        continue;
      }
      if (!target.labels.includes(opts.node_type)) {
        skip("wrong_type", `the new parent is not a ${opts.node_type}`);
        continue;
      }
      if (node.parents.length === 0) {
        skip("no_parent", "nothing hangs it anywhere now — there is no PARENT_OF edge to move");
        continue;
      }
      if (node.parents.length > 1) {
        skip("many_parents", `${node.parents.length} parents — which edge to move is ambiguous`);
        continue;
      }
      const from = node.parents[0]!;
      if (from === to) {
        skip("no_change", "already hangs there");
        continue;
      }
      const key = `${ref}:move`;
      if (seen.has(key)) {
        skip("duplicate", "an earlier finding already moves this node");
        continue;
      }
      if (n >= opts.max_changes) {
        skip("over_cap", `more than ${opts.max_changes} changes this run`);
        continue;
      }
      seen.add(key);
      n++;
      plan.moves.push({
        id: `m${plan.moves.length + 1}`,
        kind: "move",
        ref_id: ref,
        name: node.name || name,
        finding,
        edge_type: "PARENT_OF",
        direction: "reverse",
        from_ref_id: from,
        to_ref_id: to,
        node_data: {},
        properties_to_be_deleted: [],
      });
    }
  }
  return plan;
}

/** Provenance entries for the nodes a plan read. */
export function accessed(live: Iterable<LiveNode>): Array<{ ref_id: string; node_type?: string; name?: string }> {
  return [...live].map((n) => ({ ref_id: n.ref_id, node_type: n.labels.find((l) => l !== "Data_Bank"), name: n.name }));
}

const EXAMPLE = `- id: plan
  type: janitor/plan
  depends: janitor
  config:
    findings: "{{ janitor.object.findings }}"
    visited: "{{ janitor.object.visited }}"
    parent: "{{ parent }}"            # graph/graph-get of the Janitor node, with children
    mandate_ref_id: "{{ mandate.ref_id }}"
    properties: "{{ params.apply_properties }}"
    max_changes: "{{ params.apply_max }}"`;

export default defineStep({
  type: "janitor/plan",
  description:
    "Turn a graph-janitor agent's findings into a validated write plan, without writing: re-reads every target node live " +
    "(over bolt, so it sees the git seed's stamp the read steps strip) and sorts each finding's typed `action` into " +
    "`edits` / `moves` (the exact configs for graph/edit-node and graph/move-node, `node_data` built from the property), " +
    "`skipped` (with a `reason`: protected | git_seeded | not_visited | wrong_type | not_found | property_not_allowed | " +
    "stale | find_not_once | no_change | empty_value | invalid_action | no_parent | many_parents | duplicate | below_min_severity | " +
    "issue_not_applied | over_cap) or `manual` (no action). A `replace_text` action ({ property, find, replace }) is resolved " +
    "here against the live value — `find` must occur exactly once — into an edit with the full old and new values. " +
    "Never planned: the Janitor node (`parent`), its children " +
    "and anything under it, `mandate_ref_id`, `protect`, and any node whose unique_source_id starts with `seeded_prefix`. " +
    "Writes `plan.json` (`file`) into the run's artifacts. Output: { edits, moves, skipped, manual, protected, counts }.\n\n" +
    EXAMPLE,
  input: z.object({
    findings: z.array(z.any()).default([]).describe("the agent's findings; each may carry action { kind: set_property | replace_text | move | none, property, old_value, new_value, find, replace, new_parent_ref_id }"),
    visited: z.array(z.any()).default([]).describe("the nodes the agent recorded as inspected — a finding or move target outside it is skipped"),
    parent: z.any().optional().describe("the Janitor node as graph/graph-get returns it with children: it, its children and anything under it live are protected"),
    mandate_ref_id: z.string().optional().describe("this run's mandate: protected"),
    protect: z.array(z.string()).optional().describe("more ref_ids never written"),
    node_type: z.string().default("Concept").describe("the only node type written"),
    properties: z.array(z.string()).default(["description", "docs"]).describe("the properties set_property may touch (never name: it is the identity key)"),
    min_severity: z.string().default("low").describe("low | medium | high — findings below it are skipped (below_min_severity)"),
    issues: z.array(z.string()).optional().describe("when given, only findings whose issue is listed are planned"),
    max_changes: z.number().int().nonnegative().default(20).describe("cap on edits + moves per run, most severe first; the rest are skipped (over_cap)"),
    seeded_prefix: z.string().default("lab/").describe("nodes whose unique_source_id starts with this are the git seed's: skipped (git_seeded)"),
    file: z.string().default("plan.json").describe("where in the run's artifacts the plan is written"),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const findings = cfg.findings as Finding[];
    const protect = new Set<string>(cfg.protect ?? []);
    const parent = cfg.parent as { ref_id?: string; children?: Array<{ ref_id?: string }>; children_truncated?: boolean } | undefined;
    if (parent?.ref_id) protect.add(parent.ref_id);
    for (const c of parent?.children ?? []) if (c?.ref_id) protect.add(c.ref_id);
    if (cfg.mandate_ref_id) protect.add(cfg.mandate_ref_id);
    const ids = findings.flatMap((f) => [f?.ref_id, f?.action?.new_parent_ref_id]).filter((r): r is string => typeof r === "string" && r.length > 0);
    const c = ctx as StepContext<StrutCapabilities>;
    const live = ids.length ? await readLive(await graphCtx(c), ids) : new Map<string, LiveNode>();
    const plan = planChanges(findings, live, {
      visited: (cfg.visited as Array<{ ref_id: string }>).filter((v) => v && typeof v.ref_id === "string"),
      protect,
      parent_ref_id: parent?.ref_id,
      node_type: cfg.node_type,
      properties: cfg.properties,
      min_severity: cfg.min_severity,
      issues: cfg.issues,
      max_changes: cfg.max_changes,
      seeded_prefix: cfg.seeded_prefix,
    });
    const out = {
      ...plan,
      protected: [...protect],
      counts: { findings: findings.length, edits: plan.edits.length, moves: plan.moves.length, skipped: plan.skipped.length, manual: plan.manual.length },
      ...(parent?.children_truncated ? { note: "Janitor lists more than 50 children; nodes hanging under it live are protected by their edge, not that list" } : {}),
    };
    await c.services?.artifacts?.write(ctx.runId, cfg.file, JSON.stringify(out, null, 2));
    return withAccessedNodes(out, accessed(live.values()));
  },
});
