/**
 * Read-only raw Cypher against the vein graph backend — the chat builder's
 * `graph_query` tool (see ai/tools.ts). It exists so the assistant can
 * VERIFY what a workflow's `graph/*` steps actually wrote (counts by type,
 * exact properties, edge fan-out) — questions the typed read steps can't
 * answer (search is ranked + limited; get/neighbors need a ref_id).
 *
 * Deliberately NOT a workflow step: raw Cypher in published YAML would bypass
 * the schema validation and embeddings-on-write the `graph/*` steps exist to
 * provide, and since the workspace itself lives in this graph a stray write
 * corrupts the workflow store, not scratch data. Hence read-only, enforced
 * twice: a keyword pre-check for a clear error, and — the real guarantee —
 * a READ-mode transaction, which the server rejects writes in ("Writing in
 * read access mode not allowed").
 *
 * Output is capped so a `MATCH (n) RETURN n` can't hang the turn or flood
 * the context: a row cap (streamed, so the rest is never fetched), a
 * server-side transaction timeout, long strings truncated, and embedding
 * vectors (any long numeric array) collapsed to a placeholder.
 */
import type { GraphBackend } from "./backend.js";
import { rowOf, type Params, type Row } from "./bolt.js";

export interface ReadQueryOptions {
  params?: Params;
  /** Max rows returned (default 100, hard max 1000). */
  maxRows?: number;
  /** Server-side transaction timeout in ms (default 15s, hard max 60s). */
  timeoutMs?: number;
  /** Strings longer than this are truncated (default 500). */
  maxStringLength?: number;
}

export interface ReadQueryResult {
  columns: string[];
  rows: Row[];
  /** Rows returned (≤ maxRows). */
  rowCount: number;
  /** True when the query produced more rows than maxRows. */
  truncated: boolean;
  elapsedMs: number;
}

export const DEFAULT_MAX_ROWS = 100;
export const HARD_MAX_ROWS = 1000;
export const DEFAULT_TIMEOUT_MS = 15_000;
export const HARD_MAX_TIMEOUT_MS = 60_000;
export const DEFAULT_MAX_STRING = 500;
/** Numeric arrays at least this long are assumed to be embedding vectors. */
const VECTOR_MIN_LEN = 32;

/** Thrown by the pre-check; the message names the offending keyword. */
export class ReadOnlyViolation extends Error {
  readonly name = "ReadOnlyViolation";
  constructor(readonly keyword: string) {
    super(
      `graph_query is read-only: "${keyword}" is not allowed. ` +
        "Write through the graph/* steps (create-node, create-triplet, edit-node, …) instead.",
    );
  }
}

// Cypher clauses / procedures that mutate. Word-bounded and checked with
// string literals, backtick identifiers, and comments stripped, so a
// property called `data_set` or a search term 'reset' doesn't trip it.
// `apoc.` is blanket-blocked: many apoc procedures write, and none are
// needed to inspect a graph.
const WRITE_PATTERNS: Array<[RegExp, string]> = [
  [/\bCREATE\b/i, "CREATE"],
  [/\bMERGE\b/i, "MERGE"],
  [/\bDELETE\b/i, "DELETE"],
  [/\bDETACH\b/i, "DETACH"],
  [/\bSET\b/i, "SET"],
  [/\bREMOVE\b/i, "REMOVE"],
  [/\bDROP\b/i, "DROP"],
  [/\bFOREACH\b/i, "FOREACH"],
  [/\bLOAD\s+CSV\b/i, "LOAD CSV"],
  [/\bALTER\b/i, "ALTER"],
  [/\b(GRANT|DENY|REVOKE)\b/i, "GRANT/DENY/REVOKE"],
  [/\b(START|STOP)\s+DATABASE\b/i, "START/STOP DATABASE"],
  [/\bapoc\./i, "apoc.*"],
  [/\bdb\.(create|drop|index\.fulltext\.(create|drop)|index\.vector\.create)/i, "db.create*/db.drop*"],
];

/** Strip string literals, backtick identifiers, and comments so keyword
 *  matching only sees real Cypher tokens. */
function stripLiterals(cypher: string): string {
  return cypher
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(/\/\/[^\n]*/g, " ")
    .replace(/'(?:[^'\\]|\\.)*'/g, "''")
    .replace(/"(?:[^"\\]|\\.)*"/g, '""')
    .replace(/`[^`]*`/g, "``");
}

/** The first write keyword found in `cypher`, or undefined when it looks
 *  read-only. A pre-check only — the READ transaction is the guarantee. */
export function findWriteKeyword(cypher: string): string | undefined {
  const bare = stripLiterals(cypher);
  for (const [re, name] of WRITE_PATTERNS) if (re.test(bare)) return name;
  return undefined;
}

/** Shrink a value for the model: truncate long strings, collapse embedding
 *  vectors, recurse into maps/arrays. */
export function compactValue(v: unknown, maxString = DEFAULT_MAX_STRING): unknown {
  if (typeof v === "string") {
    return v.length > maxString ? `${v.slice(0, maxString)}… [+${v.length - maxString} chars]` : v;
  }
  if (Array.isArray(v)) {
    if (v.length >= VECTOR_MIN_LEN && v.every((x) => typeof x === "number")) {
      return `[vector: ${v.length} numbers]`;
    }
    return v.map((x) => compactValue(x, maxString));
  }
  if (v && typeof v === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, x] of Object.entries(v as Record<string, unknown>)) out[k] = compactValue(x, maxString);
    return out;
  }
  return v;
}

/**
 * Run `cypher` in a READ transaction and return up to `maxRows` compacted
 * rows. Throws `ReadOnlyViolation` on a write keyword; driver errors (syntax,
 * timeout, server-side write rejection) propagate as-is.
 */
export async function readQuery(
  backend: GraphBackend,
  cypher: string,
  opts: ReadQueryOptions = {},
): Promise<ReadQueryResult> {
  const kw = findWriteKeyword(cypher);
  if (kw) throw new ReadOnlyViolation(kw);
  const maxRows = Math.min(Math.max(1, Math.trunc(opts.maxRows ?? DEFAULT_MAX_ROWS)), HARD_MAX_ROWS);
  const timeout = Math.min(Math.max(1, Math.trunc(opts.timeoutMs ?? DEFAULT_TIMEOUT_MS)), HARD_MAX_TIMEOUT_MS);
  const maxString = opts.maxStringLength ?? DEFAULT_MAX_STRING;

  const started = Date.now();
  const session = backend.bolt.session("READ");
  try {
    return await session.executeRead(
      async (tx) => {
        const result = tx.run(cypher, opts.params ?? {});
        const rows: Row[] = [];
        let columns: string[] = [];
        let truncated = false;
        // Stream so a huge result stops at the cap instead of being fetched
        // whole; breaking out of the iterator discards the remainder.
        for await (const rec of result) {
          if (columns.length === 0) columns = rec.keys.map(String);
          if (rows.length >= maxRows) {
            truncated = true;
            break;
          }
          rows.push(compactValue(rowOf(rec), maxString) as Row);
        }
        if (columns.length === 0) columns = (await result.keys()).map(String);
        return { columns, rows, rowCount: rows.length, truncated, elapsedMs: Date.now() - started };
      },
      { timeout },
    );
  } finally {
    await session.close();
  }
}
