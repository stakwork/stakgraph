import { z } from "zod";
import pg from "pg";
import { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { Tool } from "../types.js";
import { parseSchema } from "../utils.js";

// ---------------------------------------------------------------------------
// Per-session evidence collector (server-side proof tagging).
// Every probe result is tagged with an ev{N} id and marked "strong"; the
// submit_verdict guard only accepts a works claim backed by a strong id.
// ---------------------------------------------------------------------------

interface EvidenceRecord {
  id: string;
  kind: string;
  summary: string;
}

interface SessionEvidence {
  records: EvidenceRecord[];
  strong: Set<string>;
}

const EVIDENCE: Record<string, SessionEvidence> = {};
const VERDICTS: Record<string, unknown> = {};

function sess(sessionId: string): SessionEvidence {
  if (!EVIDENCE[sessionId]) EVIDENCE[sessionId] = { records: [], strong: new Set() };
  return EVIDENCE[sessionId];
}

export function pushEvidence(sessionId: string, kind: string, summary: string, strong = false): string {
  const s = sess(sessionId);
  const id = `ev${s.records.length + 1}`;
  s.records.push({ id, kind, summary });
  if (strong) s.strong.add(id);
  return id;
}

export function resetVerifySession(sessionId: string): void {
  delete EVIDENCE[sessionId];
  delete VERDICTS[sessionId];
}

export function getVerdict(sessionId: string): unknown {
  return VERDICTS[sessionId];
}

// Which tool names produce strong proof, and under what evidence kind.
const PROBE_KIND: Record<string, string> = {
  stagehand_network_activity: "network",
  stagehand_logs: "console",
  stagehand_screenshot: "screenshot",
  stagehand_extract: "dom",
  [/* db_query */ "verify_db_query"]: "db",
};

function textItems(result: CallToolResult): Array<{ type: string; text?: string }> {
  return (result.content as Array<{ type: string; text?: string }>) || [];
}

// Tag a probe result with a captured evidence id the agent can cite in proof[].
export function tagEvidence(sessionId: string, toolName: string, result: CallToolResult): CallToolResult {
  const kind = PROBE_KIND[toolName];
  if (!kind) return result;
  const id = pushEvidence(sessionId, kind, toolName, true);
  return {
    ...result,
    content: [
      ...textItems(result),
      { type: "text", text: `[captured probe evidence ${id} (${kind}) — cite ${id} in proof[]]` },
    ],
  } as CallToolResult;
}

function text(t: string): CallToolResult {
  return { content: [{ type: "text", text: t }] };
}

// ---------------------------------------------------------------------------
// D2 · db_query — read-only Postgres probe (read-after-write proof)
// ---------------------------------------------------------------------------

const DbQuerySchema = z.object({
  query: z.string().describe("A single read-only SELECT statement."),
});

const DB_ROW_CAP = 50;

export const DbQueryTool: Tool = {
  name: "verify_db_query",
  description:
    "Run a READ-ONLY SQL query (Postgres) against the app database to independently confirm state persisted — e.g. after a write through the UI, SELECT the row to prove it exists. Enforced read-only (read-only transaction + statement timeout + row cap), so a write attempt fails honestly. Only available when a database URL is configured.",
  inputSchema: parseSchema(DbQuerySchema),
};

export async function dbQuery(args: Record<string, unknown>): Promise<CallToolResult> {
  const { query } = DbQuerySchema.parse(args);
  const url = process.env.AUDIT_DB_URL || process.env.DATABASE_URL;
  if (!url) return text(JSON.stringify({ unavailable: true, message: "no database configured" }));
  const client = new pg.Client({ connectionString: url, connectionTimeoutMillis: 5000 });
  try {
    await client.connect();
    await client.query("SET default_transaction_read_only = on");
    await client.query("BEGIN");
    await client.query("SET TRANSACTION READ ONLY");
    await client.query("SET LOCAL statement_timeout = '5s'");
    const res = await client.query({ text: query, values: [] });
    await client.query("ROLLBACK");
    const rows = res.rows.slice(0, DB_ROW_CAP);
    return text(JSON.stringify({ rowCount: res.rowCount, rows }));
  } catch (err: any) {
    return text(JSON.stringify({ error: `db_query failed: ${err?.message ?? String(err)}` }));
  } finally {
    await client.end().catch(() => {});
  }
}

// ---------------------------------------------------------------------------
// D3 · submit_verdict — the shared proof contract (guard + reconcile)
// ---------------------------------------------------------------------------

const OutcomeSchema = z.enum(["works", "broken", "unknown"]);

const ClaimSchema = z.object({
  claim: z.string().describe("The specific thing the task claimed to do."),
  verdict: OutcomeSchema,
  proof: z
    .array(z.string())
    .describe(
      "Captured evidence ids that back this verdict — the ev ids returned by probe tools (verify_db_query, stagehand_network_activity, stagehand_logs, stagehand_screenshot, stagehand_extract). A works verdict with no such id is downgraded to unknown.",
    ),
  reasoning: z.string(),
});

const VerdictSchema = z.object({
  overall: OutcomeSchema,
  claims: z.array(ClaimSchema),
  observations: z.array(z.string()),
  summary: z.string(),
});

export const SubmitVerdictTool: Tool = {
  name: "submit_verdict",
  description:
    "Submit the final verdict and END. A claim may be marked works ONLY if its proof[] cites at least one captured probe-evidence id; a works claim with no such id is downgraded to unknown and overall follows. This is the terminal tool.",
  inputSchema: parseSchema(VerdictSchema),
};

export async function submitVerdict(sessionId: string, args: Record<string, unknown>): Promise<CallToolResult> {
  const input = VerdictSchema.parse(args);
  const strong = sess(sessionId).strong;
  const notes: string[] = [];

  const claims = input.claims.map((c) => {
    if (c.verdict !== "works") return c;
    const backed = c.proof.filter((id) => strong.has(id));
    if (backed.length === 0) {
      notes.push(`Guard: claim "${c.claim}" marked works with no captured proof; downgraded to unknown.`);
      return {
        ...c,
        verdict: "unknown" as const,
        proof: backed,
        reasoning: `${c.reasoning} [guard: no captured proof backed this works claim]`,
      };
    }
    return { ...c, proof: backed };
  });

  const hasBroken = claims.some((c) => c.verdict === "broken");
  const allWorks = claims.length > 0 && claims.every((c) => c.verdict === "works");
  let overall = input.overall;
  if (overall === "works" && !allWorks) {
    overall = hasBroken ? "broken" : "unknown";
    notes.push(`Guard: overall downgraded from works to ${overall} because not every claim is backed as works.`);
  }

  const verdict = {
    overall,
    claims,
    observations: notes.length > 0 ? [...input.observations, ...notes] : input.observations,
    summary: input.summary,
    evidence: sess(sessionId).records,
  };
  VERDICTS[sessionId] = verdict;
  return text(JSON.stringify(verdict));
}

export const VERIFY_TOOLS: Tool[] = [DbQueryTool, SubmitVerdictTool];
