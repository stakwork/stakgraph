import { z } from "zod";
import pg from "pg";
import { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { Tool } from "../types.js";
import { parseSchema } from "../utils.js";

interface EvidenceRecord {
  id: string;
  kind: string;
  summary: string;
  data: string;
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

export function pushEvidence(
  sessionId: string,
  kind: string,
  summary: string,
  data: string,
  strong = false,
): string {
  const s = sess(sessionId);
  const id = `ev${s.records.length + 1}`;
  s.records.push({ id, kind, summary, data });
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

const PROBE_KIND: Record<string, string> = {
  stagehand_network_activity: "network",
  stagehand_logs: "console",
  stagehand_screenshot: "screenshot",
  stagehand_extract: "dom",
};

function text(t: string): CallToolResult {
  return { content: [{ type: "text", text: t }] };
}

function payloadOf(result: CallToolResult): { data: string; text: string } {
  const items = (result.content as Array<{ type: string; text?: string; data?: string }>) || [];
  const texts: string[] = [];
  let img = "";
  for (const it of items) {
    if (it.type === "text" && it.text) texts.push(it.text);
    else if (it.type === "image" && it.data) img = it.data;
  }
  return { data: img || texts.join("\n"), text: texts.join("\n") };
}

// Tag a stagehand probe result with a captured evidence id the agent can cite.
export function tagEvidence(sessionId: string, toolName: string, result: CallToolResult): CallToolResult {
  const kind = PROBE_KIND[toolName];
  if (!kind) return result;
  const { data } = payloadOf(result);
  const id = pushEvidence(sessionId, kind, `${kind} via ${toolName}`, data.slice(0, 4000), true);
  return {
    ...result,
    content: [
      ...((result.content as Array<unknown>) || []),
      { type: "text", text: `[captured probe evidence ${id} (${kind}) — cite ${id} in proof[]]` },
    ],
  } as CallToolResult;
}

// ---------------------------------------------------------------------------
// http_request — timed HTTP probe against the running app / its API
// ---------------------------------------------------------------------------

const HttpRequestSchema = z.object({
  url: z.string().describe("Absolute URL to request."),
  method: z.string().optional().describe("HTTP method (default GET)."),
  headers: z.record(z.string(), z.string()).optional(),
  body: z.string().optional().describe("Raw request body, if any."),
});

export const HttpRequestTool: Tool = {
  name: "verify_http_request",
  description:
    "Make a timed HTTP request against the running app or its API. Returns status, elapsed ms, response headers, and a snippet of the body. Captures an http evidence record and returns its id to cite in proof[]. Prefer this over driving the browser when the check is about an API's status/shape.",
  inputSchema: parseSchema(HttpRequestSchema),
};

export async function httpRequest(sessionId: string, args: Record<string, unknown>): Promise<CallToolResult> {
  const { url, method, headers, body } = HttpRequestSchema.parse(args);
  const start = Date.now();
  try {
    const resp = await fetch(url, { method: method ?? "GET", headers, body: body ?? undefined });
    const ms = Date.now() - start;
    const bodyText = await resp.text();
    const respHeaders: Record<string, string> = {};
    resp.headers.forEach((v, k) => (respHeaders[k] = v));
    const bodySnippet = bodyText.slice(0, 2000);
    const id = pushEvidence(
      sessionId,
      "http",
      `HTTP ${method ?? "GET"} ${url} -> ${resp.status} in ${ms}ms`,
      JSON.stringify({ status: resp.status, ms, headers: respHeaders, bodySnippet }),
      true,
    );
    return text(JSON.stringify({ id, status: resp.status, ms, headers: respHeaders, bodySnippet }));
  } catch (err: any) {
    const ms = Date.now() - start;
    const message = err?.message ?? String(err);
    const id = pushEvidence(
      sessionId,
      "http",
      `HTTP ${method ?? "GET"} ${url} -> request failed in ${ms}ms`,
      JSON.stringify({ status: 0, ms, bodySnippet: `request failed: ${message}` }),
      true,
    );
    return text(JSON.stringify({ id, status: 0, ms, bodySnippet: `request failed: ${message}` }));
  }
}

// ---------------------------------------------------------------------------
// sample — timing over n requests
// ---------------------------------------------------------------------------

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
  return sorted[idx];
}

const SampleSchema = z.object({
  url: z.string().describe("Absolute URL to sample."),
  n: z.number().describe("Number of requests to make."),
});

export const SampleTool: Tool = {
  name: "verify_sample",
  description:
    "Call a URL n times and measure timing. Returns count, median ms, p95 ms, and the samples. Captures a timing evidence record and returns its id to cite in proof[]. Use for performance/timing claims.",
  inputSchema: parseSchema(SampleSchema),
};

export async function sampleUrl(sessionId: string, args: Record<string, unknown>): Promise<CallToolResult> {
  const { url, n } = SampleSchema.parse(args);
  const count = Math.max(1, Math.min(50, Math.floor(n)));
  const samples: number[] = [];
  for (let i = 0; i < count; i++) {
    const start = Date.now();
    try {
      const resp = await fetch(url, { method: "GET" });
      await resp.arrayBuffer();
    } catch {
      /* record the elapsed time of the failed attempt */
    }
    samples.push(Date.now() - start);
  }
  const sorted = [...samples].sort((a, b) => a - b);
  const medianMs = percentile(sorted, 50);
  const p95Ms = percentile(sorted, 95);
  const id = pushEvidence(
    sessionId,
    "timing",
    `sampled ${url} n=${count} median=${medianMs}ms p95=${p95Ms}ms`,
    JSON.stringify({ count, medianMs, p95Ms, samples }),
    true,
  );
  return text(JSON.stringify({ id, count, medianMs, p95Ms, samples }));
}

// ---------------------------------------------------------------------------
// db_query — read-only Postgres probe (read-after-write proof)
// ---------------------------------------------------------------------------

const DbQuerySchema = z.object({
  query: z.string().describe("A single read-only SELECT statement."),
});

const DB_ROW_CAP = 50;

export const DbQueryTool: Tool = {
  name: "verify_db_query",
  description:
    "Run a READ-ONLY SQL query (Postgres) against the app database to independently confirm state persisted — e.g. after a write through the UI, SELECT the row to prove it exists. Enforced read-only. Captures a db evidence record and returns its id to cite in proof[]. Only available when a database URL is configured.",
  inputSchema: parseSchema(DbQuerySchema),
};

export async function dbQuery(sessionId: string, args: Record<string, unknown>): Promise<CallToolResult> {
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
    const id = pushEvidence(
      sessionId,
      "db",
      `db_query rows=${res.rowCount ?? rows.length}`,
      JSON.stringify({ rowCount: res.rowCount, rows }),
      true,
    );
    return text(JSON.stringify({ id, rowCount: res.rowCount, rows }));
  } catch (err: any) {
    const message = err?.message ?? String(err);
    const id = pushEvidence(sessionId, "db", "db_query failed", message, true);
    return text(JSON.stringify({ id, error: `db_query failed: ${message}` }));
  } finally {
    await client.end().catch(() => {});
  }
}

// ---------------------------------------------------------------------------
// submit_verdict — the shared proof contract (guard + reconcile)
// ---------------------------------------------------------------------------

const OutcomeSchema = z.enum(["works", "broken", "unknown"]);

const ClaimSchema = z.object({
  claim: z.string().describe("The specific thing the task claimed to do."),
  verdict: OutcomeSchema,
  proof: z
    .array(z.string())
    .describe(
      "Captured evidence ids that back this verdict — the ev ids returned by probe tools (verify_http_request, verify_sample, verify_db_query, stagehand_network_activity, stagehand_logs, stagehand_screenshot, stagehand_extract). A works verdict with no such id is downgraded to unknown.",
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

export const VERIFY_TOOLS: Tool[] = [HttpRequestTool, SampleTool, DbQueryTool, SubmitVerdictTool];
