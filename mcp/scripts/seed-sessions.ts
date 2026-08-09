/**
 * seed-sessions.ts
 *
 * Seeds ~250 AgentSession nodes into Neo4j so the paginated Sessions viewer
 * can be tested end-to-end (infinite scroll, filters, child_count, ghost rows).
 *
 * Conventions match existing seed scripts:
 *   - Import from "../src/graph/neo4j.js"
 *   - All nodes use  n.file = 'session://generated'  (the standard marker)
 *   - start_time is epoch milliseconds (matching UPSERT_AGENT_SESSION_QUERY)
 *   - Run with:  npx tsx scripts/seed-sessions.ts
 */

import { db } from "../src/graph/neo4j.js";

// ── helpers ────────────────────────────────────────────────────────────────

function randomHex(len = 8): string {
  return Math.random().toString(16).slice(2, 2 + len).padStart(len, "0");
}

function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function randomPick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

/** Generate an epoch-ms timestamp at a random offset within [daysAgo-1, daysAgo] days back. */
function tsForDay(daysAgo: number): number {
  const base = Date.now() - daysAgo * 24 * 60 * 60 * 1000;
  return base + randomInt(0, 23 * 60 * 60 * 1000);
}

// ── seed data ───────────────────────────────────────────────────────────────

const REPOS = [
  "stakwork/stakgraph",
  "stakwork/sphinx-tribes",
  "stakwork/sphinx-swarm",
  "stakwork/jarvis-boltwall",
  "stakwork/hive-relay",
];

const SOURCES = [
  "repo_agent",
  "gitree",
  "explore",
  "ask",
  "logs_agent",
  "mocks",
  "services",
  "graph_sub_agent",
];

const MODELS = [
  "claude-sonnet-4-5",
  "claude-opus-4-5",
  "claude-3-5-haiku-20241022",
  "gpt-4o",
  "gpt-4o-mini",
];

const PROVIDERS = ["anthropic", "openai"];

interface SessionSpec {
  session_id: string;
  source: string;
  repo: string;
  model: string;
  provider: string;
  start_time: number;
  end_time: number;
  input_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  output_tokens: number;
  total_tokens: number;
  duration_ms: number;
  status: "success" | "error";
  error_message: string;
  parent_session_id: string;
}

function makeSession(overrides: Partial<SessionSpec> = {}): SessionSpec {
  const model = overrides.model ?? randomPick(MODELS);
  const provider = overrides.provider ?? (model.startsWith("claude") ? "anthropic" : "openai");
  const input = overrides.input_tokens ?? randomInt(800, 40_000);
  const cache_read = overrides.cache_read_tokens ?? randomInt(0, input);
  const cache_write = overrides.cache_write_tokens ?? randomInt(0, 2_000);
  const output = overrides.output_tokens ?? randomInt(200, 4_000);
  const total = input + cache_read + cache_write + output;
  const start_time = overrides.start_time ?? tsForDay(randomInt(0, 90));
  const duration_ms = overrides.duration_ms ?? randomInt(5_000, 180_000);
  const status = overrides.status ?? (Math.random() < 0.07 ? "error" : "success");

  return {
    session_id: overrides.session_id ?? `seed-${randomHex(8)}-${randomHex(4)}`,
    source: overrides.source ?? randomPick(SOURCES),
    repo: overrides.repo ?? randomPick(REPOS),
    model,
    provider,
    start_time,
    end_time: start_time + duration_ms,
    input_tokens: input,
    cache_read_tokens: cache_read,
    cache_write_tokens: cache_write,
    output_tokens: output,
    total_tokens: total,
    duration_ms,
    status,
    error_message: status === "error" ? "Simulated error from seed script" : "",
    parent_session_id: "",
    ...overrides,
  };
}

// Upsert a single AgentSession node, matching the shape of
// UPSERT_AGENT_SESSION_QUERY (which the runtime uses).
const UPSERT = `
MERGE (n:AgentSession:Data_Bank {node_key: $session_id})
ON CREATE SET
  n.ref_id            = randomUUID(),
  n.date_added_to_graph = $ts,
  n.namespace         = 'default',
  n.name              = $session_id,
  n.file              = 'session://generated',
  n.start             = 0,
  n.end               = 0,
  n.body              = $source,
  n.source            = $source,
  n.repo              = $repo,
  n.model             = $model,
  n.provider          = $provider,
  n.start_time        = toInteger($start_time),
  n.input_tokens      = 0,
  n.cache_read_tokens = 0,
  n.cache_write_tokens= 0,
  n.output_tokens     = 0,
  n.total_tokens      = 0,
  n.duration_ms       = 0,
  n.status            = 'success',
  n.error_message     = ''
SET
  n.parent_session_id  = $parent_session_id,
  n.end_time           = toInteger($end_time),
  n.repo               = $repo,
  n.input_tokens       = toInteger($input_tokens),
  n.cache_read_tokens  = toInteger($cache_read_tokens),
  n.cache_write_tokens = toInteger($cache_write_tokens),
  n.output_tokens      = toInteger($output_tokens),
  n.total_tokens       = toInteger($total_tokens),
  n.duration_ms        = toInteger($duration_ms),
  n.status             = $status,
  n.error_message      = $error_message
RETURN n.node_key AS id
`;

async function upsertSession(spec: SessionSpec): Promise<void> {
  const s = (db as any).resilientSession();
  try {
    await s.run(UPSERT, {
      ...spec,
      ts: Date.now() / 1000,
    });
  } finally {
    await s.close();
  }
}

// ── main ────────────────────────────────────────────────────────────────────

async function main() {
  console.log("[seed-sessions] Starting …");

  const sessions: SessionSpec[] = [];

  // ── 1. Regular sessions — varied repos/sources/days (220 rows) ───────────
  for (let i = 0; i < 220; i++) {
    sessions.push(makeSession());
  }

  // ── 2. Parent + child sessions (5 families × 3 children = 20 nodes) ──────
  for (let p = 0; p < 5; p++) {
    const parentId = `seed-parent-${randomHex(8)}`;
    const parentTs = tsForDay(randomInt(1, 30));
    sessions.push(
      makeSession({
        session_id: parentId,
        source: "repo_agent",
        start_time: parentTs,
      }),
    );
    for (let c = 0; c < 3; c++) {
      const childId = `${parentId}-sub-${randomHex(8)}`;
      sessions.push(
        makeSession({
          session_id: childId,
          source: "graph_sub_agent",
          parent_session_id: parentId,
          start_time: parentTs + randomInt(1_000, 30_000),
        }),
      );
    }
  }

  // ── 3. Ghost rows — should be excluded from paginated results ─────────────
  //    ghost = source='unknown', total_tokens=0, duration_ms=0
  for (let g = 0; g < 10; g++) {
    sessions.push(
      makeSession({
        session_id: `seed-ghost-${randomHex(8)}`,
        source: "unknown",
        input_tokens: 0,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        output_tokens: 0,
        total_tokens: 0,
        duration_ms: 0,
        status: "success",
        error_message: "",
      }),
    );
  }

  // ── 4. Error sessions ──────────────────────────────────────────────────────
  for (let e = 0; e < 5; e++) {
    sessions.push(makeSession({ status: "error" }));
  }

  console.log(`[seed-sessions] Upserting ${sessions.length} sessions …`);

  let done = 0;
  for (const spec of sessions) {
    try {
      await upsertSession(spec);
      done++;
      if (done % 50 === 0) {
        console.log(`[seed-sessions] ${done}/${sessions.length} done`);
      }
    } catch (err) {
      console.error(`[seed-sessions] Failed to upsert ${spec.session_id}:`, err);
    }
  }

  console.log(`[seed-sessions] Done — ${done}/${sessions.length} sessions upserted.`);

  try {
    await db.close();
  } catch {
    // ignore
  }
}

main().catch((err) => {
  console.error("[seed-sessions] Fatal:", err);
  process.exit(1);
});
