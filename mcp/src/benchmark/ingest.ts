/**
 * Session ingest — the write half of the sessions API.
 *
 * Lets an agent running in ANOTHER process (hive's ai-sdk agents) record the
 * same live Turn chain the in-process emitter writes:
 *
 *   POST /api/sessions              stub the AgentSession (status 'running')
 *   POST /api/sessions/:id/turns    append a batch of Turns as they happen
 *   POST /api/sessions/:id/end      totals, status, final response retype
 *   POST /api/sessions/:id/concepts session-level READ_CONCEPT rollup
 *
 * The graph is the only state: `order` comes from the session's chain head
 * (GET_TURN_CHAIN_HEAD_QUERY) unless the caller pins it, and the agent label
 * is recovered from the head's turn_id. So there is no server-side session
 * state to lose, a crashed caller resumes by just posting again, and every
 * write stays idempotent on the deterministic node_key.
 *
 * Turn writes anchor on the AgentSession node (see UPSERT_TURNS_QUERY), so a
 * batch for an unknown session writes nothing — reported here as a 404 rather
 * than a silent no-op, since that is a caller sequencing bug.
 */

import { Request, Response } from "express";
import { db } from "../graph/neo4j.js";
import { toNum } from "../graph/types.js";
import {
  buildExternalTurns,
  agentFromTurnId,
  EXTERNAL_TURN_TYPES,
  type ExternalTurnInput,
} from "../repo/turns.js";
import { getProviderForModel } from "../aieo/src/provider.js";

/** Turn-id label used when the caller names none and the chain is empty. */
const DEFAULT_AGENT = "agent";
const MAX_TURNS_PER_BATCH = 500;
const MAX_CONCEPTS_PER_CALL = 1000;
const VALID_TURN_TYPES = new Set<string>(EXTERNAL_TURN_TYPES);
const VALID_STATUSES = new Set(["success", "error", "aborted"]);

/**
 * Per-session write serialization. Two batches for one session must not read
 * the same chain head and both number themselves from it; within a process
 * this queue prevents that. Cross-process concurrency for a single session is
 * the caller's problem — post one session's turns from one place, or pin
 * `start_order` yourself.
 */
const queues = new Map<string, Promise<unknown>>();

function serialize<T>(id: string, fn: () => Promise<T>): Promise<T> {
  const prev = queues.get(id) ?? Promise.resolve();
  const run = prev.then(fn);
  // The stored tail never rejects, so one failed batch doesn't poison the
  // next; it is deleted once it IS the tail, so the map can't grow forever.
  const tail = run.then(
    () => undefined,
    () => undefined,
  );
  queues.set(id, tail);
  void tail.then(() => {
    if (queues.get(id) === tail) queues.delete(id);
  });
  return run;
}

/** Session ids become Neo4j node_keys and (elsewhere) file paths. */
function invalidId(id: string): boolean {
  return !id || id.length > 256 || id.includes("..") || id.includes("/");
}

function str(v: unknown, max = 256): string {
  return v === undefined || v === null ? "" : String(v).slice(0, max);
}

function int(v: unknown): number | null {
  const n = Number(v);
  return Number.isFinite(n) ? Math.trunc(n) : null;
}

/**
 * Guard the graph handle. Called AFTER request validation in every handler:
 * a malformed request is a 400 whether or not the graph happens to be up.
 */
function requireDb(res: Response): boolean {
  if (db) return true;
  res.status(503).json({ error: "Graph unavailable" });
  return false;
}

/**
 * POST /api/sessions — create (or refresh) the AgentSession node for an
 * external run. Idempotent: re-posting the same id keeps the original node.
 */
export async function create_session(req: Request, res: Response) {
  const body = (req.body ?? {}) as Record<string, unknown>;
  const session_id = str(body.session_id);
  if (invalidId(session_id)) {
    res.status(400).json({ error: "Invalid session_id" });
    return;
  }
  if (!requireDb(res)) return;
  const start_time = int(body.start_time) ?? Date.now();
  try {
    await db!.create_agent_session_stub({
      session_id,
      parent_session_id: str(body.parent_session_id),
      // 'external' rather than 'unknown': the list endpoint hides unknown
      // zero-token sessions as junk.
      source: str(body.source) || "external",
      repo: str(body.repo, 512),
      agent_name: str(body.agent_name, 128),
      spawn_tool_call_id: str(body.spawn_tool_call_id),
      start_time,
    });
  } catch (e) {
    console.error("[ingest] create_session failed:", e);
    res.status(500).json({ error: "Failed to create session" });
    return;
  }
  res.status(201).json({ session_id, status: "running", start_time });
}

/**
 * POST /api/sessions/:id/turns — append a batch of Turns to the chain.
 * Turns are numbered from the graph's chain head unless `start_order` pins
 * them; re-posting the same orders overwrites those turns in place.
 */
export async function append_turns(req: Request, res: Response) {
  const id = String(req.params.id);
  if (invalidId(id)) {
    res.status(400).json({ error: "Invalid session id" });
    return;
  }
  const body = (req.body ?? {}) as Record<string, unknown>;
  const raw = body.turns;
  if (!Array.isArray(raw) || raw.length === 0) {
    res.status(400).json({ error: "turns must be a non-empty array" });
    return;
  }
  if (raw.length > MAX_TURNS_PER_BATCH) {
    res
      .status(400)
      .json({ error: `turns exceeds max batch size of ${MAX_TURNS_PER_BATCH}` });
    return;
  }
  const parts: ExternalTurnInput[] = [];
  for (const [i, t] of raw.entries()) {
    const turn_type = str((t as any)?.turn_type, 32);
    if (!VALID_TURN_TYPES.has(turn_type)) {
      res.status(400).json({
        error: `turns[${i}].turn_type must be one of: ${[...VALID_TURN_TYPES].join(", ")}`,
      });
      return;
    }
    parts.push({
      turn_type,
      content: (t as any)?.content,
      tool: str((t as any)?.tool, 128) || null,
      tool_call_id: str((t as any)?.tool_call_id) || null,
      timestamp: int((t as any)?.timestamp),
      concepts: Array.isArray((t as any)?.concepts) ? (t as any).concepts : [],
    });
  }
  const pinnedOrder = int(body.start_order);
  if (pinnedOrder !== null && pinnedOrder < 0) {
    res.status(400).json({ error: "start_order must be >= 0" });
    return;
  }
  const requestedAgent = str(body.agent, 64) || null;
  if (!requireDb(res)) return;

  try {
    const result = await serialize(id, async () => {
      let start = pinnedOrder;
      let agent = requestedAgent;
      if (start === null || !agent) {
        const head = await db!.get_turn_chain_head(id);
        if (start === null) start = head ? head.max_order + 1 : 0;
        if (!agent) {
          agent =
            (head && agentFromTurnId(head.turn_id, id)) || DEFAULT_AGENT;
        }
      }
      const turns = buildExternalTurns(id, agent, start, parts);
      const written = await db!.upsert_turns(
        id,
        turns.map(({ concepts: _c, ...t }) => t),
      );
      // Zero written means the anchor MATCH found no AgentSession node.
      if (written === 0) return null;
      const links = turns.flatMap((t) =>
        t.concepts.map((c) => ({
          turn_node_key: t.node_key,
          ref_id: c.ref_id,
          id: c.id,
        })),
      );
      if (links.length > 0) await db!.upsert_turn_concept_edges(links);
      return { turns, written };
    });

    if (!result) {
      res.status(404).json({
        error: `Unknown session '${id}' — POST /api/sessions to create it first`,
      });
      return;
    }
    const last = result.turns[result.turns.length - 1];
    res.status(201).json({
      session_id: id,
      written: result.written,
      next_order: last.order + 1,
      turns: result.turns.map((t) => ({
        order: t.order,
        turn_id: t.turn_id,
        node_key: t.node_key,
        turn_type: t.turn_type,
      })),
    });
  } catch (e) {
    console.error("[ingest] append_turns failed:", e);
    res.status(500).json({ error: "Failed to write turns" });
  }
}

/**
 * POST /api/sessions/:id/end — finalize the run: totals, status, and the
 * last reasoning turn retyped to 'response'.
 *
 * Token counts ACCUMULATE on the session node (same query the in-process
 * path uses), so call this once per run — calling it twice double-counts.
 */
export async function end_session(req: Request, res: Response) {
  const id = String(req.params.id);
  if (invalidId(id)) {
    res.status(400).json({ error: "Invalid session id" });
    return;
  }
  const body = (req.body ?? {}) as Record<string, unknown>;
  const status = str(body.status, 16) || "success";
  if (!VALID_STATUSES.has(status)) {
    res
      .status(400)
      .json({ error: `status must be one of: ${[...VALID_STATUSES].join(", ")}` });
    return;
  }
  if (!requireDb(res)) return;

  try {
    // Read the node first: the upsert SETs repo unconditionally, so ending a
    // session without echoing it back would wipe what /sessions recorded.
    const node = await db!.get_agent_session(id);
    if (!node) {
      res.status(404).json({
        error: `Unknown session '${id}' — POST /api/sessions to create it first`,
      });
      return;
    }
    const end_time = int(body.end_time) ?? Date.now();
    const start_time = toNum(node.start_time) || end_time;
    const usage = (body.usage ?? {}) as Record<string, unknown>;
    const input_tokens = int(usage.input_tokens) ?? 0;
    const output_tokens = int(usage.output_tokens) ?? 0;
    const cache_read_tokens = int(usage.cache_read_tokens) ?? 0;
    const cache_write_tokens = int(usage.cache_write_tokens) ?? 0;
    const model = str(body.model, 128) || str(node.model, 128);
    await db!.upsert_agent_session({
      session_id: id,
      // Empty strings are preserved by the query's CASE guards — these keep
      // whatever the stub (or a SPAWNED parent) already set.
      parent_session_id: "",
      agent_name: "",
      spawn_tool_call_id: "",
      source: str(node.source) || "external",
      repo: str(body.repo, 512) || str(node.repo, 512),
      model,
      provider: str(body.provider, 64) || (model ? getProviderForModel(model) : ""),
      start_time,
      end_time,
      duration_ms: int(body.duration_ms) ?? Math.max(0, end_time - start_time),
      input_tokens,
      cache_read_tokens,
      cache_write_tokens,
      output_tokens,
      total_tokens:
        int(usage.total_tokens) ??
        input_tokens + output_tokens + cache_read_tokens + cache_write_tokens,
      status,
      error_message: str(body.error_message, 2000),
    });

    let finalized_turn: string | null = null;
    if (body.finalize_response !== false) {
      finalized_turn = await db!.finalize_last_reasoning_turn(id);
    }
    res.json({ session_id: id, status, end_time, finalized_turn });
  } catch (e) {
    console.error("[ingest] end_session failed:", e);
    res.status(500).json({ error: "Failed to end session" });
  }
}

/**
 * POST /api/sessions/:id/concepts — the session-level READ_CONCEPT rollup
 * (rank/evidence/contradicts), the external counterpart of the reflection
 * sidecar sync. Mirrors full state on every call: a null `rank` clears it.
 */
export async function link_session_concepts(req: Request, res: Response) {
  const id = String(req.params.id);
  if (invalidId(id)) {
    res.status(400).json({ error: "Invalid session id" });
    return;
  }
  const raw = (req.body ?? {}).concepts;
  if (!Array.isArray(raw)) {
    res.status(400).json({ error: "concepts must be an array" });
    return;
  }
  if (raw.length > MAX_CONCEPTS_PER_CALL) {
    res
      .status(400)
      .json({ error: `concepts exceeds max of ${MAX_CONCEPTS_PER_CALL}` });
    return;
  }
  const concepts = raw.map((c: any) => ({
    id: str(c?.id) || undefined,
    ref_id: str(c?.ref_id) || undefined,
    read_order: int(c?.read_order) ?? undefined,
    rank: int(c?.rank),
    evidence: str(c?.evidence, 2000) || undefined,
    contradicts: str(c?.contradicts, 2000) || undefined,
  }));
  if (!requireDb(res)) return;
  try {
    const linked = await db!.upsert_session_concept_edges(id, concepts);
    res.json({ session_id: id, linked, submitted: concepts.length });
  } catch (e) {
    console.error("[ingest] link_session_concepts failed:", e);
    res.status(500).json({ error: "Failed to link concepts" });
  }
}
