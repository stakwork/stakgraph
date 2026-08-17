import { ModelMessage } from "ai";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
  unlinkSync,
} from "fs";
import path from "path";
import { db } from "../graph/neo4j.js";
import { conceptReadsFrom } from "./concepts.js";

// ── Live Turn emission ───────────────────────────────────────────────
// Mirrors each agent step into the graph as it happens:
//
//   (AgentSession)-[:HAS_TURN]->(Turn)-[:NEXT]->(Turn)-[:NEXT]->...
//
// with one Turn per message content part (user_input / reasoning /
// tool_call / tool_result, and the run's final reasoning retyped to
// response at session end) — the same granularity and node shape the
// post-hoc build_trace_edges workflow produces, so live chains and
// backfilled chains are indistinguishable. tool_result Turns that read a
// gitree Concept additionally get a (Turn)-[:READ_CONCEPT]->(Concept)
// edge, the per-moment counterpart of the session-level ranked rollup.
//
// The graph is an index, never the source of truth: the session JSONL
// remains authoritative, every write is fire-and-forget, and any failure
// is logged and absorbed. Writes for one session are serialized on a
// promise chain so NEXT links are asserted in order; node_keys are
// deterministic, so a lost batch self-heals when the next one MERGEs its
// predecessor by key.
//
// Deliberate deviations from the backfill script, all data improvements:
// system messages are not emitted (the workflow's traces surface them as
// unusable turn_type 'unknown'), tool_result turns carry the real tool
// name instead of 'unknown', continuation-nudge user messages are
// skipped, and empty content parts (reasoning/thinking blocks, blank
// text) produce no turn.

const SESSIONS_DIR = process.env.SESSIONS_DIR || ".sessions";
const TOOL_RESULT_MAX_CHARS = 100;

interface TurnStateFile {
  /** Label prefixed to turn_ids; fixed at first emission so ids stay stable. */
  agent: string;
  next_order: number;
  /** Order of the most recent 'reasoning' turn — the finalize target. */
  last_reasoning_order: number | null;
}

interface TurnState extends TurnStateFile {
  /** toolCallId -> input, for concept detection on later tool_result parts. */
  pendingToolInputs: Map<string, unknown>;
  /** Per-session write serialization so NEXT links land in order. */
  queue: Promise<void>;
}

export interface EmittedTurn {
  node_key: string;
  prev_node_key: string | null;
  turn_id: string;
  turn_type: string;
  order: number;
  content: string;
  tool: string | null;
  /** Pairs tool_call and tool_result turns even when calls run in parallel. */
  tool_call_id: string | null;
  /**
   * Epoch ms at emission — when this moment of the session happened. Null for
   * backfilled turns (the transcript records no per-message time); the query
   * then leaves the property off entirely, which is more honest than a fake.
   */
  timestamp: number | null;
  concepts: Array<{ ref_id: string | null; id: string | null }>;
}

const states = new Map<string, TurnState>();

function stateFilePath(sessionId: string): string {
  const sessionDir = path.isAbsolute(SESSIONS_DIR)
    ? SESSIONS_DIR
    : path.join(process.cwd(), SESSIONS_DIR);
  if (!existsSync(sessionDir)) {
    mkdirSync(sessionDir, { recursive: true });
  }
  return path.join(sessionDir, `${sessionId}.turns.json`);
}

/**
 * Load or create the emitter state for a session. The sidecar file is what
 * lets a resumed session (same id, new process) continue its chain instead
 * of starting a colliding one at order 0.
 */
function getState(sessionId: string, agentHint?: string): TurnState {
  const cached = states.get(sessionId);
  if (cached) return cached;

  let file: TurnStateFile | null = null;
  const filePath = stateFilePath(sessionId);
  if (existsSync(filePath)) {
    try {
      file = JSON.parse(readFileSync(filePath, "utf-8")) as TurnStateFile;
    } catch {
      file = null;
    }
  }
  const state: TurnState = {
    agent: file?.agent || agentHint || "agent",
    next_order: file?.next_order ?? 0,
    last_reasoning_order: file?.last_reasoning_order ?? null,
    pendingToolInputs: new Map(),
    queue: Promise.resolve(),
  };
  states.set(sessionId, state);
  return state;
}

function persistState(sessionId: string, state: TurnState): void {
  try {
    const file: TurnStateFile = {
      agent: state.agent,
      next_order: state.next_order,
      last_reasoning_order: state.last_reasoning_order,
    };
    writeFileSync(stateFilePath(sessionId), JSON.stringify(file));
  } catch (e) {
    console.error(`[turns] could not persist state for ${sessionId}:`, e);
  }
}

/** node_key scheme of the backfill pipeline: 'turn-' + sanitized turn_id. */
export function turnNodeKey(turnId: string): string {
  return "turn-" + turnId.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function turnId(agent: string, sessionId: string, order: number): string {
  return `${agent}-${sessionId}-turn-${order}`;
}

/**
 * Stringify a tool result the way it appears in persisted session messages
 * (the `{type, value}` output wrapper included — that is the shape the
 * backfill workflow stored), truncated like build_trace_edges.py.
 */
function toolResultContent(output: unknown): string {
  let wrapped: unknown = output;
  if (
    !(
      output &&
      typeof output === "object" &&
      "type" in (output as any) &&
      "value" in (output as any)
    )
  ) {
    wrapped =
      typeof output === "string"
        ? { type: "text", value: output }
        : { type: "json", value: output };
  }
  let text: string;
  try {
    text = JSON.stringify(wrapped) ?? "";
  } catch {
    return "";
  }
  if (text.length > TOOL_RESULT_MAX_CHARS) {
    return text.slice(0, TOOL_RESULT_MAX_CHARS) + "...";
  }
  return text;
}

/** Unwrap a `{type, value}` output envelope for concept detection. */
function toolResultValue(output: unknown): unknown {
  if (
    output &&
    typeof output === "object" &&
    "value" in (output as any) &&
    typeof (output as any).type === "string"
  ) {
    return (output as any).value;
  }
  return output;
}

function buildTurn(
  sessionId: string,
  state: TurnState,
  turn_type: string,
  content: string,
  tool: string | null,
  concepts: Array<{ ref_id: string | null; id: string | null }> = [],
  toolCallId?: string,
): EmittedTurn {
  const order = state.next_order++;
  const id = turnId(state.agent, sessionId, order);
  if (turn_type === "reasoning") state.last_reasoning_order = order;
  return {
    node_key: turnNodeKey(id),
    prev_node_key:
      order === 0 ? null : turnNodeKey(turnId(state.agent, sessionId, order - 1)),
    turn_id: id,
    turn_type,
    order,
    content,
    tool,
    tool_call_id: toolCallId ?? null,
    timestamp: Date.now(),
    concepts,
  };
}

/** Queue a fire-and-forget graph write; failures are logged, never thrown. */
function scheduleWrite(
  sessionId: string,
  state: TurnState,
  turns: EmittedTurn[],
): void {
  if (!db || turns.length === 0) return;
  const links = turns.flatMap((t) =>
    t.concepts.map((c) => ({
      turn_node_key: t.node_key,
      ref_id: c.ref_id,
      id: c.id,
    })),
  );
  const batch = turns.map(({ concepts: _concepts, ...t }) => t);
  state.queue = state.queue
    .then(() => db!.upsert_turns(sessionId, batch))
    .then(() =>
      links.length > 0 ? db!.upsert_turn_concept_edges(links) : undefined,
    )
    .catch((e) => console.error("[turns] Neo4j turn write failed:", e));
}

/**
 * Emit the user_input turn(s) for a run's prompt. Call once per run, before
 * the tool loop starts. `agent` labels the turn_ids (typically the session
 * source); it only takes effect on the session's first-ever emission.
 * Returns the emitted turns (empty on skip or failure).
 */
export function emitUserTurn(
  sessionId: string,
  agent: string | undefined,
  message: ModelMessage,
): EmittedTurn[] {
  try {
    const state = getState(sessionId, agent);
    const parts =
      typeof message.content === "string"
        ? [{ type: "text", text: message.content }]
        : (message.content as Array<any>);
    const turns: EmittedTurn[] = [];
    for (const part of parts) {
      if (part?.type !== "text" || !part.text) continue;
      turns.push(buildTurn(sessionId, state, "user_input", part.text, null));
    }
    if (turns.length === 0) return [];
    persistState(sessionId, state);
    scheduleWrite(sessionId, state, turns);
    return turns;
  } catch (e) {
    console.error("[turns] emitUserTurn failed:", e);
    return [];
  }
}

/**
 * Emit the turns for one completed agent step. `content` is the step's
 * content-part array (text / tool-call / tool-result). Parts are emitted
 * assistant-parts-first, tool-results after, matching the message layout
 * the transcript persists (assistant message, then tool message).
 * Returns the emitted turns (empty on skip or failure).
 */
export function emitStepTurns(
  sessionId: string,
  agent: string | undefined,
  content: unknown,
): EmittedTurn[] {
  try {
    if (!Array.isArray(content)) return [];
    const state = getState(sessionId, agent);
    const assistantParts = content.filter((p: any) => p?.type !== "tool-result");
    const resultParts = content.filter((p: any) => p?.type === "tool-result");
    const turns: EmittedTurn[] = [];

    for (const part of assistantParts) {
      if (part?.type === "text" && part.text) {
        turns.push(buildTurn(sessionId, state, "reasoning", part.text, null));
      } else if (part?.type === "tool-call") {
        if (part.toolCallId) {
          state.pendingToolInputs.set(part.toolCallId, part.input);
        }
        let inputJson: string;
        try {
          inputJson = JSON.stringify(part.input ?? "") ?? '""';
        } catch {
          inputJson = '""';
        }
        turns.push(
          buildTurn(
            sessionId,
            state,
            "tool_call",
            inputJson,
            part.toolName ?? "unknown",
            [],
            part.toolCallId,
          ),
        );
      }
    }

    for (const part of resultParts) {
      const toolName = part.toolName ?? "unknown";
      const input = part.toolCallId
        ? state.pendingToolInputs.get(part.toolCallId)
        : undefined;
      if (part.toolCallId) state.pendingToolInputs.delete(part.toolCallId);
      let concepts: Array<{ ref_id: string | null; id: string | null }> = [];
      try {
        concepts = conceptReadsFrom(toolName, input, toolResultValue(part.output))
          .filter((r) => r.ref_id || r.id)
          .map((r) => ({ ref_id: r.ref_id ?? null, id: r.id ?? null }));
      } catch {
        // concept detection is best-effort; the turn itself still lands
      }
      turns.push(
        buildTurn(
          sessionId,
          state,
          "tool_result",
          toolResultContent(part.output),
          toolName,
          concepts,
          part.toolCallId,
        ),
      );
    }

    if (turns.length === 0) return [];
    persistState(sessionId, state);
    scheduleWrite(sessionId, state, turns);
    return turns;
  } catch (e) {
    console.error("[turns] emitStepTurns failed:", e);
    return [];
  }
}

/**
 * Retype the run's final reasoning turn to 'response', like the backfill
 * workflow does for the last assistant text turn. Called from session end;
 * fire-and-forget. Clears the tracked order so a resumed session finalizes
 * its own last turn rather than re-targeting this one — meaning a
 * multi-run session carries one 'response' per run, each run's actual
 * final answer. Returns the retyped node_key, or null when there was no
 * reasoning turn to finalize.
 */
export function finalizeTurns(sessionId: string): string | null {
  try {
    const state = getState(sessionId);
    if (state.last_reasoning_order === null) return null;
    const nodeKey = turnNodeKey(
      turnId(state.agent, sessionId, state.last_reasoning_order),
    );
    state.last_reasoning_order = null;
    persistState(sessionId, state);
    if (db) {
      state.queue = state.queue
        .then(() => db!.finalize_turn_response(nodeKey))
        .catch((e) => console.error("[turns] finalize failed:", e));
    }
    return nodeKey;
  } catch (e) {
    console.error("[turns] finalizeTurns failed:", e);
    return null;
  }
}

// ── Transcript classification (backfill) ────────────────────────────
// The live emitters above classify step content as it happens; this is the
// same classification applied to a PERSISTED transcript, for sessions that
// ran before live emission existed. Pure: no state map, no sidecar, no
// writes — the caller owns persistence. Matches build_trace_edges.py where
// it matters (one turn per content part, last reasoning retyped to
// response) and the live emitter where they deliberately deviate (system
// messages and continuation nudges skipped, real tool names kept).

/** True for the nudge user messages the live path never emits. */
function isNudgeMessage(message: ModelMessage): boolean {
  const tag = (message as any).providerOptions?.stakgraph;
  return Boolean(tag?.continuationNudge || tag?.timeNudge);
}

/**
 * Classify a full stored transcript into the Turn chain it would have
 * produced had live emission been running. Timestamps are null — the
 * transcript records no per-message time.
 */
export function turnsFromTranscript(
  sessionId: string,
  agent: string,
  messages: ModelMessage[],
): EmittedTurn[] {
  const turns: EmittedTurn[] = [];
  const pendingInputs = new Map<string, unknown>();
  let lastReasoningIdx = -1;

  const push = (
    turn_type: string,
    content: string,
    tool: string | null,
    toolCallId?: string,
    concepts: Array<{ ref_id: string | null; id: string | null }> = [],
  ) => {
    const order = turns.length;
    const id = turnId(agent, sessionId, order);
    if (turn_type === "reasoning") lastReasoningIdx = order;
    turns.push({
      node_key: turnNodeKey(id),
      prev_node_key:
        order === 0 ? null : turnNodeKey(turnId(agent, sessionId, order - 1)),
      turn_id: id,
      turn_type,
      order,
      content,
      tool,
      tool_call_id: toolCallId ?? null,
      timestamp: null,
      concepts,
    });
  };

  for (const message of messages) {
    if (message.role === "system") continue;
    if (message.role === "user" && isNudgeMessage(message)) continue;
    const parts =
      typeof message.content === "string"
        ? [{ type: "text", text: message.content }]
        : (message.content as Array<any>);
    if (!Array.isArray(parts)) continue;

    for (const part of parts) {
      if (message.role === "user") {
        if (part?.type === "text" && part.text) {
          push("user_input", part.text, null);
        }
      } else if (message.role === "assistant") {
        if (part?.type === "text" && part.text) {
          push("reasoning", part.text, null);
        } else if (part?.type === "tool-call") {
          if (part.toolCallId) pendingInputs.set(part.toolCallId, part.input);
          let inputJson: string;
          try {
            inputJson = JSON.stringify(part.input ?? "") ?? '""';
          } catch {
            inputJson = '""';
          }
          push("tool_call", inputJson, part.toolName ?? "unknown", part.toolCallId);
        }
      } else if (message.role === "tool") {
        if (part?.type !== "tool-result") continue;
        const toolName = part.toolName ?? "unknown";
        const input = part.toolCallId ? pendingInputs.get(part.toolCallId) : undefined;
        if (part.toolCallId) pendingInputs.delete(part.toolCallId);
        let concepts: Array<{ ref_id: string | null; id: string | null }> = [];
        try {
          concepts = conceptReadsFrom(toolName, input, toolResultValue(part.output))
            .filter((r) => r.ref_id || r.id)
            .map((r) => ({ ref_id: r.ref_id ?? null, id: r.id ?? null }));
        } catch {
          // best-effort, like the live path
        }
        push(
          "tool_result",
          toolResultContent(part.output),
          toolName,
          part.toolCallId,
          concepts,
        );
      }
    }
  }

  // The whole transcript is in hand, so the retype is a pre-write edit
  // rather than a finalize query: last reasoning turn becomes the response,
  // exactly like build_trace_edges.py.
  if (lastReasoningIdx >= 0) turns[lastReasoningIdx].turn_type = "response";

  return turns;
}

/** True when a session already has a turn cursor (live emission ran). */
export function hasTurnCursor(sessionId: string): boolean {
  return states.has(sessionId) || existsSync(stateFilePath(sessionId));
}

/**
 * Record that a session's transcript has been emitted through order
 * `nextOrder`. Writes the same cursor sidecar the live path maintains, so a
 * later live run on this session continues the chain instead of colliding —
 * and so the backfill skips it on any future sweep. last_reasoning_order is
 * null because turnsFromTranscript already retyped the response pre-write.
 */
export function markTranscriptEmitted(
  sessionId: string,
  agent: string,
  nextOrder: number,
): void {
  const state = getState(sessionId, agent);
  state.agent = state.agent || agent;
  state.next_order = Math.max(state.next_order, nextOrder);
  state.last_reasoning_order = null;
  persistState(sessionId, state);
}

/** Drop the sidecar + in-memory state; called when a session is deleted. */
export function deleteTurnState(sessionId: string): void {
  states.delete(sessionId);
  try {
    const filePath = stateFilePath(sessionId);
    if (existsSync(filePath)) unlinkSync(filePath);
  } catch {
    // best-effort cleanup
  }
}
