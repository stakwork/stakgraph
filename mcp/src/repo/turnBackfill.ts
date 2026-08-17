import { existsSync, readFileSync, writeFileSync } from "fs";
import { db } from "../graph/neo4j.js";
import {
  listSessionFiles,
  loadSession,
  loadSessionConfig,
  sessionsDirFile,
} from "./session.js";
import {
  turnsFromTranscript,
  hasTurnCursor,
  markTranscriptEmitted,
  type EmittedTurn,
} from "./turns.js";

/**
 * One-off backfill of Turn chains for sessions that ran before live emission.
 *
 * The live emitter is a deterministic function of the persisted transcript,
 * so any session whose JSONL still exists can get the exact chain it would
 * have produced — same node_keys, same NEXT/HAS_TURN/READ_CONCEPT edges. Two
 * deliberate limits keep it honest:
 *
 *  - No per-turn timestamps. The transcript records no per-message time, so
 *    backfilled turns simply lack the property rather than carrying a fake.
 *  - No orphan chains. UPSERT_TURNS_QUERY anchors on the AgentSession node;
 *    a session that never reached appendSessionEnd has no node, writes zero
 *    turns, and is skipped — not written unanchored (the whole point of the
 *    live-emission work was ending orphan chains).
 *
 * Runs from the startup task block, mirroring backfillConceptReads: a marker
 * file records the mtime high-water, sessions with a turn cursor sidecar
 * (live emission already ran) are never touched, and one bad session stalls
 * the marker rather than sinking the sweep.
 */

const MARKER = ".turn-backfill.json";
/** How far back to look; matches the session retention window. */
const DEFAULT_WINDOW_MS = 30 * 24 * 60 * 60 * 1000;
/**
 * Sessions written this recently are skipped: they are (or are about to be)
 * live, and racing the live emitter's first sidecar write could double-emit
 * order 0. They age past the guard and are swept on a later boot — by which
 * point a live run would have left a cursor and excluded them properly.
 */
const RECENT_GUARD_MS = 10 * 60 * 1000;
/** UNWIND batch size; chains longer than this span multiple idempotent calls. */
const CHUNK = 500;

interface Marker {
  completed_at: string;
  /** Sessions last written at or before this are done; newer ones aren't. */
  through_mtime: number;
}

function readMarker(): Marker | null {
  try {
    const filePath = sessionsDirFile(MARKER);
    if (!existsSync(filePath)) return null;
    return JSON.parse(readFileSync(filePath, "utf-8")) as Marker;
  } catch {
    return null;
  }
}

function writeMarker(through_mtime: number): void {
  try {
    const marker: Marker = {
      completed_at: new Date().toISOString(),
      through_mtime,
    };
    writeFileSync(sessionsDirFile(MARKER), JSON.stringify(marker, null, 2));
  } catch (e) {
    console.error("[turn-backfill] could not write marker:", e);
  }
}

/**
 * The agent label for a session's turn_ids, matching what the live path
 * would have used: the caller-assigned agentName when the config sidecar
 * recorded one, else the recorded source, the literal "graph_sub_agent" for
 * child sessions (which never get a config sidecar), and "agent" as the
 * last resort — the same precedence the live emitter applies.
 */
export function backfillAgentLabel(sessionId: string): string {
  const config = loadSessionConfig(sessionId);
  if (config?.agentName) return config.agentName;
  if (config?.source) return config.source;
  if (sessionId.includes("-sub-")) return "graph_sub_agent";
  return "agent";
}

async function writeChain(
  sessionId: string,
  turns: EmittedTurn[],
): Promise<number> {
  let written = 0;
  for (let i = 0; i < turns.length; i += CHUNK) {
    const chunk = turns.slice(i, i + CHUNK);
    const batch = chunk.map(({ concepts: _c, ...t }) => t);
    const n = await db!.upsert_turns(sessionId, batch);
    // Anchor missing (no AgentSession node): the query writes nothing, and
    // neither should the remaining chunks.
    if (n === 0) return written;
    written += n;
    const links = chunk.flatMap((t) =>
      t.concepts.map((c) => ({
        turn_node_key: t.node_key,
        ref_id: c.ref_id,
        id: c.id,
      })),
    );
    if (links.length > 0) await db!.upsert_turn_concept_edges(links);
  }
  return written;
}

export async function backfillTurns(
  windowMs: number = DEFAULT_WINDOW_MS,
): Promise<{ scanned: number; sessions: number; turns: number }> {
  if (!db) return { scanned: 0, sessions: 0, turns: 0 };

  const marker = readMarker();
  const floor = Math.max(marker?.through_mtime ?? 0, Date.now() - windowMs);
  const ceiling = Date.now() - RECENT_GUARD_MS;
  // Oldest first, so the high-water mark can stop at the first session that
  // failed and everything after it is retried on the next boot.
  const sessions = listSessionFiles()
    .filter((s) => s.mtimeMs > floor && s.mtimeMs <= ceiling)
    .sort((a, b) => a.mtimeMs - b.mtimeMs);
  if (sessions.length === 0) {
    return { scanned: 0, sessions: 0, turns: 0 };
  }

  console.log(`[turn-backfill] scanning ${sessions.length} session(s)`);
  let backfilled = 0;
  let turnCount = 0;
  let highWater = floor;
  let stalled = false;
  const advance = (mtimeMs: number) => {
    if (!stalled) highWater = mtimeMs;
  };

  for (const session of sessions) {
    try {
      // A cursor sidecar means live emission already owns this chain.
      if (hasTurnCursor(session.sessionId)) {
        advance(session.mtimeMs);
        continue;
      }
      const agent = backfillAgentLabel(session.sessionId);
      const turns = turnsFromTranscript(
        session.sessionId,
        agent,
        loadSession(session.sessionId),
      );
      if (turns.length === 0) {
        advance(session.mtimeMs);
        continue;
      }

      const written = await writeChain(session.sessionId, turns);
      if (written > 0) {
        // Cursor makes a later live run continue this chain, and makes this
        // sweep (and any future one) skip the session.
        markTranscriptEmitted(session.sessionId, agent, turns.length);
        backfilled++;
        turnCount += written;
      }
      // written === 0: no AgentSession node to anchor on — deliberately not
      // an error, and no cursor either, so a node appearing later (a resumed
      // run creating the stub) lets a rerun sweep pick the session up.
      advance(session.mtimeMs);
    } catch (e) {
      // One bad session must not sink the sweep, but the marker must not
      // move past it either — a Neo4j blip mid-sweep would otherwise mark
      // every remaining session done without ever having written it.
      stalled = true;
      console.error(`[turn-backfill] session ${session.sessionId} failed:`, e);
    }
  }

  writeMarker(highWater);
  console.log(
    `[turn-backfill] backfilled ${backfilled} session(s), ${turnCount} turn(s)`,
  );
  return { scanned: sessions.length, sessions: backfilled, turns: turnCount };
}
