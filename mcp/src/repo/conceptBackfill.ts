import { ModelMessage } from "ai";
import { existsSync, readFileSync, writeFileSync } from "fs";
import { db } from "../graph/neo4j.js";
import { listConcepts } from "../gitree/service.js";
import {
  listSessionFiles,
  loadSession,
  loadSessionConfig,
  loadReflection,
  mergeReflection,
  sessionsDirFile,
} from "./session.js";
import { mergeConceptReads, type ConceptIdentity, type ConceptRead } from "./concepts.js";

/**
 * One-off backfill of the observed half of the concept record.
 *
 * Which Concepts a session read is recoverable from its stored transcript, so
 * sessions that ran before collection existed can be filled in after the fact.
 * Only the observed half — read set and read order. Ranking is deliberately
 * not backfilled: those transcripts are long cold, so a reflect pass would pay
 * full input price for a judgement made days after the fact.
 *
 * Runs from the startup task block. Safe to call on every boot: a marker file
 * records how far it got, and sessions that already have a reflection sidecar
 * are never touched, so live-collected data can't be overwritten.
 */

const MARKER = ".concept-backfill.json";
/** How far back to look. Sessions older than this are left alone. */
const DEFAULT_WINDOW_MS = 7 * 24 * 60 * 60 * 1000;

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
    console.error("[concept-backfill] could not write marker:", e);
  }
}

/** Tool calls whose INPUT names a concept the agent went on to read. */
const READ_TOOL_INPUTS: Record<string, "id" | "ref_id"> = {
  learn_concept: "id",
  graph_get: "ref_id",
};

/**
 * Pull concept reads out of a stored transcript, in call order.
 *
 * Reads the tool-call INPUTS rather than the results, because a session that
 * ran with `truncateToolResults` has a clipped `graph_get` body that may not
 * parse — inputs are never truncated. The cost is that the input alone can't
 * say whether a ref_id belongs to a Concept or a Function, so the caller must
 * intersect these against the catalog before recording any of them.
 */
export function conceptReadsFromTranscript(messages: ModelMessage[]): ConceptRead[] {
  const reads: ConceptRead[] = [];
  for (const message of messages) {
    if (message.role !== "assistant" || !Array.isArray(message.content)) continue;
    for (const part of message.content as any[]) {
      if (part?.type !== "tool-call") continue;
      const field = READ_TOOL_INPUTS[part.toolName];
      if (!field) continue;
      const value = part.input?.[field === "id" ? "concept_id" : "ref_id"];
      if (typeof value !== "string" || !value) continue;
      reads.push({ [field]: value, via: part.toolName } as ConceptRead);
    }
  }
  return reads;
}

/**
 * Keep only the reads the catalog confirms are Concepts, then normalize them.
 *
 * This is where backfill differs from live collection, and it's the step that
 * keeps it honest: a `graph_get` input alone can't say whether a ref_id names
 * a Concept or a Function, so anything the catalog doesn't know is dropped
 * rather than recorded. The live path has the node type in the tool result and
 * needs no catalog to be sure.
 *
 * The trade is that a concept deleted since the run is unrecoverable, so a
 * backfilled session can undercount. It never over-counts.
 */
export function confirmConceptReads(
  candidates: ConceptRead[],
  catalog: ConceptIdentity[],
): ConceptRead[] {
  const known = new Set<string>();
  for (const c of catalog) {
    if (c.id) known.add(c.id);
    if (c.ref_id) known.add(c.ref_id);
  }
  const confirmed = candidates.filter(
    (r) => (r.id && known.has(r.id)) || (r.ref_id && known.has(r.ref_id)),
  );
  return confirmed.length > 0 ? mergeConceptReads(confirmed, catalog) : [];
}

/** Catalog lookups are per-repo and reused across every session in the sweep. */
async function catalogFor(
  repo: string | undefined,
  cache: Map<string, ConceptIdentity[]>,
): Promise<ConceptIdentity[]> {
  const key = repo ?? "*";
  const cached = cache.get(key);
  if (cached) return cached;
  const { concepts } = await listConcepts(repo);
  cache.set(key, concepts);
  return concepts;
}

export async function backfillConceptReads(
  windowMs: number = DEFAULT_WINDOW_MS,
): Promise<{ scanned: number; written: number; concepts: number }> {
  const marker = readMarker();
  const floor = Math.max(marker?.through_mtime ?? 0, Date.now() - windowMs);
  // Oldest first, so the high-water mark can stop at the first session that
  // failed and everything after it is retried on the next boot.
  const sessions = listSessionFiles()
    .filter((s) => s.mtimeMs > floor)
    .sort((a, b) => a.mtimeMs - b.mtimeMs);
  if (sessions.length === 0) {
    return { scanned: 0, written: 0, concepts: 0 };
  }

  console.log(`[concept-backfill] scanning ${sessions.length} session(s)`);
  const catalogCache = new Map<string, ConceptIdentity[]>();
  let written = 0;
  let conceptCount = 0;
  let highWater = floor;
  let stalled = false;
  const advance = (mtimeMs: number) => {
    if (!stalled) highWater = mtimeMs;
  };

  for (const session of sessions) {
    try {
      // Never overwrite a sidecar the live path already produced.
      if (loadReflection(session.sessionId)) {
        advance(session.mtimeMs);
        continue;
      }
      const candidates = conceptReadsFromTranscript(loadSession(session.sessionId));
      if (candidates.length === 0) {
        advance(session.mtimeMs);
        continue;
      }

      const config = loadSessionConfig(session.sessionId);
      const repo = config?.repos?.length === 1 ? config.repos[0] : undefined;
      const catalog = await catalogFor(repo, catalogCache);

      const concepts = confirmConceptReads(candidates, catalog);
      if (concepts.length === 0) {
        advance(session.mtimeMs);
        continue;
      }

      mergeReflection(session.sessionId, {
        concepts: concepts.map((c) => ({
          id: c.id,
          ref_id: c.ref_id,
          repo: c.repo,
          name: c.name,
          rank: null,
        })),
      });
      written++;
      conceptCount += concepts.length;
      advance(session.mtimeMs);
    } catch (e) {
      // One bad session must not sink the sweep, but the marker must not move
      // past it either — a Neo4j blip mid-sweep would otherwise mark every
      // remaining session done without ever having read it. Keep going to
      // salvage what we can; leave the mark where it was.
      stalled = true;
      console.error(`[concept-backfill] session ${session.sessionId} failed:`, e);
    }
  }

  writeMarker(highWater);
  console.log(
    `[concept-backfill] wrote ${written} sidecar(s) covering ${conceptCount} concept read(s)`,
  );
  return { scanned: sessions.length, written, concepts: conceptCount };
}

const EDGE_MARKER = ".concept-edge-backfill.json";

/**
 * One-off mirror of existing reflection sidecars into READ_CONCEPT edges.
 *
 * Edge syncing shipped after reflection collection had already been running,
 * so sessions that reflected before it never got edges: the live sync fires
 * only from mergeReflection and appendSessionEnd, and backfillConceptReads
 * skips any session that already has a sidecar. This sweep reads each
 * existing sidecar and issues the same idempotent MERGEs the live path does —
 * no deletions, no model calls, and no data-quality downgrade (unlike
 * regenerating sidecars, which would lose the ranked half forever).
 *
 * Sequential on purpose, one query per session, so boot doesn't flood the
 * driver pool. A sweep with any db failure leaves the marker unwritten and
 * retries next boot; sessions whose AgentSession node is missing simply link
 * nothing (the Cypher MATCHes the session node) and are not failures.
 */
export async function backfillConceptEdges(): Promise<{
  scanned: number;
  synced: number;
  failed: number;
}> {
  if (!db || existsSync(sessionsDirFile(EDGE_MARKER))) {
    return { scanned: 0, synced: 0, failed: 0 };
  }

  const sessions = listSessionFiles();
  let synced = 0;
  let failed = 0;
  for (const session of sessions) {
    const reflection = loadReflection(session.sessionId);
    if (!reflection) continue;
    const linkable = reflection.concepts.filter((c) => c.ref_id || c.id);
    if (linkable.length === 0) continue;
    try {
      await db.upsert_session_concept_edges(session.sessionId, linkable);
      synced++;
    } catch (e) {
      failed++;
      console.error(
        `[concept-edge-backfill] session ${session.sessionId} failed:`,
        e,
      );
    }
  }

  if (failed === 0) {
    try {
      writeFileSync(
        sessionsDirFile(EDGE_MARKER),
        JSON.stringify(
          { completed_at: new Date().toISOString(), synced },
          null,
          2,
        ),
      );
    } catch (e) {
      console.error("[concept-edge-backfill] could not write marker:", e);
    }
  }
  if (synced > 0 || failed > 0) {
    console.log(
      `[concept-edge-backfill] synced ${synced} session(s), ${failed} failure(s)`,
    );
  }
  return { scanned: sessions.length, synced, failed };
}
