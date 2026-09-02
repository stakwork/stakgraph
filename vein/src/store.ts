import { mkdir, writeFile, appendFile, readdir, readFile, open } from "node:fs/promises";
import { join } from "node:path";
import type { RunEvent, RunSummary } from "./core.js";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** A run is terminal once its log records a `run.end`, `run.error`, or
 *  `run.cancelled` — though a later `run.resumed` REOPENS it (§5.2: a
 *  resumed run appends to the same log past its old terminal event). */
function isTerminal(event: RunEvent): boolean {
  return (
    event.type === "run.end" ||
    event.type === "run.error" ||
    event.type === "run.cancelled"
  );
}

/** A `run.resumed` marker reopens a log whose previous event was terminal. */
function reopensRun(event: RunEvent): boolean {
  return event.type === "run.resumed";
}

/**
 * Generic tail of an append-only JSONL file: yield every parsed line from the
 * start of the file (history), then follow appends (live) until `isTerminal`
 * returns true for a line, at which point the generator returns. This is the
 * shared engine behind `FileRunStore.tailEvents` (runs) and `FileChatStore`
 * (chat turns) — the background-job reattach model (EVAL_SPEC §8).
 *
 * The append-only log is the ordered source of truth, so the history→live
 * join is race-free: read from a byte offset to EOF, then keep re-reading
 * from the new offset. Partial trailing lines (a write caught mid-flush) are
 * buffered until their newline arrives. One code path serves completed *and*
 * in-flight producers — a completed file drains and returns immediately; a
 * live one polls (`intervalMs`) for appends. Pass an `AbortSignal` (e.g. on
 * client disconnect) to stop early. A file that doesn't exist yet is polled
 * until it appears (race-free with a producer that hasn't written line 1).
 */
export async function* tailJsonl<T>(
  file: string,
  isTerminal: (event: T) => boolean,
  opts: {
    intervalMs?: number;
    signal?: AbortSignal;
    /** A later event that REOPENS a log whose previous event was terminal
     *  (a resumed run's `run.resumed`, RUN_CONTROL_SPEC §5.2). When set, a
     *  terminal event doesn't end the tail immediately: the tail scans
     *  ahead for a reopening event, and only closes at EOF (or, if
     *  `stillLive` says the producer is live again, keeps following). */
    reopens?: (event: T) => boolean;
    /** Consulted at EOF after a terminal event when `reopens` is set: a live
     *  producer (a registered run controller) means a resume is in flight —
     *  keep following instead of closing. Default: close at EOF. */
    stillLive?: () => boolean;
  } = {},
): AsyncGenerator<T> {
  const intervalMs = opts.intervalMs ?? 250;
  const signal = opts.signal;
  let offset = 0;
  let leftover = "";
  // Deferred-close mode (opts.reopens set): saw a terminal event, close at
  // EOF unless a reopening event arrives first.
  let sawTerminal = false;

  while (true) {
    if (signal?.aborted) return;

    let chunk = "";
    try {
      const fh = await open(file, "r");
      try {
        const { size } = await fh.stat();
        if (size > offset) {
          const buf = Buffer.alloc(size - offset);
          await fh.read(buf, 0, buf.length, offset);
          chunk = buf.toString("utf-8");
          offset = size;
        }
      } finally {
        await fh.close();
      }
    } catch {
      // File not created yet — poll until it appears.
    }

    if (chunk) {
      leftover += chunk;
      const nl = leftover.lastIndexOf("\n");
      if (nl >= 0) {
        const complete = leftover.slice(0, nl);
        leftover = leftover.slice(nl + 1);
        for (const line of complete.split("\n")) {
          if (!line) continue;
          const event = JSON.parse(line) as T;
          yield event;
          if (isTerminal(event)) {
            if (!opts.reopens) return;
            sawTerminal = true;
          } else if (sawTerminal && opts.reopens?.(event)) {
            sawTerminal = false;
          }
        }
      }
    }

    // After a terminal event: re-check for appended bytes immediately (no
    // poll delay for the common completed-run tail); at EOF, close — unless
    // the producer is live again (a resume re-attached), then keep following.
    if (sawTerminal) {
      if (chunk) continue;
      if (!(opts.stillLive?.() ?? false)) return;
    }

    await sleep(intervalMs);
  }
}

// ── Interface ──────────────────────────────────────────────────────────────

/** Options for `RunStore.tailEvents`. */
export interface TailOpts {
  /** Poll interval while following a live log (default 250ms). */
  intervalMs?: number;
  /** Stop the tail early (e.g. on client disconnect). */
  signal?: AbortSignal;
  /** Consulted at EOF after a terminal event: a live producer (the server's
   *  registered run controller) means a resume is in flight — keep following
   *  instead of closing. Default: close at EOF. */
  stillLive?: () => boolean;
}

/**
 * The persistence boundary for runs — the full contract, writes AND reads.
 * Every backend (filesystem, memory, a database) implements all of it; the
 * server capability-gates on nothing else. `tailEvents` has a generic
 * polling implementation (`tailFromPolling`) built on `getRunEvents`, so a
 * backend without a native tail implements the five data methods and
 * delegates.
 *
 * Tail contract (RUN_CONTROL_SPEC §5.2): yield the run's history from event
 * 0, then follow appends until a terminal event (`run.end` / `run.error` /
 * `run.cancelled`). A terminal event doesn't close the tail immediately — a
 * later `run.resumed` REOPENS the log (durable resume appends past the old
 * terminal event), so the tail scans ahead and only closes at EOF, or keeps
 * following if `opts.stillLive()` reports a resume in flight.
 */
export interface RunStore {
  append(workflow: string, runId: string, event: RunEvent): Promise<void>;
  finalize(workflow: string, runId: string, summary: RunSummary): Promise<void>;
  /** Run ids for a workflow, newest first. */
  listRuns(workflow: string): Promise<string[]>;
  /** The finalized summary, or null while the run is in flight / if it never
   *  finalized (crash) — callers fall back to `summarizeFromEvents`. */
  getRunSummary(workflow: string, runId: string): Promise<RunSummary | null>;
  /** The full event log (empty for an unknown run). */
  getRunEvents(workflow: string, runId: string): Promise<RunEvent[]>;
  /** History → live tail; see the interface doc for terminality. */
  tailEvents(workflow: string, runId: string, opts?: TailOpts): AsyncGenerator<RunEvent>;
  /** Start time (epoch ms) of the most recent run, or null if never run. */
  lastRunAt(workflow: string): Promise<number | null>;
}

/**
 * Generic `tailEvents` for backends without a native append-following
 * primitive: re-read the run's events on each poll and yield past the index
 * cursor. Same terminal / reopen / `stillLive` semantics as the file tail
 * (`tailJsonl`), which `FileRunStore` keeps because a byte-offset read is
 * cheaper than a full re-read per poll.
 */
export async function* tailFromPolling(
  store: Pick<RunStore, "getRunEvents">,
  workflow: string,
  runId: string,
  opts: TailOpts = {},
): AsyncGenerator<RunEvent> {
  const intervalMs = opts.intervalMs ?? 250;
  let cursor = 0;
  let sawTerminal = false;
  while (true) {
    if (opts.signal?.aborted) return;
    const events = await store.getRunEvents(workflow, runId);
    const fresh = events.slice(cursor);
    cursor = events.length;
    for (const event of fresh) {
      yield event;
      if (isTerminal(event)) sawTerminal = true;
      else if (sawTerminal && reopensRun(event)) sawTerminal = false;
    }
    if (sawTerminal) {
      if (fresh.length > 0) continue; // drain immediately, no poll delay
      if (!(opts.stillLive?.() ?? false)) return;
    }
    await sleep(intervalMs);
  }
}

/** Newest run's start time from the run-id list — run ids are millisecond
 *  timestamps (`generateRunId`), so the max parseable id is the latest.
 *  Shared by backends whose ids follow that convention. */
export function lastRunAtFromIds(runIds: string[]): number | null {
  let max: number | null = null;
  for (const id of runIds) {
    const t = parseInt(id, 10);
    if (!isNaN(t) && (max == null || t > max)) max = t;
  }
  return max;
}

// ── Partial summary (reconstructed from events) ────────────────────────────

/**
 * A best-effort summary for a run with no `run.json` — in-flight, or
 * orphaned by a crash/restart before `finalize` ran. Everything here is
 * derived from the append-only event log, which IS durable per-step: the
 * run's input from `run.start`, the latest output of every top-level step,
 * and the last error seen anywhere in the tree. `partial: true` is the
 * discriminator — a consumer that needs a terminal result must not treat
 * this as one.
 */
export interface PartialRunSummary {
  runId: string;
  workflow: string;
  partial: true;
  /** Live state when the caller knows it ("running" / "paused"), else
   *  "stale" (no controller — the process that ran it is gone; resumable). */
  status: string;
  startedAt?: string;
  lastEventAt?: string;
  eventCount: number;
  input?: unknown;
  /** Latest completed output per TOP-LEVEL step (path `<wf>/<stepId>` with
   *  no deeper segment and no `#iteration`), in completion order. */
  steps: Record<string, unknown>;
  /** The last `step.error` seen at any depth — where a dead run stopped. */
  lastError?: { path: string; message: string; ts: string };
  /** The last event of any kind — how far the log got. */
  lastEvent?: { type: string; path: string; ts: string };
}

/**
 * Reconstruct a `PartialRunSummary` from a run's event log. Pure over the
 * events array so it is equally usable on a live tail, a stale run's log,
 * or in tests. Returns null for an empty log (no such run).
 */
export function summarizeFromEvents(
  workflow: string,
  runId: string,
  events: RunEvent[],
  status = "stale",
): PartialRunSummary | null {
  if (events.length === 0) return null;

  const prefix = `${workflow}/`;
  const isTopLevelStep = (path: string): boolean => {
    if (!path.startsWith(prefix)) return false;
    const rest = path.slice(prefix.length);
    return rest.length > 0 && !rest.includes("/") && !rest.includes("#");
  };

  const summary: PartialRunSummary = {
    runId,
    workflow,
    partial: true,
    status,
    eventCount: events.length,
    steps: {},
  };

  for (const e of events) {
    if (e.type === "run.start") {
      summary.startedAt ??= e.ts;
      if (e.input !== undefined) summary.input = e.input;
    }
    if ((e.type === "step.end" || e.type === "step.replayed") && isTopLevelStep(e.path)) {
      const stepId = e.path.slice(prefix.length);
      delete summary.steps[stepId]; // re-insert so key order tracks completion order
      summary.steps[stepId] = e.output;
    }
    if (e.type === "step.error") {
      summary.lastError = { path: e.path, message: e.error?.message ?? "unknown", ts: e.ts };
    }
  }

  const last = events[events.length - 1]!;
  summary.lastEventAt = last.ts;
  summary.lastEvent = { type: last.type, path: last.path, ts: last.ts };
  return summary;
}

// ── Filesystem implementation ──────────────────────────────────────────────

/**
 * Stores runs under `<workspaceRoot>/workflows/<workflow>/runs/<runId>/`.
 * runId is a millisecond timestamp, giving natural sort order and easy pagination.
 */
export class FileRunStore implements RunStore {
  private workspaceRoot: string;

  constructor(workspaceRoot: string) {
    this.workspaceRoot = workspaceRoot;
  }

  private runDir(workflow: string, runId: string): string {
    return join(this.workspaceRoot, "workflows", workflow, "runs", runId);
  }

  async append(workflow: string, runId: string, event: RunEvent): Promise<void> {
    const dir = this.runDir(workflow, runId);
    await mkdir(dir, { recursive: true });
    const line = JSON.stringify(event) + "\n";
    await appendFile(join(dir, "events.jsonl"), line, "utf-8");
  }

  async finalize(workflow: string, runId: string, summary: RunSummary): Promise<void> {
    const dir = this.runDir(workflow, runId);
    await mkdir(dir, { recursive: true });
    await writeFile(
      join(dir, "run.json"),
      JSON.stringify(summary, null, 2),
      "utf-8",
    );
  }

  /** List runs for a workflow, sorted newest first. Returns dir names (timestamps). */
  async listRuns(workflow: string): Promise<string[]> {
    const runsDir = join(this.workspaceRoot, "workflows", workflow, "runs");
    try {
      const entries = await readdir(runsDir);
      // Sort descending (newest first) — timestamps sort lexicographically
      return entries.sort((a, b) => b.localeCompare(a));
    } catch {
      return [];
    }
  }

  /** Read run.json for a specific run. */
  async getRunSummary(workflow: string, runId: string): Promise<RunSummary | null> {
    try {
      const raw = await readFile(
        join(this.runDir(workflow, runId), "run.json"),
        "utf-8",
      );
      return JSON.parse(raw) as RunSummary;
    } catch {
      return null;
    }
  }

  /**
   * Tail a run's event log: yield every event from the start of the file
   * (history), then follow appends (live) until a terminal event
   * (`run.end` / `run.error`) is seen, at which point the generator returns.
   *
   * The append-only log is the ordered source of truth, so the history→live
   * join is naturally race-free: we read from a byte offset to EOF, then
   * keep re-reading from the new offset. Partial trailing lines (a write
   * caught mid-flush) are buffered until their newline arrives. One code path
   * serves completed *and* in-flight runs — a completed run drains the file
   * and returns immediately; a live run polls (`intervalMs`) for appends.
   *
   * The only "polling" is the server noticing appends — invisible to clients.
   * Pass an `AbortSignal` (e.g. on client disconnect) to stop early.
   */
  async *tailEvents(
    workflow: string,
    runId: string,
    opts: TailOpts = {},
  ): AsyncGenerator<RunEvent> {
    const file = join(this.runDir(workflow, runId), "events.jsonl");
    // `run.error`/`run.cancelled` are no longer unconditionally terminal: a
    // later `run.resumed` reopens the stream (historical tails scan ahead;
    // live tails consult `opts.stillLive` — the server's controllers map).
    yield* tailJsonl<RunEvent>(file, isTerminal, { ...opts, reopens: reopensRun });
  }

  /** Read events.jsonl for a specific run. Tolerates a TORN TAIL: a process
   *  killed mid-append can leave a truncated final line (single-write
   *  atomicity is not guaranteed for large outputs) — it belongs to an
   *  incomplete unit by definition, so it is skipped, not fatal (§5.1). */
  async getRunEvents(workflow: string, runId: string): Promise<RunEvent[]> {
    let raw: string;
    try {
      raw = await readFile(
        join(this.runDir(workflow, runId), "events.jsonl"),
        "utf-8",
      );
    } catch {
      return [];
    }
    const lines = raw.trim().split("\n").filter(Boolean);
    const events: RunEvent[] = [];
    for (let i = 0; i < lines.length; i++) {
      try {
        events.push(JSON.parse(lines[i]!) as RunEvent);
      } catch (err) {
        if (i === lines.length - 1) continue; // torn tail — skip
        throw err; // corruption anywhere else is a real error
      }
    }
    return events;
  }

  async lastRunAt(workflow: string): Promise<number | null> {
    return lastRunAtFromIds(await this.listRuns(workflow));
  }
}

// ── In-memory implementation ───────────────────────────────────────────────

/**
 * A complete ephemeral backend (not just a write-only test stub): run
 * history, SSE reattach, durable resume, and promotions all work over it —
 * the records just don't survive the process.
 */
export class MemoryRunStore implements RunStore {
  events: Map<string, RunEvent[]> = new Map();
  summaries: Map<string, RunSummary> = new Map();

  private key(workflow: string, runId: string): string {
    return `${workflow}/${runId}`;
  }

  async listRuns(workflow: string): Promise<string[]> {
    const prefix = `${workflow}/`;
    const ids = new Set<string>();
    for (const k of this.events.keys()) if (k.startsWith(prefix)) ids.add(k.slice(prefix.length));
    for (const k of this.summaries.keys()) if (k.startsWith(prefix)) ids.add(k.slice(prefix.length));
    return [...ids].sort((a, b) => b.localeCompare(a));
  }

  async getRunSummary(workflow: string, runId: string): Promise<RunSummary | null> {
    return this.summaries.get(this.key(workflow, runId)) ?? null;
  }

  async getRunEvents(workflow: string, runId: string): Promise<RunEvent[]> {
    return [...(this.events.get(this.key(workflow, runId)) ?? [])];
  }

  tailEvents(workflow: string, runId: string, opts: TailOpts = {}): AsyncGenerator<RunEvent> {
    return tailFromPolling(this, workflow, runId, opts);
  }

  async lastRunAt(workflow: string): Promise<number | null> {
    return lastRunAtFromIds(await this.listRuns(workflow));
  }

  async append(workflow: string, runId: string, event: RunEvent): Promise<void> {
    const k = this.key(workflow, runId);
    if (!this.events.has(k)) {
      this.events.set(k, []);
    }
    this.events.get(k)!.push(event);
  }

  async finalize(workflow: string, runId: string, summary: RunSummary): Promise<void> {
    this.summaries.set(this.key(workflow, runId), summary);
  }

  /** Helper for tests: get events by workflow + runId. */
  getEvents(workflow: string, runId: string): RunEvent[] {
    return this.events.get(this.key(workflow, runId)) ?? [];
  }

  /** Helper for tests: get summary by workflow + runId. */
  getSummary(workflow: string, runId: string): RunSummary | undefined {
    return this.summaries.get(this.key(workflow, runId));
  }
}

/** Generate a timestamp-based run ID. */
export function generateRunId(): string {
  return Date.now().toString();
}
