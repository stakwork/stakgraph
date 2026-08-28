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

export interface RunStore {
  append(workflow: string, runId: string, event: RunEvent): Promise<void>;
  finalize(workflow: string, runId: string, summary: RunSummary): Promise<void>;
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
    opts: { intervalMs?: number; signal?: AbortSignal; stillLive?: () => boolean } = {},
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
}

// ── In-memory implementation (for testing) ─────────────────────────────────

export class MemoryRunStore implements RunStore {
  events: Map<string, RunEvent[]> = new Map();
  summaries: Map<string, RunSummary> = new Map();

  private key(workflow: string, runId: string): string {
    return `${workflow}/${runId}`;
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
