import { mkdir, writeFile, appendFile, readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import type { RunEvent, RunSummary } from "./core.js";

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

  /** Read events.jsonl for a specific run. */
  async getRunEvents(workflow: string, runId: string): Promise<RunEvent[]> {
    try {
      const raw = await readFile(
        join(this.runDir(workflow, runId), "events.jsonl"),
        "utf-8",
      );
      return raw
        .trim()
        .split("\n")
        .filter(Boolean)
        .map((line) => JSON.parse(line) as RunEvent);
    } catch {
      return [];
    }
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
