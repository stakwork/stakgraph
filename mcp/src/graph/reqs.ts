import * as uuid from "uuid";
import {
  existsSync,
  mkdirSync,
  writeFileSync,
  readFileSync,
  unlinkSync,
  readdirSync,
  statSync,
} from "fs";
import path from "path";

const REQS_DIR = process.env.REQS_DIR || ".reqs";
const MAX_REQS = 100;

type Status = "pending" | "completed" | "failed";

// Lightweight in-memory index — only status, no heavy payloads
interface ReqMeta {
  status: Status;
}

const META: Record<string, ReqMeta> = {};
const REQ_ORDER: string[] = []; // Track insertion order for eviction

// Full on-disk shape (same interface callers expect from checkReq)
interface Request {
  status: Status;
  result?: any;
  error?: any;
  progress?: any;
  // Caller's terminal callback, persisted so the startup sweep can still
  // notify the receiver about runs orphaned by a process restart.
  webhookUrl?: string;
}

function ensureDir(): string {
  const dir = path.join(process.cwd(), REQS_DIR);
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
  return dir;
}

function reqFile(id: string): string {
  return path.join(ensureDir(), `${id}.json`);
}

function writeToDisk(id: string, data: Request): void {
  try {
    writeFileSync(reqFile(id), JSON.stringify(data));
  } catch (e) {
    console.error(`[reqs] Failed to write ${id}:`, e);
  }
}

function readFromDisk(id: string): Request | null {
  const fp = reqFile(id);
  if (!existsSync(fp)) return null;
  try {
    return JSON.parse(readFileSync(fp, "utf-8")) as Request;
  } catch (e) {
    console.error(`[reqs] Failed to read ${id}:`, e);
    return null;
  }
}

function deleteFromDisk(id: string): void {
  const fp = reqFile(id);
  try {
    if (existsSync(fp)) unlinkSync(fp);
  } catch (_) {}
}

export function startReq(webhookUrl?: string): string {
  const key = uuid.v4();

  // Evict oldest if at limit
  if (REQ_ORDER.length >= MAX_REQS) {
    const oldestKey = REQ_ORDER.shift();
    if (oldestKey) {
      delete META[oldestKey];
      deleteFromDisk(oldestKey);
    }
  }

  META[key] = { status: "pending" };
  REQ_ORDER.push(key);

  writeToDisk(key, { status: "pending", ...(webhookUrl && { webhookUrl }) });

  return key;
}

export interface OrphanedReq {
  request_id: string;
  error: string;
  webhookUrl?: string;
}

const RESTART_ORPHAN_ERROR = "server restarted while request was in-flight";

/**
 * Startup reconciliation for the on-disk request store. The in-memory META /
 * REQ_ORDER index dies with the process, which leaves two problems after a
 * restart: pre-restart files are never evicted (startReq only evicts ids it
 * knows about), and runs that were in-flight sit at "pending" forever because
 * the work driving them is gone.
 *
 * This re-adopts every surviving file into the eviction index (oldest first,
 * trimming past MAX_REQS) and flips still-"pending" files to "failed".
 * Returns the flipped entries so the caller can fire their terminal webhooks.
 */
export function sweepOrphanedReqs(): OrphanedReq[] {
  const dir = ensureDir();
  let files: string[];
  try {
    files = readdirSync(dir).filter((f) => f.endsWith(".json"));
  } catch (e) {
    console.error("[reqs] Startup sweep failed to list dir:", e);
    return [];
  }

  // Oldest first so re-adopted eviction order matches creation order
  const byAge = files
    .map((f) => {
      try {
        return { f, mtime: statSync(path.join(dir, f)).mtimeMs };
      } catch (_) {
        return null;
      }
    })
    .filter((x): x is { f: string; mtime: number } => x !== null)
    .sort((a, b) => a.mtime - b.mtime);

  const orphaned: OrphanedReq[] = [];
  for (const { f } of byAge) {
    const id = f.slice(0, -".json".length);
    const data = readFromDisk(id);
    if (!data) continue;

    if (data.status === "pending") {
      failReq(id, RESTART_ORPHAN_ERROR);
      orphaned.push({
        request_id: id,
        error: RESTART_ORPHAN_ERROR,
        ...(data.webhookUrl && { webhookUrl: data.webhookUrl }),
      });
    }

    if (!META[id]) {
      if (REQ_ORDER.length >= MAX_REQS) {
        const oldestKey = REQ_ORDER.shift();
        if (oldestKey) {
          delete META[oldestKey];
          deleteFromDisk(oldestKey);
        }
      }
      META[id] = { status: data.status === "pending" ? "failed" : data.status };
      REQ_ORDER.push(id);
    }
  }

  if (orphaned.length > 0) {
    console.log(
      `[reqs] Startup sweep marked ${orphaned.length} orphaned pending request(s) as failed`,
    );
  }
  return orphaned;
}

// Terminal writes always hit disk, even when the in-memory META entry was
// evicted (long run outliving MAX_REQS churn) or wiped (process restart):
// the .reqs file is the source of truth a late poller reads.

export function finishReq(id: string, result: any) {
  if (META[id]) META[id].status = "completed";
  writeToDisk(id, { status: "completed", result });
}

export function failReq(id: string, error: any) {
  if (META[id]) META[id].status = "failed";
  // Serialize error safely — Error objects don't JSON.stringify well
  const serializedError =
    error instanceof Error
      ? { message: error.message, stack: error.stack }
      : error;
  writeToDisk(id, { status: "failed", error: serializedError });
}

export function updateReq(id: string, progress: any) {
  // Read existing, merge progress, write back
  const existing = readFromDisk(id);
  if (existing) {
    existing.progress = progress;
    writeToDisk(id, existing);
  }
}

export function checkReq(id: string): Request {
  // Disk is authoritative: the in-memory META gate previously rejected any
  // id created before a process restart, 404ing completed runs whose
  // .reqs/<id>.json file physically survived.
  const data = readFromDisk(id);
  if (!data) return data as unknown as Request;
  // webhookUrl is internal bookkeeping, not part of the /progress shape
  const { webhookUrl: _webhookUrl, ...rest } = data;
  return rest;
}
