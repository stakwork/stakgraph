import * as uuid from "uuid";
import { existsSync, mkdirSync, writeFileSync, readFileSync, unlinkSync } from "fs";
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

export function startReq(): string {
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

  writeToDisk(key, { status: "pending" });

  return key;
}

export function finishReq(id: string, result: any) {
  if (META[id]) {
    META[id].status = "completed";
    writeToDisk(id, { status: "completed", result });
  }
}

export function failReq(id: string, error: any) {
  if (META[id]) {
    META[id].status = "failed";
    // Serialize error safely — Error objects don't JSON.stringify well
    const serializedError =
      error instanceof Error
        ? { message: error.message, stack: error.stack }
        : error;
    writeToDisk(id, { status: "failed", error: serializedError });
  }
}

export function updateReq(id: string, progress: any) {
  if (META[id]) {
    // Read existing, merge progress, write back
    const existing = readFromDisk(id);
    if (existing) {
      existing.progress = progress;
      writeToDisk(id, existing);
    }
  }
}

export function checkReq(id: string): Request {
  // Quick reject if we've never seen this id
  if (!META[id]) return undefined as any;
  // Read full data from disk
  return readFromDisk(id) as Request;
}
