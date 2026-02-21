import * as fs from "fs";
import * as crypto from "crypto";
import * as path from "path";

export const LOGS_DIR = process.env.LOGS_DIR || "/tmp/logs";

/**
 * Get (and create) a per-run logs directory.
 * - With a sessionId: uses that as the subdirectory (persists across turns).
 * - Without: generates a random ID (caller should clean up after).
 */
export function createRunLogsDir(sessionId?: string): string {
  const subdir = sessionId || crypto.randomUUID().slice(0, 8);
  const dir = path.join(LOGS_DIR, subdir);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  return dir;
}

/** Remove a per-run logs directory and all its contents. */
export function cleanupRunLogsDir(dir: string): void {
  try {
    if (fs.existsSync(dir)) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  } catch (e) {
    console.warn("Failed to clean up logs dir:", dir, e);
  }
}
