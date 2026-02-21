import * as fs from "fs";

export const LOGS_DIR = process.env.LOGS_DIR || "/tmp/logs";

export function ensureLogsDir(): string {
  if (!fs.existsSync(LOGS_DIR)) {
    fs.mkdirSync(LOGS_DIR, { recursive: true });
  }
  return LOGS_DIR;
}
