import { existsSync, readdirSync, statSync, unlinkSync, rmSync } from "node:fs";
import path from "node:path";

/**
 * A persistent, well-known directory the agent may write output files to
 * (reports, answer.json, etc.) that must survive container restarts. Unlike
 * `/tmp` (ephemeral) and cloned repos (re-clonable), this is expected to be
 * backed by a durable volume in the deployment. Set via AGENT_ARTIFACTS_DIR.
 */
export const AGENT_ARTIFACTS_DIR = process.env.AGENT_ARTIFACTS_DIR;

const ARTIFACT_MAX_AGE_MS = parseInt(
  process.env.AGENT_ARTIFACTS_MAX_AGE_MS || String(7 * 24 * 60 * 60 * 1000),
  10
); // default 7 days, configurable via AGENT_ARTIFACTS_MAX_AGE_MS env var

/**
 * Delete artifacts older than ARTIFACT_MAX_AGE_MS from AGENT_ARTIFACTS_DIR.
 * Because the dir lives on a durable volume, files accumulate across restarts;
 * this reclaims space. No-op when AGENT_ARTIFACTS_DIR is unset or missing.
 * Call on startup and periodically.
 */
export function pruneExpiredArtifacts(): number {
  const dir = AGENT_ARTIFACTS_DIR;
  if (!dir || !existsSync(dir)) return 0;

  const now = Date.now();
  let pruned = 0;
  for (const entry of readdirSync(dir)) {
    const entryPath = path.join(dir, entry);
    try {
      const stat = statSync(entryPath);
      if (now - stat.mtimeMs <= ARTIFACT_MAX_AGE_MS) continue;
      if (stat.isDirectory()) {
        rmSync(entryPath, { recursive: true, force: true });
      } else {
        unlinkSync(entryPath);
      }
      pruned++;
    } catch {
      // ignore stat/unlink errors for individual entries
    }
  }
  if (pruned > 0) {
    console.log(`[artifacts] pruned ${pruned} expired artifact(s) from ${dir}`);
  }
  return pruned;
}
