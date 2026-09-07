/**
 * Where strut caches downloaded model files (MiniLM embeddings, sherpa STT).
 *
 * Resolution order:
 *   1. `STRUT_MODEL_DIR` — explicit.
 *   2. `STRUT_MODEL_CACHE` — the older alias MiniLM shipped with.
 *   3. `<cache root>/strut/models`, where the cache root is `STRUT_CACHE_DIR`,
 *      else `XDG_CACHE_HOME`, else `~/.cache`. Same convention as the GAIA
 *      checkout in mcp (`<cache root>/strut/gaia`), so on a server that
 *      mounts a volume at `~/.cache/strut` — or sets `STRUT_CACHE_DIR` — models
 *      persist across restarts with no extra configuration.
 *   4. Back-compat: with nothing configured, a pre-existing legacy dir from
 *      before the rename — `~/.cache/vein/models`, then the older
 *      `~/.cache/vein-models` — keeps being used so dev machines don't re-download.
 */
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

/** Pre-rename default locations, newest first. Only consulted when nothing is
 *  configured and the modern dir does not exist yet. */
export const LEGACY_MODEL_DIRS = [
  join(homedir(), ".cache", "vein", "models"),
  join(homedir(), ".cache", "vein-models"),
];

export function modelDirFromEnv(env: Record<string, string | undefined> = process.env): string {
  const explicit = env["STRUT_MODEL_DIR"] ?? env["STRUT_MODEL_CACHE"];
  if (explicit) return explicit;
  const root = env["STRUT_CACHE_DIR"] ?? env["XDG_CACHE_HOME"];
  if (root) return join(root, "strut", "models");
  const modern = join(homedir(), ".cache", "strut", "models");
  if (!existsSync(modern)) {
    const legacy = LEGACY_MODEL_DIRS.find((d) => existsSync(d));
    if (legacy) return legacy;
  }
  return modern;
}
