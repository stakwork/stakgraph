/**
 * Where vein caches downloaded model files (MiniLM embeddings, sherpa STT).
 *
 * Resolution order:
 *   1. `VEIN_MODEL_DIR` — explicit.
 *   2. `VEIN_MODEL_CACHE` — the older alias MiniLM shipped with.
 *   3. `<cache root>/vein/models`, where the cache root is `VEIN_CACHE_DIR`,
 *      else `XDG_CACHE_HOME`, else `~/.cache`. Same convention as the GAIA
 *      checkout in mcp (`<cache root>/vein/gaia`), so on a server that
 *      mounts a volume at `~/.cache/vein` — or sets `VEIN_CACHE_DIR` — models
 *      persist across restarts with no extra configuration.
 *   4. Back-compat: with nothing configured, a pre-existing `~/.cache/vein-models`
 *      (the old default) keeps being used so dev machines don't re-download.
 */
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export const LEGACY_MODEL_DIR = join(homedir(), ".cache", "vein-models");

export function modelDirFromEnv(env: Record<string, string | undefined> = process.env): string {
  const explicit = env["VEIN_MODEL_DIR"] ?? env["VEIN_MODEL_CACHE"];
  if (explicit) return explicit;
  const root = env["VEIN_CACHE_DIR"] ?? env["XDG_CACHE_HOME"];
  if (root) return join(root, "vein", "models");
  const modern = join(homedir(), ".cache", "vein", "models");
  if (!existsSync(modern) && existsSync(LEGACY_MODEL_DIR)) return LEGACY_MODEL_DIR;
  return modern;
}
