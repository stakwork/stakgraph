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
 */
import { homedir } from "node:os";
import { join } from "node:path";

export function modelDirFromEnv(env: Record<string, string | undefined> = process.env): string {
  const explicit = env["STRUT_MODEL_DIR"] ?? env["STRUT_MODEL_CACHE"];
  if (explicit) return explicit;
  const root = env["STRUT_CACHE_DIR"] ?? env["XDG_CACHE_HOME"];
  if (root) return join(root, "strut", "models");
  return join(homedir(), ".cache", "strut", "models");
}
