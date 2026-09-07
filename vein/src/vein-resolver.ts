/**
 * Make `import "vein"` resolve to this running vein from anywhere — see
 * vein-resolve-hook.ts for why. Idempotent; called before the registry's
 * first dynamic import. Works under `node build/…` (entry = build/index.js)
 * and under tsx in dev (entry = src/index.ts, transpiled by tsx's own hook
 * further down the chain).
 */
import { register } from "node:module";

let registered = false;

export function registerVeinResolver(): void {
  if (registered) return;
  registered = true;
  const ts = import.meta.url.endsWith(".ts");
  const entry = new URL(ts ? "./index.ts" : "./index.js", import.meta.url).href;
  const hook = new URL(ts ? "./vein-resolve-hook.ts" : "./vein-resolve-hook.js", import.meta.url).href;
  register(hook, { parentURL: import.meta.url, data: { entry } });
}
