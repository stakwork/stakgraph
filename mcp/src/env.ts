/**
 * The process environment, read in ONE place — mcp's twin of strut's
 * `src/env.ts`.
 *
 * The lab embeds strut, and strut's env boundary (stakwork/strut#228) is
 * that a step is TypeScript running in THIS process: `process.env` is
 * readable by any step and by any child a step spawns with it. So once the
 * lab has booted, mcp calls strut's `scrubProcessEnv()` (lab/mount.ts),
 * which empties `process.env` down to what a shell needs (`KEEP_ENV`). A
 * step then sees no deployment key, while `ctx.services.secrets` still
 * resolves one through strut's own boot snapshot.
 *
 * mcp's own code reads the env at request time too — `API_TOKEN` on every
 * authenticated route, `NEO4J_*` when a driver reconnects, the provider keys
 * aieo resolves per call, `GITHUB_TOKEN` on a sweep. Those reads come here,
 * never to `process.env`:
 *
 * - `env` is `process.env` until `freezeEnv()` ran, and the copy it took
 *   after — the same copy strut's `bootEnv()` holds, since `freezeEnv` is
 *   called right before `scrubProcessEnv` (lab/mount.ts). `env.X`, `env[X]`,
 *   `{ ...env }` and `X in env` all work, so a read is a one-word change.
 * - A module-level `const X = process.env.X ?? …` is read at import, before
 *   any scrub, and may stay as it is.
 * - A write (`process.env.X = …`) that must reach a child spawned with
 *   `process.env` goes to `process.env` itself, before the scrub; it is in
 *   the copy too.
 *
 * Tests never freeze (nothing in them boots the lab), so `env` tracks the
 * `process.env` they set and restore.
 */

let snapshot: NodeJS.ProcessEnv = process.env;
let frozen = false;

/** Copy `process.env` as it is now; `env` reads the copy from here on.
 *  Idempotent. Returns whether this call took the copy. */
export function freezeEnv(): boolean {
  if (frozen) return false;
  snapshot = { ...process.env };
  frozen = true;
  return true;
}

/** Whether `freezeEnv()` ran (so `env` is a copy, not `process.env`). */
export function envFrozen(): boolean {
  return frozen;
}

const key = (k: string | symbol): string | undefined => (typeof k === "string" ? k : undefined);

/** The environment mcp booted with — see the module comment. */
export const env: NodeJS.ProcessEnv = new Proxy({} as NodeJS.ProcessEnv, {
  get: (_t, k) => (key(k) === undefined ? undefined : snapshot[k as string]),
  set: (_t, k, v) => {
    const name = key(k);
    if (name === undefined) return false;
    snapshot[name] = v;
    return true;
  },
  has: (_t, k) => key(k) !== undefined && k in snapshot,
  deleteProperty: (_t, k) => {
    const name = key(k);
    if (name === undefined) return false;
    delete snapshot[name];
    return true;
  },
  ownKeys: () => Reflect.ownKeys(snapshot),
  getOwnPropertyDescriptor: (_t, k) => {
    const d = Object.getOwnPropertyDescriptor(snapshot, k);
    return d && { ...d, configurable: true };
  },
});
