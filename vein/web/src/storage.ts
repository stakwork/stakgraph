// ── localStorage helpers ───────────────────────────────────────────────────
//
// Small, typed wrapper over window.localStorage for UI preferences and
// session state. All keys are namespaced under `vein/` so they don't
// collide with anything else served on the same origin. All operations
// are crash-safe: a thrown error (quota, privacy mode, JSON parse) is
// swallowed and the fallback is returned.

const PREFIX = "vein/";

export function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

export function save(key: string, value: unknown): void {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    // quota exceeded or storage disabled — swallow
  }
}

export function remove(key: string): void {
  try {
    localStorage.removeItem(PREFIX + key);
  } catch {
    // ignore
  }
}

// ── Typed accessors ────────────────────────────────────────────────────────

/** Most recent run-input form values, keyed by workflow name. */
export const recentRunInput = {
  get: (workflow: string) =>
    load<Record<string, unknown>>(`runInput/${workflow}`, {}),
  set: (workflow: string, input: Record<string, unknown>) =>
    save(`runInput/${workflow}`, input),
};
