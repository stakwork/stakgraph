// ── localStorage helpers ───────────────────────────────────────────────────
//
// Small, typed wrapper over window.localStorage for UI preferences and
// session state. All keys are namespaced under `strut/` so they don't
// collide with anything else served on the same origin. All operations
// are crash-safe: a thrown error (quota, privacy mode, JSON parse) is
// swallowed and the fallback is returned.

const PREFIX = "strut/";

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

/** Dictation (SettingsDialog + the mic in ChatFlyout). Model installation is
 *  a server fact; this is the browser's choice to use it, and which models. */
export interface SttSettings {
  enabled: boolean;
  /** Finals model id (hotword-capable by default). */
  model: string | null;
  /** Fast partials model id; null = the finals model does both. */
  partialModel: string | null;
  /** Inline hotwords, one per line, optional ` :score`. */
  hotwords: string;
}
const STT_DEFAULTS: SttSettings = { enabled: false, model: null, partialModel: null, hotwords: "" };
export const sttSettings = {
  get: (): SttSettings => ({ ...STT_DEFAULTS, ...load<Partial<SttSettings>>("stt", {}) }),
  set: (s: SttSettings) => save("stt", s),
};

/** Most recent run-input form values, keyed by workflow name. */
export const recentRunInput = {
  get: (workflow: string) =>
    load<Record<string, unknown>>(`runInput/${workflow}`, {}),
  set: (workflow: string, input: Record<string, unknown>) =>
    save(`runInput/${workflow}`, input),
};
