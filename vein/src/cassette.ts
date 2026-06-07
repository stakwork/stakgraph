/**
 * Record / replay for the services bag — the "safe inner loop" that lets the
 * AI chat author an adapter and hammer on it OFFLINE: record once against the
 * real world, then iterate against the recording (deterministic, no rate
 * limits, no cost, no side effects).
 *
 * It works by wrapping the services bag (the seam every adapter already goes
 * through — see `capabilities.ts`). In `record` mode each service call is run
 * for real and its `(key, args) → result` captured; in `replay` mode the call
 * is served from the recording by matching `(key, args)` instead of hitting the
 * real service.
 *
 * **Secret hygiene.** Secrets enter through one boundary (`services.secrets`).
 * The wrapper canonicalizes every secret VALUE to a stable token
 * (`{{secret:NAME}}`) before anything is written to — or matched against — the
 * cassette, so real keys never reach disk, and record↔replay still match even
 * though replay has no real credentials.
 */

const SECRET_TOKEN = (name: string) => `{{secret:${name}}}`;

/** One recorded service call. `args`/`result` are post-redaction (secret-safe). */
export interface CassetteEntry {
  /** `"<service>.<method>"`, or just `"<service>"` for a callable service. */
  key: string;
  args: unknown[];
  result?: unknown;
  /** Present instead of `result` when the recorded call threw. */
  error?: string;
}

export interface Cassette {
  entries: CassetteEntry[];
}

export type CassetteMode = "record" | "replay";

export interface WithCassetteOptions {
  mode: CassetteMode;
  /** In `record` mode, entries are appended here. In `replay` mode, calls are
   *  matched against these entries. */
  cassette: Cassette;
  /** Name of the service treated as the secrets boundary (its return values are
   *  canonicalized to tokens and scrubbed from the cassette). Default
   *  `"secrets"`; pass `null` to disable secret handling. */
  secretsService?: string | null;
}

export function emptyCassette(): Cassette {
  return { entries: [] };
}

/**
 * Wrap a services bag with record/replay. Returns a structurally-identical bag
 * (same call sites in adapter code) backed by the cassette per `mode`.
 */
export function withCassette<TServices extends Record<string, unknown>>(
  services: TServices,
  opts: WithCassetteOptions,
): TServices {
  const { mode, cassette } = opts;
  const secretsService =
    opts.secretsService === undefined ? "secrets" : opts.secretsService;

  // name → real secret value, learned at the secrets boundary during record.
  // Used to scrub real values out of recorded args/results.
  const secretValues = new Map<string, string>();
  // Indices already returned in replay, so repeated identical calls walk the
  // recording in order (e.g. paginated GETs) instead of re-matching the first.
  const consumed = new Set<number>();

  const redact = (v: unknown): unknown => {
    if (secretValues.size === 0) return v;
    return deepReplace(v, (s) => {
      let out = s;
      for (const [name, value] of secretValues) {
        if (value) out = out.split(value).join(SECRET_TOKEN(name));
      }
      return out;
    });
  };

  /** Handle one wrapped call (non-secrets services). */
  const handleCall = async (
    key: string,
    fn: (...a: unknown[]) => unknown,
    args: unknown[],
  ): Promise<unknown> => {
    const redactedArgs = redact(args) as unknown[];
    if (mode === "replay") {
      const idx = cassette.entries.findIndex(
        (e, i) =>
          !consumed.has(i) && e.key === key && argsEqual(e.args, redactedArgs),
      );
      if (idx === -1) {
        throw new Error(
          `cassette replay: no recorded call for "${key}" with args ${stableJson(redactedArgs)}`,
        );
      }
      consumed.add(idx);
      const entry = cassette.entries[idx]!;
      if (entry.error !== undefined) throw new Error(entry.error);
      return entry.result;
    }
    // record
    try {
      const result = await fn(...args);
      cassette.entries.push({ key, args: redactedArgs, result: redact(result) });
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      cassette.entries.push({ key, args: redactedArgs, error: message });
      throw err;
    }
  };

  /** The secrets boundary: NOT recorded. In record mode, learn name→value so it
   *  can be scrubbed elsewhere; in replay mode, return the stable token so
   *  downstream args carry tokens that match the recording. */
  const wrapSecretsGet =
    (realGet: (name: string) => Promise<string | undefined>) =>
    async (name: string): Promise<string | undefined> => {
      if (mode === "replay") return SECRET_TOKEN(name);
      const value = await realGet(name);
      if (typeof value === "string") secretValues.set(name, value);
      return value;
    };

  const wrapService = (name: string, svc: unknown): unknown => {
    const isSecrets = secretsService != null && name === secretsService;

    if (typeof svc === "function") {
      const fn = svc as (...a: unknown[]) => unknown;
      return (...args: unknown[]) => handleCall(name, fn, args);
    }
    if (svc && typeof svc === "object") {
      return new Proxy(svc as Record<string, unknown>, {
        get(target, prop) {
          const member = target[prop as string];
          if (typeof member !== "function") return member;
          const method = String(prop);
          const bound = (member as (...a: unknown[]) => unknown).bind(target);
          if (isSecrets && method === "get") {
            return wrapSecretsGet(
              bound as (name: string) => Promise<string | undefined>,
            );
          }
          return (...args: unknown[]) =>
            handleCall(`${name}.${method}`, bound, args);
        },
      });
    }
    return svc;
  };

  return new Proxy(services, {
    get(target, prop) {
      const name = String(prop);
      const svc = target[prop as keyof TServices];
      if (svc == null) return svc;
      return wrapService(name, svc);
    },
  }) as TServices;
}

// ── helpers ──────────────────────────────────────────────────────────────

/** Recursively map every string in a JSON-like value through `fn`. */
function deepReplace(v: unknown, fn: (s: string) => string): unknown {
  if (typeof v === "string") return fn(v);
  if (Array.isArray(v)) return v.map((x) => deepReplace(x, fn));
  if (v && typeof v === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, val] of Object.entries(v)) out[k] = deepReplace(val, fn);
    return out;
  }
  return v;
}

/** Deterministic JSON with sorted object keys, so arg comparison is stable. */
function stableJson(v: unknown): string {
  return JSON.stringify(v, (_k, val) => {
    if (val && typeof val === "object" && !Array.isArray(val)) {
      const sorted: Record<string, unknown> = {};
      for (const k of Object.keys(val).sort()) sorted[k] = (val as Record<string, unknown>)[k];
      return sorted;
    }
    return val;
  });
}

function argsEqual(a: unknown[], b: unknown[]): boolean {
  return stableJson(a) === stableJson(b);
}

// ── persistence ──────────────────────────────────────────────────────────

/** Load a cassette from disk, or an empty one if the file doesn't exist. */
export async function loadCassette(path: string): Promise<Cassette> {
  const { readFile } = await import("node:fs/promises");
  try {
    const parsed = JSON.parse(await readFile(path, "utf-8"));
    return { entries: Array.isArray(parsed?.entries) ? parsed.entries : [] };
  } catch {
    return emptyCassette();
  }
}

/** Persist a cassette to disk (pretty JSON, creating parent dirs). */
export async function saveCassette(path: string, cassette: Cassette): Promise<void> {
  const { writeFile, mkdir } = await import("node:fs/promises");
  const { dirname } = await import("node:path");
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, JSON.stringify(cassette, null, 2));
}
