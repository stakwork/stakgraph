import neo4j, { Integer } from "neo4j-driver";

/**
 * Canonical `date_added_to_graph` stamp: **epoch milliseconds** as a Neo4j
 * **Integer** (a plain JS number would store as a Float — always bind through
 * this helper so the wire format can't drift back to float-seconds,
 * integer-seconds, or formatted strings).
 *
 * Semantic: "epoch milliseconds, Neo4j Integer, **set once on create** — do
 * not use for ON MATCH / plain `SET` updates." Merge templates must assign it
 * only under `ON CREATE SET` (or via `COALESCE(..., $ts)`).
 */
export function nowEpochMs(): Integer {
  return neo4j.int(Date.now());
}

/**
 * Canonical magnitude boundary for a `date_added_to_graph` value: anything
 * **above** this is epoch-milliseconds (already canonical); at-or-below is
 * still epoch-seconds. Must match jarvis-backend's
 * `TimeFormatter.epoch_value_to_ms` boundary exactly (values ≤ 10**12 are
 * seconds) — do not introduce a divergent variant.
 *
 * Mirrors the Cypher expression in `queries.ts#epochMsExpr` — keep in sync.
 */
export const EPOCH_MS_BOUNDARY = 1_000_000_000_000;

export type EpochInput =
  | number
  | string
  | { low: number; high?: number }
  | null
  | undefined;

/**
 * Magnitude discriminator: normalize any stored `date_added_to_graph` value
 * to epoch **milliseconds**.
 *
 * The graph holds a mix of legacy formats until the data backfill migration
 * runs (7-decimal strings, float seconds, integer seconds) alongside new
 * epoch-ms Integers. Rule (defined ONCE here — do not reinvent per site):
 * - value <= 1e12 → legacy epoch-seconds → × 1000
 * - value  > 1e12 → already epoch-milliseconds → pass through
 * - null / undefined / unparseable → null
 *
 * Sub-ms precision is truncated (`floor`) to match jarvis-backend's integer
 * conversion. Also tolerates a raw Neo4j Integer (`{low, high}`) object in
 * case a read path bypassed `clean_node`/`deser_node`.
 */
export function toEpochMs(value: EpochInput): number | null {
  if (value === null || value === undefined) return null;
  if (typeof value === "object") {
    if (typeof value.low !== "number") return null;
    return toEpochMs((value.high ?? 0) * 2 ** 32 + (value.low >>> 0));
  }
  const n = typeof value === "number" ? value : parseFloat(value);
  if (!Number.isFinite(n)) return null;
  return n <= EPOCH_MS_BOUNDARY ? Math.floor(n * 1000) : Math.floor(n);
}

/**
 * Normalize an epoch value that may be in seconds or milliseconds to
 * epoch-milliseconds, mirroring jarvis-backend's
 * `TimeFormatter.epoch_value_to_ms`. Thin wrapper over `toEpochMs` for
 * numeric bind sites (e.g. `$since` in `nodes_by_type`).
 */
export function epochValueToMs(value: number): number {
  return toEpochMs(value) as number;
}

/**
 * Age of a node in **hours**, given its stored `date_added_to_graph` value in
 * any legacy/new format (routed through `toEpochMs`). Returns null when the
 * stored value is missing or unparseable — callers decide the fallback.
 */
export function nodeAgeHours(
  nodeAge: EpochInput,
  nowMs: number = Date.now(),
): number | null {
  const ms = toEpochMs(nodeAge);
  return ms === null ? null : (nowMs - ms) / 3_600_000;
}

/**
 * Age in hours of a node stamped with `date_added_to_graph`. Mixed legacy
 * seconds / new-ms values are normalized via `toEpochMs` first. Used by the
 * intelligence tools' cache-control age check.
 */
export function dateAddedAgeHours(
  dateAddedMs: number,
  now: number = Date.now(),
): number {
  const ms = toEpochMs(dateAddedMs) ?? dateAddedMs;
  return (now - ms) / (3600 * 1000);
}
