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
 */
export const EPOCH_MS_BOUNDARY = 1_000_000_000_000;

/**
 * Normalize an epoch value that may be in seconds or milliseconds to
 * epoch-milliseconds, mirroring jarvis-backend's
 * `TimeFormatter.epoch_value_to_ms`: seconds-magnitude values (≤ 10^12)
 * scale ×1000, ms-magnitude values pass through. Sub-ms precision is
 * truncated (`floor`) to match the backend's integer conversion.
 */
export function epochValueToMs(value: number): number {
  return value <= EPOCH_MS_BOUNDARY
    ? Math.floor(value * 1000)
    : Math.floor(value);
}

/**
 * Age in hours of a node stamped with the canonical `date_added_to_graph`
 * stamp (**epoch milliseconds** — the unit `nowEpochMs()` writes). Used by the
 * intelligence tools' cache-control age check. Kept here, next to the stamp
 * definition, so the canonical unit lives in exactly one module.
 */
export function dateAddedAgeHours(
  dateAddedMs: number,
  now: number = Date.now(),
): number {
  return (now - dateAddedMs) / (3600 * 1000);
}
