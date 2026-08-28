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
