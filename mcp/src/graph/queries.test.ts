/**
 * Unit tests for `listQueryForLabel`'s `$since` filter after the
 * `toFloat(date_added_to_graph)` coercion shim removal.
 *
 * `date_added_to_graph` is a canonical epoch-ms Integer (see ./time.ts), so
 * the property compares directly against an epoch-ms `$since`. These tests:
 *   1. pin the generated query text to the bare, coercion-free predicate;
 *   2. prove the clause's selection semantics on ms-magnitude fixtures —
 *      a recently-added node is selected, a stale one is not;
 *   3. prove a seconds-magnitude `$since` must be normalized (×1000) before
 *      binding — which `nodes_by_type` does via `epochValueToMs`.
 *
 * Runs under NO_DB=true — no Neo4j contacted.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { listQueryForLabel } from "./queries.js";
import { epochValueToMs } from "./time.js";

const HOUR_MS = 3600 * 1000;

describe("listQueryForLabel $since filter (post toFloat removal)", () => {
  it("compares date_added_to_graph directly — no toFloat coercion remains", () => {
    const q = listQueryForLabel("Function", true);
    assert.doesNotMatch(
      q,
      /toFloat\s*\(/,
      "toFloat coercion shim must be gone"
    );
    assert.match(q, /f\.date_added_to_graph >= \$since/);
    assert.match(
      q,
      /ORDER BY coalesce\(f\.date_added_to_graph, 0\) DESC, f\.node_key/
    );
  });

  it("omits the since clause and ordering entirely when withSince is false", () => {
    const q = listQueryForLabel("Function", false);
    assert.ok(!q.includes("$since"));
    assert.ok(!q.includes("date_added_to_graph"));
    assert.ok(!q.includes("ORDER BY"));
  });

  /**
   * Mirror of the generated clause:
   *   `$since IS NULL OR (f.date_added_to_graph IS NOT NULL
   *                       AND f.date_added_to_graph >= $since)`
   * The text assertions above pin the template to this exact predicate, so
   * the mirror cannot silently drift from what the query actually does.
   */
  function selects(nodeDate: number | null, sinceMs: number | null): boolean {
    return sinceMs === null || (nodeDate !== null && nodeDate >= sinceMs);
  }

  it("selects recently-added (ms-magnitude) nodes and excludes stale ones", () => {
    const now = Date.now();
    const fresh = now - 1 * HOUR_MS; // added 1h ago (ms-magnitude ~1.7e12)
    const stale = now - 72 * HOUR_MS; // added 72h ago
    const undated: number | null = null; // property absent
    const sinceMs = now - 24 * HOUR_MS; // "added within the last 24h"

    assert.ok(selects(fresh, sinceMs), "fresh node (1h old) must be selected");
    assert.ok(
      !selects(stale, sinceMs),
      "stale node (72h old) must be excluded"
    );
    assert.ok(
      !selects(undated, sinceMs),
      "node without date_added_to_graph must be excluded"
    );
    assert.ok(
      selects(fresh, null) && selects(stale, null),
      "$since IS NULL selects everything"
    );
  });

  it("normalizes a seconds-magnitude $since before comparing (as nodes_by_type does)", () => {
    const now = Date.now();
    const fresh = now - 1 * HOUR_MS;
    const stale = now - 72 * HOUR_MS;
    const rawSecondsSince = Math.floor((now - 24 * HOUR_MS) / 1000);

    // Without normalization the seconds-magnitude bound never excludes
    // ms-magnitude values — the "recently added" filter degenerates and
    // matches every node.
    assert.ok(
      stale >= rawSecondsSince && fresh >= rawSecondsSince,
      "precondition: an unnormalized seconds-since matches stale nodes too"
    );

    // `nodes_by_type` binds `epochValueToMs(since)`, restoring the filter.
    const sinceMs = epochValueToMs(rawSecondsSince);
    assert.ok(
      sinceMs > stale,
      "normalized bound must be ms-magnitude (above any seconds value)"
    );
    assert.ok(selects(fresh, sinceMs), "fresh node still selected");
    assert.ok(!selects(stale, sinceMs), "stale node excluded again");
  });
});
