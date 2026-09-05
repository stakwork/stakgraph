/**
 * Unit tests for `listQueryForLabel`'s `$since` filter with mixed stored
 * timestamp formats.
 *
 * Stored `date_added_to_graph` values may be legacy epoch-seconds or new
 * epoch-ms Integers until the data backfill migration. The generated query
 * therefore normalizes the stored property via `epochMsExpr` (Cypher mirror
 * of `toEpochMs`) before comparing against an epoch-ms `$since`.
 *
 * `nodes_by_type` also normalizes a seconds-magnitude caller `$since` via
 * `epochValueToMs` before binding.
 *
 * Runs under NO_DB=true — no Neo4j contacted.
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { epochMsExpr, listQueryForLabel } from "./queries.js";
import { epochValueToMs, toEpochMs } from "./time.js";

const HOUR_MS = 3600 * 1000;

describe("listQueryForLabel $since filter (mixed timestamp formats)", () => {
  it("normalizes stored date_added_to_graph via epochMsExpr before comparing", () => {
    const q = listQueryForLabel("Function", true);
    assert.match(q, /CASE WHEN toFloat\(f\.date_added_to_graph\) <= 1000000000000/);
    assert.match(q, /f\.date_added_to_graph\) END >= \$since/);
    assert.match(
      q,
      /ORDER BY CASE WHEN toFloat\(coalesce\(f\.date_added_to_graph, 0\)\) <= 1000000000000/,
    );
  });

  it("omits the since clause and ordering entirely when withSince is false", () => {
    const q = listQueryForLabel("Function", false);
    assert.ok(!q.includes("$since"));
    assert.ok(!q.includes("date_added_to_graph"));
    assert.ok(!q.includes("ORDER BY"));
  });

  /**
   * Mirror of the generated clause after both sides are normalized to ms:
   *   `$since IS NULL OR (storedMs IS NOT NULL AND storedMs >= sinceMs)`
   */
  function selects(nodeDate: number | null, sinceMs: number | null): boolean {
    if (sinceMs === null) return true;
    if (nodeDate === null) return false;
    const storedMs = toEpochMs(nodeDate);
    return storedMs !== null && storedMs >= sinceMs;
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

  it("a millisecond $since cursor returns both legacy-seconds and new-ms nodes", () => {
    const now = Date.now();
    const sinceMs = now - 24 * HOUR_MS;
    const legacySeconds = (now - 1 * HOUR_MS) / 1000; // 1h ago, stored as seconds
    const newMs = now - 2 * HOUR_MS; // 2h ago, stored as ms
    const tooOldSeconds = (now - 72 * HOUR_MS) / 1000;

    assert.ok(
      selects(legacySeconds, sinceMs),
      "legacy-seconds node within the window must match a ms cursor"
    );
    assert.ok(
      selects(newMs, sinceMs),
      "new-ms node within the window must match the same cursor"
    );
    assert.ok(
      !selects(tooOldSeconds, sinceMs),
      "legacy-seconds node outside the window must still be excluded"
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

  it("epochMsExpr uses the same <= 1e12 boundary as toEpochMs", () => {
    assert.equal(
      epochMsExpr("f.date_added_to_graph"),
      "CASE WHEN toFloat(f.date_added_to_graph) <= 1000000000000 THEN toFloat(f.date_added_to_graph) * 1000 ELSE toFloat(f.date_added_to_graph) END",
    );
  });
});
