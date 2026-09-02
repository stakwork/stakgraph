import { readFileSync } from "node:fs";
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import neo4j from "neo4j-driver";

import { nowEpochMs, epochValueToMs, dateAddedAgeHours } from "./time.js";
import {
  ADD_NODE_QUERY,
  CREATE_AGENT_SESSION_STUB_QUERY,
  CREATE_HINT_QUERY,
  CREATE_MOCK_QUERY,
  CREATE_PROMPT_QUERY,
  CREATE_PULL_REQUEST_QUERY,
  UPDATE_REPO_DOCS_QUERY,
  UPSERT_AGENT_SESSION_QUERY,
  UPSERT_LEARNING_QUERY,
  UPSERT_SCOPE_QUERY,
  UPSERT_TURNS_QUERY,
  UPSERT_WORKFLOW_DOCUMENTATION_QUERY,
} from "./queries.js";
import {
  prepareAgeNode,
  prepareCommitsNode,
  prepareContributorNode,
  prepareGitHubRepoNode,
  prepareIssuesNode,
  prepareStarsNode,
} from "./gitsee-nodes.js";

describe("nowEpochMs", () => {
  it("returns a Neo4j Integer, not a Float or string", () => {
    const ts = nowEpochMs();
    assert.ok(neo4j.isInt(ts), "expected a neo4j Integer (isInt === true)");
  });

  it("returns epoch milliseconds within a sane delta of Date.now()", () => {
    const before = Date.now();
    const ts = nowEpochMs().toNumber();
    const after = Date.now();
    assert.ok(
      ts >= before - 5 && ts <= after + 5,
      `expected ~${before}..${after}, got ${ts}`,
    );
  });

  it("is millisecond magnitude (>= 1e12), not seconds", () => {
    assert.ok(nowEpochMs().toNumber() >= 1e12);
  });
});

describe("date_added_to_graph set-once semantics (query templates)", () => {
  /**
   * `date_added_to_graph` is "set once on create": every Cypher line that
   * assigns it must be an `ON CREATE` branch or a `COALESCE(...)` (first
   * write wins). Anything else re-stamps the value on every merge/update.
   * DB-free structural guard — the binder attaches params by query text, so
   * the template text itself is the contract.
   */
  function assertSetOnce(name: string, query: string) {
    const offenders = query
      .split("\n")
      .filter((line) => line.includes("date_added_to_graph"))
      .filter(
        (line) => !/ON CREATE/i.test(line) && !/COALESCE\s*\(/i.test(line),
      );
    assert.deepEqual(
      offenders,
      [],
      `${name} assigns date_added_to_graph outside ON CREATE/COALESCE:\n${offenders.join("\n")}`,
    );
  }

  it("CREATE_HINT_QUERY is set-once", () =>
    assertSetOnce("CREATE_HINT_QUERY", CREATE_HINT_QUERY));
  it("CREATE_PROMPT_QUERY is set-once", () =>
    assertSetOnce("CREATE_PROMPT_QUERY", CREATE_PROMPT_QUERY));
  it("CREATE_PULL_REQUEST_QUERY is set-once", () =>
    assertSetOnce("CREATE_PULL_REQUEST_QUERY", CREATE_PULL_REQUEST_QUERY));
  it("UPSERT_LEARNING_QUERY is set-once", () =>
    assertSetOnce("UPSERT_LEARNING_QUERY", UPSERT_LEARNING_QUERY));
  it("UPSERT_SCOPE_QUERY is set-once", () =>
    assertSetOnce("UPSERT_SCOPE_QUERY", UPSERT_SCOPE_QUERY));
  it("CREATE_MOCK_QUERY is set-once", () =>
    assertSetOnce("CREATE_MOCK_QUERY", CREATE_MOCK_QUERY));
  it("CREATE_AGENT_SESSION_STUB_QUERY is set-once", () =>
    assertSetOnce(
      "CREATE_AGENT_SESSION_STUB_QUERY",
      CREATE_AGENT_SESSION_STUB_QUERY,
    ));
  it("UPSERT_AGENT_SESSION_QUERY is set-once", () =>
    assertSetOnce("UPSERT_AGENT_SESSION_QUERY", UPSERT_AGENT_SESSION_QUERY));
  it("UPSERT_TURNS_QUERY is set-once", () =>
    assertSetOnce("UPSERT_TURNS_QUERY", UPSERT_TURNS_QUERY));
  it("UPSERT_WORKFLOW_DOCUMENTATION_QUERY is set-once", () =>
    assertSetOnce(
      "UPSERT_WORKFLOW_DOCUMENTATION_QUERY",
      UPSERT_WORKFLOW_DOCUMENTATION_QUERY,
    ));
  it("ADD_NODE_QUERY (gitsee path) is set-once", () => {
    const q = ADD_NODE_QUERY("GitHubRepo");
    assertSetOnce("ADD_NODE_QUERY", q);
    // The ON CREATE branch stamps via the dedicated param, not via $properties
    assert.match(q, /n\.date_added_to_graph = \$dateAddedToGraph/);
  });

  it("UPDATE_REPO_DOCS_QUERY never touches date_added_to_graph", () => {
    assert.ok(
      !UPDATE_REPO_DOCS_QUERY.includes("date_added_to_graph"),
      "repo docs updates must not re-stamp the creation timestamp",
    );
    assert.match(UPDATE_REPO_DOCS_QUERY, /SET r\.documentation = \$documentation/);
  });
});

describe("gitsee node prep leaves timestamping to the write path", () => {
  const builders: Array<[string, () => { node_data: Record<string, unknown> }]> =
    [
      ["repo", () => prepareGitHubRepoNode({ id: "o/r", name: "o/r" })],
      [
        "contributor",
        () =>
          prepareContributorNode({
            id: 1,
            login: "u",
            avatar_url: "",
            contributions: 1,
          }),
      ],
      ["stars", () => prepareStarsNode(5, "o/r")],
      ["commits", () => prepareCommitsNode(5, "o/r")],
      ["age", () => prepareAgeNode(1.5, "o/r")],
      ["issues", () => prepareIssuesNode(3, "o/r")],
    ];

  for (const [name, build] of builders) {
    it(`${name} node_data has no inline date_added_to_graph`, () => {
      const { node_data } = build();
      assert.equal(
        node_data.date_added_to_graph,
        undefined,
        "timestamp is owned by db.add_node (nowEpochMs, ON CREATE only)",
      );
    });
  }
});

describe("epochValueToMs", () => {
  it("scales seconds-magnitude values ×1000", () => {
    assert.equal(epochValueToMs(1_700_000_000), 1_700_000_000_000);
  });

  it("treats exactly 10**12 as still-seconds (matches TimeFormatter.epoch_value_to_ms boundary)", () => {
    assert.equal(epochValueToMs(1e12), 1e15);
  });

  it("passes ms-magnitude values through without double-scaling", () => {
    const ms = 1_700_000_000_123; // already canonical epoch-ms
    assert.equal(epochValueToMs(ms), ms);
  });

  it("truncates sub-ms precision like the backend int() conversion", () => {
    // 7-decimal string seconds (legacy shape) round-trips to ms via floor
    assert.equal(epochValueToMs(1700000000.1234567), 1700000000123);
  });

  it("accepts a nowEpochMs() stamp unchanged", () => {
    const ts = nowEpochMs().toNumber();
    assert.equal(epochValueToMs(ts), ts);
  });
});

describe("dateAddedAgeHours (intelligence cache-control age math)", () => {
  const HOUR_MS = 3600 * 1000;

  it("computes age in hours from a canonical ms-magnitude stamp", () => {
    const stamp = 1_700_000_000_000; // canonical epoch-ms Integer
    const now = stamp + 3 * HOUR_MS;
    assert.equal(dateAddedAgeHours(stamp, now), 3);
  });

  it("a fresh node (1h old) is within a 24h maxAgeHours window", () => {
    const now = Date.now();
    const stamp = now - 1 * HOUR_MS;
    assert.ok(dateAddedAgeHours(stamp, now) < 24);
  });

  it("a stale node (72h old) exceeds a 24h maxAgeHours window", () => {
    const now = Date.now();
    const stamp = now - 72 * HOUR_MS;
    assert.ok(dateAddedAgeHours(stamp, now) > 24);
  });

  it("defaults `now` to Date.now()", () => {
    const stamp = Date.now() - 2 * HOUR_MS;
    const hours = dateAddedAgeHours(stamp);
    assert.ok(hours > 1.9 && hours < 2.1, `expected ~2h, got ${hours}`);
  });
});

describe("intelligence cache-control regression guard (source scan)", () => {
  /**
   * The cache-control branch reads `date_added_to_graph` without going
   * through any toInteger/toFloat shim, so a shim-only grep would miss a
   * regression to the old seconds-assuming math (`Date.now() / 1000`,
   * `... / 3600`). Scan the source directly to pin the ms semantics.
   */
  it("reads date_added_to_graph as epoch-ms via dateAddedAgeHours", () => {
    const src = readFileSync(
      new URL("../tools/intelligence/index.ts", import.meta.url),
      "utf8"
    );
    assert.ok(
      src.includes("dateAddedAgeHours(nodeAge)"),
      "cache branch must use the canonical dateAddedAgeHours helper"
    );
    assert.ok(
      !src.includes("Date.now() / 1000"),
      "seconds-assuming currentTime must not return"
    );
    assert.ok(
      !src.includes("/ 3600"),
      "hours conversion must go through the helper, not a seconds-based divide"
    );
  });
});
