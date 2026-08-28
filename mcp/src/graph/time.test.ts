import { describe, it } from "node:test";
import assert from "node:assert/strict";
import neo4j from "neo4j-driver";

import { nowEpochMs, toEpochMs, nodeAgeHours } from "./time.js";
import {
  ADD_NODE_QUERY,
  CREATE_AGENT_SESSION_STUB_QUERY,
  CREATE_HINT_QUERY,
  CREATE_MOCK_QUERY,
  CREATE_PROMPT_QUERY,
  CREATE_PULL_REQUEST_QUERY,
  GET_ALL_LEARNINGS_WITH_SCOPES_QUERY,
  listQueryForLabel,
  UPDATE_REPO_DOCS_QUERY,
  UPSERT_AGENT_SESSION_QUERY,
  UPSERT_LEARNING_QUERY,
  UPSERT_SCOPE_QUERY,
  UPSERT_TURNS_QUERY,
  UPSERT_WORKFLOW_DOCUMENTATION_QUERY,
} from "./queries.js";
import { toReturnNode, toReturnNodeNoBody } from "./utils.js";
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

describe("toEpochMs (magnitude discriminator)", () => {
  it("treats < 1e12 as legacy epoch-seconds and converts to ms", () => {
    assert.equal(toEpochMs(1700000000), 1700000000000);
    // just under the threshold is still seconds
    assert.equal(toEpochMs(999999999999), 999999999999000);
  });

  it("passes >= 1e12 through unchanged (already ms)", () => {
    assert.equal(toEpochMs(1700000000000), 1700000000000);
    assert.equal(toEpochMs(1e12), 1e12);
  });

  it("parses legacy 7-decimal seconds strings (old Rust ingest format)", () => {
    const ms = toEpochMs("1700000000.1234567");
    assert.ok(ms !== null);
    assert.ok(Math.abs(ms - 1700000000123.4567) < 0.01, `got ${ms}`);
  });

  it("parses ms-magnitude strings", () => {
    assert.equal(toEpochMs("1700000000000"), 1700000000000);
  });

  it("handles raw Neo4j Integer objects ({low, high})", () => {
    const int = neo4j.int(1700000000000);
    assert.equal(toEpochMs(int), 1700000000000);
  });

  it("returns null for null/undefined/empty/unparseable input", () => {
    assert.equal(toEpochMs(null), null);
    assert.equal(toEpochMs(undefined), null);
    assert.equal(toEpochMs(""), null);
    assert.equal(toEpochMs("not-a-timestamp"), null);
    assert.equal(toEpochMs({} as any), null);
  });
});

describe("nodeAgeHours (cache-age over mixed stored formats)", () => {
  // Fixed "now" in ms (~2027) so expectations are exact.
  const NOW_MS = 1_800_000_000_000;

  it("ages a legacy-seconds stored value correctly", () => {
    // stored 2h ago as epoch-seconds
    assert.equal(nodeAgeHours((NOW_MS - 2 * 3600_000) / 1000, NOW_MS), 2);
  });

  it("ages a legacy 7-decimal seconds string correctly", () => {
    const stored = ((NOW_MS - 3 * 3600_000) / 1000).toFixed(7);
    assert.equal(nodeAgeHours(stored, NOW_MS), 3);
  });

  it("ages a new epoch-ms stored value correctly", () => {
    assert.equal(nodeAgeHours(NOW_MS - 3600_000, NOW_MS), 1);
  });

  it("returns null for missing/unparseable values", () => {
    assert.equal(nodeAgeHours(null, NOW_MS), null);
    assert.equal(nodeAgeHours(undefined, NOW_MS), null);
    assert.equal(nodeAgeHours("garbage", NOW_MS), null);
  });
});

describe("listQueryForLabel delta filter (ms cursor over mixed stored formats)", () => {
  it("normalizes the stored value before the $since comparison", () => {
    const q = listQueryForLabel("Hint", true);
    assert.match(
      q,
      /CASE WHEN toFloat\(f\.date_added_to_graph\) >= 1000000000000/,
    );
    assert.match(q, /toFloat\(f\.date_added_to_graph\) \* 1000 END >= \$since/);
    // the legacy bare-seconds comparison must be gone
    assert.ok(
      !q.includes("toFloat(f.date_added_to_graph) >= $since"),
      "legacy comparison would silently drop all legacy-seconds nodes for a ms cursor",
    );
  });

  it("normalizes the paired ORDER BY too", () => {
    const q = listQueryForLabel("Hint", true);
    assert.match(
      q,
      /ORDER BY CASE WHEN toFloat\(coalesce\(f\.date_added_to_graph, 0\)\) >= 1000000000000/,
    );
    assert.match(q, /DESC, f\.node_key/);
    assert.ok(!q.includes("coalesce(toFloat(f.date_added_to_graph), 0)"));
  });

  it("omits since clauses when withSince=false", () => {
    const q = listQueryForLabel("Hint", false);
    assert.ok(!q.includes("$since"));
    assert.ok(!q.includes("ORDER BY"));
  });

  it("normalizes the learnings ORDER BY (mixed-format sort)", () => {
    assert.match(
      GET_ALL_LEARNINGS_WITH_SCOPES_QUERY,
      /ORDER BY CASE WHEN toFloat\(l\.date_added_to_graph\) >= 1000000000000/,
    );
  });
});

describe("API responses never leak raw Neo4j Integer objects", () => {
  const int = neo4j.int(1700000000000);
  const rawNode = {
    labels: ["Data_Bank", "Hint"],
    properties: {
      ref_id: "r1",
      name: "n",
      body: "b",
      date_added_to_graph: int,
    },
  } as any;

  it("toReturnNode coerces {low, high} to a plain number", () => {
    const ret = toReturnNode(rawNode);
    assert.equal(ret.date_added_to_graph, 1700000000000);
    assert.equal(typeof ret.date_added_to_graph, "number");
    assert.equal(typeof ret.properties.date_added_to_graph, "number");
    const json = JSON.stringify(ret);
    assert.ok(!json.includes('"low"'), `leaked Integer object: ${json}`);
  });

  it("toReturnNodeNoBody coerces too", () => {
    const ret = toReturnNodeNoBody(rawNode);
    assert.equal(ret.date_added_to_graph, 1700000000000);
    const json = JSON.stringify(ret);
    assert.ok(!json.includes('"low"'), `leaked Integer object: ${json}`);
  });

  it("is idempotent for already-plain number values", () => {
    const ret = toReturnNode({
      ...rawNode,
      properties: { ...rawNode.properties, date_added_to_graph: 1700000000000 },
    });
    assert.equal(ret.date_added_to_graph, 1700000000000);
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
