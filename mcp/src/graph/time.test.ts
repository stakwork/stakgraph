import { describe, it } from "node:test";
import assert from "node:assert/strict";
import neo4j from "neo4j-driver";

import { nowEpochMs } from "./time.js";
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
