import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { stripSystemStamps, buildNodeDedupKey } from "./toolsJarvis.js";

describe("stripSystemStamps", () => {
  it("drops caller-supplied updated_at and date_added_to_graph, keeps the rest", () => {
    const out = stripSystemStamps({ name: "a", updated_at: "2020-01-01T00:00:00Z", date_added_to_graph: 1.5 });
    assert.deepEqual(out, { name: "a" });
  });
  it("does not mutate the input and passes undefined through", () => {
    const i = { name: "a", updated_at: 1 };
    stripSystemStamps(i);
    assert.equal(i.updated_at, 1);
    assert.equal(stripSystemStamps(undefined), undefined);
  });
  it("dedup key ignores stamps", () => {
    assert.equal(buildNodeDedupKey("T", { name: "a", updated_at: "x" }), buildNodeDedupKey("T", { name: "a" }));
  });
});

describe("cypher writers stamp updated_at", () => {
  const q = readFileSync(new URL("../graph/queries.ts", import.meta.url), "utf8");
  it("ADD_NODE_QUERY stamps on create and conditionally on match, before += $properties", () => {
    const m = q.match(/export const ADD_NODE_QUERY[\s\S]*?`;/)![0];
    assert.match(m, /ON CREATE SET[^\n]*n\.updated_at = \$dateAddedToGraph/);
    assert.ok(m.indexOf("n.updated_at = CASE") < m.indexOf("n += $properties", m.indexOf("ON MATCH")));
  });
  it("every MERGE-with-date_added upsert also sets updated_at", () => {
    for (const name of ["CREATE_HINT_QUERY", "CREATE_PROMPT_QUERY", "CREATE_PULL_REQUEST_QUERY", "UPSERT_LEARNING_QUERY", "CREATE_MOCK_QUERY", "UPSERT_AGENT_SESSION_QUERY", "UPSERT_TURNS_QUERY"]) {
      const m = q.match(new RegExp(`export const ${name} = \`[\\s\\S]*?\`;`));
      if (m) assert.match(m[0], /updated_at = \$ts/, name);
    }
  });
  it("derived writes (embeddings, token_count) leave updated_at alone", () => {
    for (const name of ["UPDATE_EMBEDDINGS_QUERY", "BULK_UPDATE_EMBEDDINGS_QUERY", "BULK_UPDATE_TOKEN_COUNT_QUERY", "BULK_UPDATE_EMBEDDINGS_BY_REF_ID_QUERY"]) {
      const m = q.match(new RegExp(`export const ${name} = \`[\\s\\S]*?\`;`))!;
      assert.doesNotMatch(m[0], /updated_at/, name);
    }
  });
});
