import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { graphConfigFromEnv } from "./bolt.js";
import { DEFAULT_NEO4J_HOST, graphWorkspaceRequested } from "./wiring.js";

describe("graph workspace wiring defaults", () => {
  it("VEIN_WORKSPACE_BACKEND=graph is the switch (case-insensitive)", () => {
    assert.equal(graphWorkspaceRequested({}), false);
    assert.equal(graphWorkspaceRequested({ VEIN_WORKSPACE_BACKEND: "Graph" }), true);
  });
  it("connection defaults match the mcp host: localhost:7687 / neo4j / testtest; env wins", () => {
    const d = graphConfigFromEnv({ NEO4J_HOST: DEFAULT_NEO4J_HOST })!;
    assert.deepEqual([d.uri, d.user, d.password, d.namespace], ["bolt://localhost:7687", "neo4j", "testtest", "default"]);
    const e = graphConfigFromEnv({ NEO4J_HOST: DEFAULT_NEO4J_HOST, NEO4J_URI: "bolt://db:7688", NEO4J_PASSWORD: "pw" })!;
    assert.deepEqual([e.uri, e.password], ["bolt://db:7688", "pw"]);
  });
});
