import { test, expect } from "@playwright/test";
import { buildOntologyPayload } from "../toolsJarvis.js";

const fixtureSchemaData = {
  schemas: [
    { type: "Person", domain: "Entity", description: "A person node", is_deleted: false },
    { type: "Episode", domain: "Content", description: "A podcast episode", is_deleted: false },
    { type: "Topic", domain: "Entity", description: "A topic node", is_deleted: false },
    { type: "Workflow", domain: "Workflow", description: "A workflow node", is_deleted: false },
    { type: "Orphan", domain: null, description: "No domain node", is_deleted: false },
    { type: "NoDomainField", description: "Missing domain field entirely", is_deleted: false },
    // should be excluded: is_deleted
    { type: "DeletedType", domain: "Entity", description: "deleted", is_deleted: true },
    // should be excluded: type === "*"
    { type: "*", domain: "Entity", description: "wildcard", is_deleted: false },
  ],
  edges: [
    { edge_type: "KNOWS", source_type: "Person", target_type: "Person", extra: "ignored" },
    { edge_type: "ABOUT", source_type: "Episode", target_type: "Topic", extra: "ignored" },
    // duplicate triple — should be deduped
    { edge_type: "KNOWS", source_type: "Person", target_type: "Person" },
    // another edge — sorts before KNOWS
    { edge_type: "AUTHORED", source_type: "Person", target_type: "Episode" },
  ],
};

test.describe("buildOntologyPayload", () => {
  test("excludes type='*' and is_deleted entries", () => {
    const payload = buildOntologyPayload(fixtureSchemaData);
    const allTypes = Object.values(payload.node_types).flat().map((n) => n.type);
    expect(allTypes).not.toContain("*");
    expect(allTypes).not.toContain("DeletedType");
  });

  test("includes non-deleted, non-wildcard types", () => {
    const payload = buildOntologyPayload(fixtureSchemaData);
    const allTypes = Object.values(payload.node_types).flat().map((n) => n.type);
    expect(allTypes).toContain("Person");
    expect(allTypes).toContain("Episode");
    expect(allTypes).toContain("Topic");
    expect(allTypes).toContain("Workflow");
  });

  test("lowercases domain on each node type", () => {
    const payload = buildOntologyPayload(fixtureSchemaData);
    const person = Object.values(payload.node_types).flat().find((n) => n.type === "Person");
    expect(person?.domain).toBe("entity");

    const episode = Object.values(payload.node_types).flat().find((n) => n.type === "Episode");
    expect(episode?.domain).toBe("content");
  });

  test("domains list is distinct, non-null, lowercased, and sorted", () => {
    const payload = buildOntologyPayload(fixtureSchemaData);
    expect(payload.domains).toEqual(["content", "entity", "workflow"]);
    // sorted
    expect([...payload.domains]).toEqual([...payload.domains].sort());
    // no nulls
    expect(payload.domains.every((d) => d !== null)).toBe(true);
    // distinct
    expect(new Set(payload.domains).size).toBe(payload.domains.length);
  });

  test("null-domain types land in 'ungrouped' bucket and are absent from domains", () => {
    const payload = buildOntologyPayload(fixtureSchemaData);
    expect(payload.node_types["ungrouped"]).toBeDefined();
    const ungroupedTypes = payload.node_types["ungrouped"].map((n) => n.type);
    expect(ungroupedTypes).toContain("Orphan");
    expect(ungroupedTypes).toContain("NoDomainField");

    // ungrouped types must not appear in domains list
    expect(payload.domains).not.toContain(null);
    expect(payload.domains).not.toContain("ungrouped");
  });

  test("null-domain node type has domain: null", () => {
    const payload = buildOntologyPayload(fixtureSchemaData);
    const orphan = payload.node_types["ungrouped"].find((n) => n.type === "Orphan");
    expect(orphan?.domain).toBeNull();
  });

  test("edges are deduped compact triples sorted by edge_type", () => {
    const payload = buildOntologyPayload(fixtureSchemaData);
    // KNOWS appears twice in fixture — should appear once
    const knowsEdges = payload.edges.filter((e) => e.edge_type === "KNOWS");
    expect(knowsEdges).toHaveLength(1);

    // Only compact fields: edge_type, source_type, target_type
    for (const edge of payload.edges) {
      expect(Object.keys(edge)).toEqual(["edge_type", "source_type", "target_type"]);
    }

    // Sorted by edge_type: ABOUT, AUTHORED, KNOWS
    expect(payload.edges.map((e) => e.edge_type)).toEqual(["ABOUT", "AUTHORED", "KNOWS"]);
  });

  test("handles missing schemas and edges gracefully", () => {
    const payload = buildOntologyPayload({});
    expect(payload.domains).toEqual([]);
    expect(payload.node_types).toEqual({});
    expect(payload.edges).toEqual([]);
  });

  test("node types include description field", () => {
    const payload = buildOntologyPayload(fixtureSchemaData);
    const person = Object.values(payload.node_types).flat().find((n) => n.type === "Person");
    expect(person?.description).toBe("A person node");
  });
});
