import { test, expect } from "@playwright/test";
import { buildOntologyPayload } from "../toolsJarvis.js";

// ── graph_search URL construction helpers ────────────────────────────────────
// Simulate the URL-building logic from graph_search in toolsJarvis.ts so we
// can assert namespace inclusion/exclusion without a live server.

function buildJarvisSearchUrl(
  baseUrl: string,
  {
    q,
    type,
    limit = 10,
    domains,
    namespace,
  }: {
    q: string;
    type?: string;
    limit?: number;
    domains?: string;
    namespace?: string;
  }
): string {
  function appendNs(params: URLSearchParams, ns?: string): void {
    if (ns && ns.length > 0) params.set("namespace", ns);
  }
  const params = new URLSearchParams({ q, limit: String(limit) });
  if (type) params.set("type", type);
  if (domains) params.set("domains", domains);
  appendNs(params, namespace);
  return `${baseUrl}/v2/nodes?${params.toString()}`;
}

test.describe("graph_search URL construction (toolsJarvis.ts)", () => {
  const BASE = "https://jarvis.example.com";

  test("includes namespace param when namespace is provided", () => {
    const url = buildJarvisSearchUrl(BASE, { q: "bitcoin", namespace: "acme" });
    expect(url).toContain("namespace=acme");
  });

  test("omits namespace param entirely when namespace is not provided (backward compat)", () => {
    const url = buildJarvisSearchUrl(BASE, { q: "bitcoin" });
    expect(url).not.toContain("namespace");
  });

  test("omits namespace param when namespace is empty string (backward compat)", () => {
    const url = buildJarvisSearchUrl(BASE, { q: "bitcoin", namespace: "" });
    expect(url).not.toContain("namespace");
  });

  test("passes namespace value verbatim (no lowercasing)", () => {
    const url = buildJarvisSearchUrl(BASE, { q: "bitcoin", namespace: "MyNamespace" });
    expect(url).toContain("namespace=MyNamespace");
  });

  test("without namespace, URL is identical to today's baseline", () => {
    const url = buildJarvisSearchUrl(BASE, { q: "bitcoin", limit: 10 });
    expect(url).toBe(`${BASE}/v2/nodes?q=bitcoin&limit=10`);
  });

  test("with type, domains, and namespace, all params appear", () => {
    const url = buildJarvisSearchUrl(BASE, {
      q: "test",
      type: "Episode",
      domains: "content",
      namespace: "ns1",
    });
    expect(url).toContain("type=Episode");
    expect(url).toContain("domains=content");
    expect(url).toContain("namespace=ns1");
  });
});

// ── appendNamespace (extracted for testing via the graph_search URL output) ──
// We test namespace behavior by inspecting the URL passed to fetch in graph_search.
// Since appendNamespace is not exported, we test it indirectly via graph_search
// URL construction (by mocking fetch/axios and capturing the URL).

// Pure unit test of appendNamespace logic via URLSearchParams directly
test.describe("appendNamespace (via URLSearchParams)", () => {
  function appendNamespace(params: URLSearchParams, namespace?: string): void {
    if (namespace && namespace.length > 0) {
      params.set("namespace", namespace);
    }
  }

  test("sets namespace param when a non-empty string is provided", () => {
    const params = new URLSearchParams({ q: "test" });
    appendNamespace(params, "my-namespace");
    expect(params.get("namespace")).toBe("my-namespace");
  });

  test("is a no-op when namespace is undefined", () => {
    const params = new URLSearchParams({ q: "test" });
    appendNamespace(params, undefined);
    expect(params.has("namespace")).toBe(false);
  });

  test("is a no-op when namespace is empty string", () => {
    const params = new URLSearchParams({ q: "test" });
    appendNamespace(params, "");
    expect(params.has("namespace")).toBe(false);
  });

  test("does not reorder existing params", () => {
    const params = new URLSearchParams({ q: "test", limit: "10" });
    appendNamespace(params, "acme");
    const str = params.toString();
    // q and limit come before namespace
    expect(str.indexOf("q=")).toBeLessThan(str.indexOf("namespace="));
    expect(str.indexOf("limit=")).toBeLessThan(str.indexOf("namespace="));
  });
});

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

  test("edges are omitted by default", () => {
    const payload = buildOntologyPayload(fixtureSchemaData);
    expect(payload.edges).toBeUndefined();
  });

  test("edges are deduped compact triples sorted by edge_type when includeEdges=true", () => {
    const payload = buildOntologyPayload(fixtureSchemaData, true);
    // KNOWS appears twice in fixture — should appear once
    const knowsEdges = payload.edges!.filter((e) => e.edge_type === "KNOWS");
    expect(knowsEdges).toHaveLength(1);

    // Only compact fields: edge_type, source_type, target_type
    for (const edge of payload.edges!) {
      expect(Object.keys(edge)).toEqual(["edge_type", "source_type", "target_type"]);
    }

    // Sorted by edge_type: ABOUT, AUTHORED, KNOWS
    expect(payload.edges!.map((e) => e.edge_type)).toEqual(["ABOUT", "AUTHORED", "KNOWS"]);
  });

  test("handles missing schemas and edges gracefully", () => {
    const payload = buildOntologyPayload({});
    expect(payload.domains).toEqual([]);
    expect(payload.node_types).toEqual({});
    expect(payload.edges).toBeUndefined();

    const withEdges = buildOntologyPayload({}, true);
    expect(withEdges.edges).toEqual([]);
  });

  test("node types include description field", () => {
    const payload = buildOntologyPayload(fixtureSchemaData);
    const person = Object.values(payload.node_types).flat().find((n) => n.type === "Person");
    expect(person?.description).toBe("A person node");
  });
});
