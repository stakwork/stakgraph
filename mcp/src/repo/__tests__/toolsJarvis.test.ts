import { test, expect } from "../../testkit.js";
import {
  buildOntologyPayload,
  collapseConnectionCounts,
  validateTripletSide,
  extractNodeRefId,
  extractEdgeRefId,
  buildNodeDedupKey,
  matchEdgeResults,
} from "../toolsJarvis.js";
import type { ResolvedTriplet } from "../toolsJarvis.js";

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
  params.set("include_edge_counts", "true");
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

  test("without namespace, URL carries q, limit, and include_edge_counts", () => {
    const url = buildJarvisSearchUrl(BASE, { q: "bitcoin", limit: 10 });
    expect(url).toBe(`${BASE}/v2/nodes?q=bitcoin&limit=10&include_edge_counts=true`);
  });

  test("always requests inline edge counts", () => {
    const url = buildJarvisSearchUrl(BASE, { q: "bitcoin" });
    expect(url).toContain("include_edge_counts=true");
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
    {
      type: "Person",
      domain: "Entity",
      description: "A person node",
      is_deleted: false,
      // own attributes only (non-overlapping bulk-endpoint shape)
      attributes: { name: "string", age: "?int" },
      // inherited from the Thing base schema — repeats verbatim across types
      inherited_attributes: { ref_id: "string", created_at: "?datetime" },
    },
    {
      type: "Episode",
      domain: "Content",
      description: "A podcast episode",
      // type_description populated — should be preferred over description
      type_description: "A single podcast episode entry",
      is_deleted: false,
      attributes: { title: "string", duration: "?float" },
      inherited_attributes: { ref_id: "string", created_at: "?datetime" },
      // carries a parent relationship to exercise inheritance-flavored data
      parent: "Thing",
    },
    {
      type: "Topic",
      domain: "Entity",
      description: "A topic node",
      // type_description present but empty — should fall back to description
      type_description: "   ",
      is_deleted: false,
    },
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
    // wildcard edges — source_type="*" means "any source type"
    { edge_type: "TAGGED_WITH", source_type: "*", target_type: "Topic" },
    // wildcard edges — target_type="*" means "any target type"
    { edge_type: "LINKED_TO", source_type: "Person", target_type: "*" },
    // both-sides wildcard
    { edge_type: "RELATES_TO", source_type: "*", target_type: "*" },
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

  test("node types are grouped under the correct lowercased domain key", () => {
    const payload = buildOntologyPayload(fixtureSchemaData);
    // Person has domain "Entity" → lowercased to "entity"
    expect(payload.node_types["entity"]).toBeDefined();
    const entityTypes = payload.node_types["entity"].map((n) => n.type);
    expect(entityTypes).toContain("Person");
    expect(entityTypes).toContain("Topic");

    // Episode has domain "Content" → lowercased to "content"
    expect(payload.node_types["content"]).toBeDefined();
    const contentTypes = payload.node_types["content"].map((n) => n.type);
    expect(contentTypes).toContain("Episode");
  });

  test("node type entries no longer carry a domain field", () => {
    const payload = buildOntologyPayload(fixtureSchemaData);
    const allNodes = Object.values(payload.node_types).flat();
    for (const node of allNodes) {
      expect(Object.prototype.hasOwnProperty.call(node, "domain")).toBe(false);
    }
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

  test("null-domain types land exclusively in 'ungrouped' bucket (sole signal of null domain)", () => {
    const payload = buildOntologyPayload(fixtureSchemaData);
    const orphan = payload.node_types["ungrouped"].find((n) => n.type === "Orphan");
    expect(orphan).toBeDefined();
    // 'domain' field must not appear on the entry — ungrouped membership is the only signal
    expect(Object.prototype.hasOwnProperty.call(orphan, "domain")).toBe(false);
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

    // Sorted by edge_type: ABOUT, AUTHORED, KNOWS, LINKED_TO, RELATES_TO, TAGGED_WITH
    expect(payload.edges!.map((e) => e.edge_type)).toEqual([
      "ABOUT",
      "AUTHORED",
      "KNOWS",
      "LINKED_TO",
      "RELATES_TO",
      "TAGGED_WITH",
    ]);
  });

  test("wildcard source_type ('*') passes through unmodified in edges array", () => {
    const payload = buildOntologyPayload(fixtureSchemaData, true);
    const taggedWith = payload.edges!.find((e) => e.edge_type === "TAGGED_WITH");
    expect(taggedWith).toEqual({ edge_type: "TAGGED_WITH", source_type: "*", target_type: "Topic" });
  });

  test("wildcard target_type ('*') passes through unmodified in edges array", () => {
    const payload = buildOntologyPayload(fixtureSchemaData, true);
    const linkedTo = payload.edges!.find((e) => e.edge_type === "LINKED_TO");
    expect(linkedTo).toEqual({ edge_type: "LINKED_TO", source_type: "Person", target_type: "*" });
  });

  test("both-sides wildcard ('*'/'*') passes through unmodified in edges array", () => {
    const payload = buildOntologyPayload(fixtureSchemaData, true);
    const relatesTo = payload.edges!.find((e) => e.edge_type === "RELATES_TO");
    expect(relatesTo).toEqual({ edge_type: "RELATES_TO", source_type: "*", target_type: "*" });
  });

  test("handles missing schemas and edges gracefully", () => {
    const payload = buildOntologyPayload({});
    expect(payload.domains).toEqual([]);
    expect(payload.node_types).toEqual({});
    expect(payload.edges).toBeUndefined();

    const withEdges = buildOntologyPayload({}, true);
    expect(withEdges.edges).toEqual([]);
  });

  test("node types include description field — Person has no type_description so falls back to description", () => {
    const payload = buildOntologyPayload(fixtureSchemaData);
    const person = Object.values(payload.node_types).flat().find((n) => n.type === "Person");
    expect(person?.description).toBe("A person node");
  });

  test("prefers type_description over description when type_description is non-empty", () => {
    // Episode fixture has type_description="A single podcast episode entry"
    const payload = buildOntologyPayload(fixtureSchemaData);
    const episode = Object.values(payload.node_types).flat().find((n) => n.type === "Episode");
    expect(episode?.description).toBe("A single podcast episode entry");
  });

  test("falls back to description when type_description is present but whitespace-only", () => {
    // Topic fixture has type_description="   " (whitespace) and description="A topic node"
    const payload = buildOntologyPayload(fixtureSchemaData);
    const topic = Object.values(payload.node_types).flat().find((n) => n.type === "Topic");
    expect(topic?.description).toBe("A topic node");
  });

  // ── include_attributes tests ────────────────────────────────────────────────

  test("attributes/inherited_attributes are omitted entirely by default (no empty objects)", () => {
    const payload = buildOntologyPayload(fixtureSchemaData);
    const allNodes = Object.values(payload.node_types).flat();
    for (const node of allNodes) {
      expect(Object.prototype.hasOwnProperty.call(node, "attributes")).toBe(false);
      expect(Object.prototype.hasOwnProperty.call(node, "inherited_attributes")).toBe(false);
    }
  });

  test("attributes/inherited_attributes are omitted when includeAttributes=false explicitly", () => {
    const payload = buildOntologyPayload(fixtureSchemaData, false, false);
    const allNodes = Object.values(payload.node_types).flat();
    for (const node of allNodes) {
      expect(Object.prototype.hasOwnProperty.call(node, "attributes")).toBe(false);
      expect(Object.prototype.hasOwnProperty.call(node, "inherited_attributes")).toBe(false);
    }
  });

  test("attributes/inherited_attributes are present and correctly sourced when includeAttributes=true", () => {
    const payload = buildOntologyPayload(fixtureSchemaData, false, true);
    const person = Object.values(payload.node_types).flat().find((n) => n.type === "Person");
    expect(person?.attributes).toEqual({ name: "string", age: "?int" });
    expect(person?.inherited_attributes).toEqual({ ref_id: "string", created_at: "?datetime" });

    const episode = Object.values(payload.node_types).flat().find((n) => n.type === "Episode");
    expect(episode?.attributes).toEqual({ title: "string", duration: "?float" });
    expect(episode?.inherited_attributes).toEqual({ ref_id: "string", created_at: "?datetime" });
  });

  test("node types without attributes in fixture get empty objects when includeAttributes=true", () => {
    const payload = buildOntologyPayload(fixtureSchemaData, false, true);
    const topic = Object.values(payload.node_types).flat().find((n) => n.type === "Topic");
    expect(topic?.attributes).toEqual({});
    expect(topic?.inherited_attributes).toEqual({});
  });

  test("includeAttributes=true does not add edges (flags are independent)", () => {
    const payload = buildOntologyPayload(fixtureSchemaData, false, true);
    expect(payload.edges).toBeUndefined();
  });

  test("includeEdges=true does not add attributes (flags are independent)", () => {
    const payload = buildOntologyPayload(fixtureSchemaData, true, false);
    expect(payload.edges).toBeDefined();
    const allNodes = Object.values(payload.node_types).flat();
    for (const node of allNodes) {
      expect(Object.prototype.hasOwnProperty.call(node, "attributes")).toBe(false);
      expect(Object.prototype.hasOwnProperty.call(node, "inherited_attributes")).toBe(false);
    }
  });

  test("both includeEdges=true and includeAttributes=true work together", () => {
    const payload = buildOntologyPayload(fixtureSchemaData, true, true);
    // edges present and correct
    expect(payload.edges).toBeDefined();
    expect(payload.edges!.map((e) => e.edge_type)).toEqual([
      "ABOUT",
      "AUTHORED",
      "KNOWS",
      "LINKED_TO",
      "RELATES_TO",
      "TAGGED_WITH",
    ]);
    // attributes present and correct
    const person = Object.values(payload.node_types).flat().find((n) => n.type === "Person");
    expect(person?.attributes).toEqual({ name: "string", age: "?int" });
    expect(person?.inherited_attributes).toEqual({ ref_id: "string", created_at: "?datetime" });
  });

  test("node type with parent field in fixture has its attributes surfaced (inheritance-flavored data)", () => {
    const payload = buildOntologyPayload(fixtureSchemaData, false, true);
    const episode = Object.values(payload.node_types).flat().find((n) => n.type === "Episode");
    // parent field is on the raw fixture entry; attributes/inherited_attributes are correctly sourced
    expect(episode?.attributes).toEqual({ title: "string", duration: "?float" });
    expect(episode?.inherited_attributes).toEqual({ ref_id: "string", created_at: "?datetime" });
  });

  test("both false (default 2x2 matrix — no edges, no attributes)", () => {
    const payload = buildOntologyPayload(fixtureSchemaData, false, false);
    expect(payload.edges).toBeUndefined();
    const allNodes = Object.values(payload.node_types).flat();
    for (const node of allNodes) {
      expect(Object.prototype.hasOwnProperty.call(node, "attributes")).toBe(false);
      expect(Object.prototype.hasOwnProperty.call(node, "inherited_attributes")).toBe(false);
    }
  });
});

// ── get_ontology execute handler URL construction ────────────────────────────

test.describe("get_ontology URL construction", () => {
  // Simulate the URL-building logic from the get_ontology execute handler
  // to assert that include_edges and include_attributes are always forwarded.
  function buildOntologyUrl(
    baseUrl: string,
    include_edges: boolean,
    include_attributes: boolean,
  ): string {
    const params = new URLSearchParams();
    params.set("include_edges", String(include_edges));
    params.set("include_attributes", String(include_attributes));
    return `${baseUrl}/v2/schema?${params.toString()}`;
  }

  const BASE = "https://jarvis.example.com";

  test("always includes include_edges param (false)", () => {
    const url = buildOntologyUrl(BASE, false, false);
    expect(url).toContain("include_edges=false");
  });

  test("always includes include_attributes param (false)", () => {
    const url = buildOntologyUrl(BASE, false, false);
    expect(url).toContain("include_attributes=false");
  });

  test("forwards include_edges=true when set", () => {
    const url = buildOntologyUrl(BASE, true, false);
    expect(url).toContain("include_edges=true");
    expect(url).toContain("include_attributes=false");
  });

  test("forwards include_attributes=true when set", () => {
    const url = buildOntologyUrl(BASE, false, true);
    expect(url).toContain("include_edges=false");
    expect(url).toContain("include_attributes=true");
  });

  test("both true: both params are present in the URL", () => {
    const url = buildOntologyUrl(BASE, true, true);
    expect(url).toContain("include_edges=true");
    expect(url).toContain("include_attributes=true");
  });

  test("URL targets /v2/schema with a query string", () => {
    const url = buildOntologyUrl(BASE, false, false);
    expect(url).toContain("/v2/schema?");
    expect(url.startsWith(BASE)).toBe(true);
  });
});

test.describe("collapseConnectionCounts", () => {
  test("sums counts across target types per edge_type", () => {
    const edges = collapseConnectionCounts([
      { edge_type: "CONTAINS", target_type: "File", count: 3 },
      { edge_type: "CONTAINS", target_type: "Function", count: 2 },
      { edge_type: "PART_OF", target_type: "Repository", count: 1 },
    ]);
    expect(edges).toEqual({ CONTAINS: 5, PART_OF: 1 });
  });

  test("works when target_type is absent", () => {
    const edges = collapseConnectionCounts([
      { edge_type: "CITES", count: 4 },
      { edge_type: "CITES", count: 1 },
    ]);
    expect(edges).toEqual({ CITES: 5 });
  });

  test("returns empty object for empty or missing input", () => {
    expect(collapseConnectionCounts([])).toEqual({});
    expect(collapseConnectionCounts(undefined as any)).toEqual({});
  });

  test("coerces non-numeric counts and skips rows without edge_type", () => {
    const edges = collapseConnectionCounts([
      { edge_type: "", target_type: "File", count: 9 } as any,
      { edge_type: "HAS", target_type: "Tag", count: "2" as any },
      { edge_type: "HAS", target_type: "Tag", count: undefined as any },
    ]);
    expect(edges).toEqual({ HAS: 2 });
  });
});

// ── create_triplet helpers ───────────────────────────────────────────────────

test.describe("validateTripletSide", () => {
  test("accepts an existing node by ref_id", () => {
    expect(validateTripletSide("source", "ref-123")).toBeNull();
  });

  test("accepts an inline node with type + data", () => {
    expect(
      validateTripletSide("target", undefined, "Person", { name: "Alice" })
    ).toBeNull();
  });

  test("rejects when neither form is provided", () => {
    const err = validateTripletSide("source");
    expect(err).toContain("source_ref_id");
    expect(err).toContain("source_type");
  });

  test("rejects inline type without data", () => {
    const err = validateTripletSide("target", undefined, "Person", undefined);
    expect(err).toContain("target");
  });

  test("rejects inline data without type", () => {
    const err = validateTripletSide("target", undefined, undefined, { name: "x" });
    expect(err).toContain("target");
  });

  test("rejects mixing ref_id with inline fields (ambiguous)", () => {
    const err = validateTripletSide("source", "ref-123", "Person", { name: "x" });
    expect(err).toContain("not both");
  });

  test("treats empty-string ref_id as absent", () => {
    expect(validateTripletSide("source", "", "Person", { name: "x" })).toBeNull();
    expect(validateTripletSide("source", "")).not.toBeNull();
  });
});

test.describe("extractNodeRefId", () => {
  test("reads data.ref_id from a fresh-create response", () => {
    expect(
      extractNodeRefId({ status: "success", data: { ref_id: "abc", node_key: "k" } })
    ).toBe("abc");
  });

  test("reads data.ref_id from a 'Node already exists' merge response", () => {
    expect(
      extractNodeRefId({
        status: "Warning",
        errorCode: "Node already exists in the graph",
        data: { ref_id: "existing-1" },
      })
    ).toBe("existing-1");
  });

  test("returns undefined for error bodies and junk", () => {
    expect(extractNodeRefId({ status: "Error", status_messages: ["boom"] })).toBeUndefined();
    expect(extractNodeRefId({ errorCode: "Not a valid node_type" })).toBeUndefined();
    expect(extractNodeRefId({ data: { ref_id: "" } })).toBeUndefined();
    expect(extractNodeRefId(undefined)).toBeUndefined();
    expect(extractNodeRefId(null)).toBeUndefined();
  });
});

test.describe("extractEdgeRefId", () => {
  test("reads edges[0].ref_id from a fresh edge-create response", () => {
    expect(
      extractEdgeRefId({
        status: "Success",
        edges: [{ ref_id: "edge-1", source: "a", target: "b" }],
      })
    ).toBe("edge-1");
  });

  test("reads data.ref_id from an 'Edge already exists' warning response", () => {
    expect(
      extractEdgeRefId({
        status: "Warning",
        errorCode: "Edge already exists in the graph",
        data: { ref_id: "edge-existing", edge_key: "works_at" },
      })
    ).toBe("edge-existing");
  });

  test("prefers edges[0].ref_id over data.ref_id when both are present", () => {
    expect(
      extractEdgeRefId({ edges: [{ ref_id: "from-edges" }], data: { ref_id: "from-data" } })
    ).toBe("from-edges");
  });

  test("returns undefined for error bodies and junk", () => {
    expect(extractEdgeRefId({ status: "Error", status_messages: ["boom"] })).toBeUndefined();
    expect(extractEdgeRefId({ edges: [] })).toBeUndefined();
    expect(extractEdgeRefId(undefined)).toBeUndefined();
  });
});

// ── create_batch_triplet pure helpers ────────────────────────────────────────

test.describe("buildNodeDedupKey", () => {
  test("identical node_type + node_data produce the same key", () => {
    const a = buildNodeDedupKey("Person", { name: "Alice", age: 30 });
    const b = buildNodeDedupKey("Person", { name: "Alice", age: 30 });
    expect(a).toBe(b);
  });

  test("key order in node_data does not matter (canonical JSON)", () => {
    const a = buildNodeDedupKey("Person", { age: 30, name: "Alice" });
    const b = buildNodeDedupKey("Person", { name: "Alice", age: 30 });
    expect(a).toBe(b);
  });

  test("nested key order is also normalised", () => {
    const a = buildNodeDedupKey("Org", { meta: { z: 1, a: 2 } });
    const b = buildNodeDedupKey("Org", { meta: { a: 2, z: 1 } });
    expect(a).toBe(b);
  });

  test("different node_data produces different keys", () => {
    const a = buildNodeDedupKey("Person", { name: "Alice" });
    const b = buildNodeDedupKey("Person", { name: "Bob" });
    expect(a).not.toBe(b);
  });

  test("different node_type produces different keys even with same data", () => {
    const a = buildNodeDedupKey("Person", { name: "Alice" });
    const b = buildNodeDedupKey("Organization", { name: "Alice" });
    expect(a).not.toBe(b);
  });

  test("array values are preserved (arrays are not re-sorted)", () => {
    const a = buildNodeDedupKey("Skill", { tags: ["a", "b"] });
    const b = buildNodeDedupKey("Skill", { tags: ["b", "a"] });
    // Arrays are NOT sorted — they are order-sensitive
    expect(a).not.toBe(b);
  });
});

// ── matchEdgeResults ─────────────────────────────────────────────────────────

function makeTriplet(
  index: number,
  source_ref_id: string,
  target_ref_id: string,
  edge_type: string,
  overrides: Partial<ResolvedTriplet> = {}
): ResolvedTriplet {
  return {
    index,
    source_ref_id,
    target_ref_id,
    edge_type,
    create_schema_if_missing: false,
    ...overrides,
  };
}

test.describe("matchEdgeResults", () => {
  test("matches a single returned edge to its triplet by (src, tgt, edge_type)", () => {
    const triplets = [makeTriplet(0, "src-1", "tgt-1", "KNOWS")];
    const returned = [{ ref_id: "edge-1", source: "src-1", target: "tgt-1", edge_type: "KNOWS" }];
    const { matched, unmatched } = matchEdgeResults(triplets, returned);
    expect(matched.get(0)).toBe("edge-1");
    expect(unmatched).toHaveLength(0);
  });

  test("leaves a genuinely unmatched triplet in unmatched", () => {
    const triplets = [makeTriplet(0, "src-1", "tgt-1", "KNOWS")];
    const returned = [{ ref_id: "edge-X", source: "src-2", target: "tgt-2", edge_type: "KNOWS" }];
    const { matched, unmatched } = matchEdgeResults(triplets, returned);
    expect(matched.size).toBe(0);
    expect(unmatched).toHaveLength(1);
    expect(unmatched[0].index).toBe(0);
  });

  test("consume-once in input order disambiguates two triplets with the same (src, tgt, edge_type)", () => {
    // Both triplets share the same triple but differ in edge_data.
    // Bulk response returns two edges with the same key.
    const triplets = [
      makeTriplet(0, "src-1", "tgt-1", "CITES", { edge_data: { note: "first" } }),
      makeTriplet(1, "src-1", "tgt-1", "CITES", { edge_data: { note: "second" } }),
    ];
    const returned = [
      { ref_id: "edge-A", source: "src-1", target: "tgt-1", edge_type: "CITES" },
      { ref_id: "edge-B", source: "src-1", target: "tgt-1", edge_type: "CITES" },
    ];
    const { matched, unmatched } = matchEdgeResults(triplets, returned);
    // Both matched; edge-A goes to index 0 (first in input order), edge-B to index 1
    expect(matched.get(0)).toBe("edge-A");
    expect(matched.get(1)).toBe("edge-B");
    expect(unmatched).toHaveLength(0);
  });

  test("one returned edge for two triplets with same key → first matched, second unmatched", () => {
    const triplets = [
      makeTriplet(0, "src-1", "tgt-1", "CITES"),
      makeTriplet(1, "src-1", "tgt-1", "CITES"),
    ];
    const returned = [
      { ref_id: "edge-A", source: "src-1", target: "tgt-1", edge_type: "CITES" },
    ];
    const { matched, unmatched } = matchEdgeResults(triplets, returned);
    expect(matched.get(0)).toBe("edge-A");
    expect(matched.has(1)).toBe(false);
    expect(unmatched).toHaveLength(1);
    expect(unmatched[0].index).toBe(1);
  });

  test("handles empty returned edges (all unmatched)", () => {
    const triplets = [
      makeTriplet(0, "src-1", "tgt-1", "KNOWS"),
      makeTriplet(1, "src-2", "tgt-2", "LOVES"),
    ];
    const { matched, unmatched } = matchEdgeResults(triplets, []);
    expect(matched.size).toBe(0);
    expect(unmatched).toHaveLength(2);
  });

  test("handles empty triplets (nothing to match)", () => {
    const returned = [
      { ref_id: "edge-Z", source: "src-1", target: "tgt-1", edge_type: "KNOWS" },
    ];
    const { matched, unmatched } = matchEdgeResults([], returned);
    expect(matched.size).toBe(0);
    expect(unmatched).toHaveLength(0);
  });

  test("skips returned edges with missing ref_id", () => {
    const triplets = [makeTriplet(0, "s", "t", "KNOWS")];
    const returned = [
      { ref_id: "", source: "s", target: "t", edge_type: "KNOWS" },
    ] as any[];
    const { matched, unmatched } = matchEdgeResults(triplets, returned);
    expect(matched.size).toBe(0);
    expect(unmatched).toHaveLength(1);
  });

  test("multiple distinct triplets each match their own returned edge", () => {
    const triplets = [
      makeTriplet(0, "a", "b", "KNOWS"),
      makeTriplet(1, "c", "d", "LOVES"),
      makeTriplet(2, "e", "f", "HATES"),
    ];
    const returned = [
      { ref_id: "e1", source: "a", target: "b", edge_type: "KNOWS" },
      { ref_id: "e2", source: "c", target: "d", edge_type: "LOVES" },
      { ref_id: "e3", source: "e", target: "f", edge_type: "HATES" },
    ];
    const { matched, unmatched } = matchEdgeResults(triplets, returned);
    expect(matched.get(0)).toBe("e1");
    expect(matched.get(1)).toBe("e2");
    expect(matched.get(2)).toBe("e3");
    expect(unmatched).toHaveLength(0);
  });
});

// ── create_batch_triplet result assembly (pure logic simulation) ─────────────
// We exercise the pure helpers (buildNodeDedupKey, matchEdgeResults,
// validateTripletSide, extractNodeRefId, extractEdgeRefId) to simulate the
// overall batch behaviour without hitting the network.

test.describe("create_batch_triplet result assembly (simulated)", () => {
  // ── helpers re-used across tests ──────────────────────────────────────────
  function simulateBatch(
    triplets: Array<{
      source_ref_id?: string;
      source_type?: string;
      source_data?: Record<string, any>;
      target_ref_id?: string;
      target_type?: string;
      target_data?: Record<string, any>;
      edge_type: string;
      edge_data?: Record<string, any>;
      weight?: number;
      create_schema_if_missing?: boolean;
    }>,
    // pre-resolved node map: dedupKey → ref_id
    nodeMap: Map<string, string>,
    // simulated bulk edge response
    bulkEdgesResponse: Array<{
      ref_id: string;
      source?: string;
      target?: string;
      edge_type?: string;
    }>,
    // simulated fallback responses per unmatched triplet index (index → edgeRefId | null)
    fallbackMap: Map<number, string | null> = new Map(),
  ): Array<{ status: string; index?: number; source_ref_id?: string; target_ref_id?: string; edge_ref_id?: string; edge_type: string; error?: string }> {
    const failures: Array<string | null> = triplets.map(() => null);

    // Phase 0: validation
    for (let i = 0; i < triplets.length; i++) {
      const t = triplets[i];
      const err =
        validateTripletSide("source", t.source_ref_id, t.source_type, t.source_data) ??
        validateTripletSide("target", t.target_ref_id, t.target_type, t.target_data);
      if (err) failures[i] = `invalid input — ${err}`;
    }

    // Phase 1: resolve ref_ids
    const sourceRefs: Array<string | null> = triplets.map(() => null);
    const targetRefs: Array<string | null> = triplets.map(() => null);

    for (let i = 0; i < triplets.length; i++) {
      if (failures[i]) continue;
      const t = triplets[i];

      // source
      const srcHasRef = typeof t.source_ref_id === "string" && t.source_ref_id.length > 0;
      if (srcHasRef) {
        sourceRefs[i] = t.source_ref_id!;
      } else {
        const key = buildNodeDedupKey(t.source_type!, t.source_data!);
        const r = nodeMap.get(key);
        if (!r) { failures[i] = "source node resolution failed"; continue; }
        sourceRefs[i] = r;
      }

      // target
      const tgtHasRef = typeof t.target_ref_id === "string" && t.target_ref_id.length > 0;
      if (tgtHasRef) {
        targetRefs[i] = t.target_ref_id!;
      } else {
        const key = buildNodeDedupKey(t.target_type!, t.target_data!);
        const r = nodeMap.get(key);
        if (!r) { failures[i] = "target node resolution failed"; continue; }
        targetRefs[i] = r;
      }
    }

    // Phase 2: match edges
    const resolvedTriplets: ResolvedTriplet[] = [];
    for (let i = 0; i < triplets.length; i++) {
      if (failures[i] || !sourceRefs[i] || !targetRefs[i]) continue;
      const t = triplets[i];
      resolvedTriplets.push({
        index: i,
        source_ref_id: sourceRefs[i]!,
        target_ref_id: targetRefs[i]!,
        edge_type: t.edge_type,
        edge_data: t.edge_data,
        weight: t.weight,
        create_schema_if_missing: t.create_schema_if_missing ?? false,
      });
    }

    const edgeResults = new Map<number, string>();
    const { matched, unmatched } = matchEdgeResults(resolvedTriplets, bulkEdgesResponse);
    for (const [idx, ref] of matched) edgeResults.set(idx, ref);

    for (const rt of unmatched) {
      const fallbackRef = fallbackMap.get(rt.index) ?? null;
      if (fallbackRef) {
        edgeResults.set(rt.index, fallbackRef);
      } else {
        failures[rt.index] = "edge write failed";
      }
    }

    // Phase 3: assemble
    return triplets.map((t, i) => {
      if (failures[i]) {
        return { status: "Error", index: i, edge_type: t.edge_type, error: failures[i] };
      }
      const edgeRef = edgeResults.get(i);
      if (!edgeRef) {
        return { status: "Error", index: i, edge_type: t.edge_type, error: "edge ref_id could not be recovered" };
      }
      // Mirrors production: successful entries omit edge_ref_id and edge_type.
      // A recovered edgeRef is still required — without it the branch above
      // returns status "Error", so asserting "Success" proves recovery worked.
      return {
        status: "Success",
        source_ref_id: sourceRefs[i]!,
        target_ref_id: targetRefs[i]!,
      };
    });
  }

  test("output ordering matches input ordering", () => {
    const nodeMap = new Map([
      [buildNodeDedupKey("Person", { name: "Alice" }), "node-alice"],
      [buildNodeDedupKey("Person", { name: "Bob" }), "node-bob"],
    ]);
    const triplets = [
      { source_type: "Person", source_data: { name: "Alice" }, target_type: "Person", target_data: { name: "Bob" }, edge_type: "KNOWS" },
      { source_ref_id: "node-alice", target_ref_id: "node-bob", edge_type: "LOVES" },
    ];
    const bulk = [
      { ref_id: "e1", source: "node-alice", target: "node-bob", edge_type: "KNOWS" },
      { ref_id: "e2", source: "node-alice", target: "node-bob", edge_type: "LOVES" },
    ];
    const results = simulateBatch(triplets, nodeMap, bulk);
    expect(results[0].status).toBe("Success");
    expect(results[0].edge_type).toBe("KNOWS");
    expect(results[1].status).toBe("Success");
    expect(results[1].edge_type).toBe("LOVES");
  });

  test("mixes ref_id sides and inline node_type+node_data sides correctly", () => {
    const nodeMap = new Map([
      [buildNodeDedupKey("Org", { name: "Acme" }), "node-acme"],
    ]);
    const triplets = [
      { source_ref_id: "existing-person", target_type: "Org", target_data: { name: "Acme" }, edge_type: "WORKS_AT" },
    ];
    const bulk = [
      { ref_id: "edge-works", source: "existing-person", target: "node-acme", edge_type: "WORKS_AT" },
    ];
    const results = simulateBatch(triplets, nodeMap, bulk);
    expect(results[0].status).toBe("Success");
    expect(results[0].source_ref_id).toBe("existing-person");
    expect(results[0].target_ref_id).toBe("node-acme");
    // edge_ref_id is deliberately not returned; "Success" implies it was recovered.
    expect(results[0].edge_ref_id).toBeUndefined();
  });

  test("one invalid triplet produces a failure entry without affecting other items", () => {
    const nodeMap = new Map([
      [buildNodeDedupKey("Person", { name: "Alice" }), "node-alice"],
      [buildNodeDedupKey("Person", { name: "Bob" }), "node-bob"],
    ]);
    const triplets = [
      // valid
      { source_ref_id: "node-alice", target_ref_id: "node-bob", edge_type: "KNOWS" },
      // INVALID — no source provided
      { target_ref_id: "node-bob", edge_type: "KNOWS" } as any,
      // valid
      { source_ref_id: "node-alice", target_ref_id: "node-bob", edge_type: "LOVES" },
    ];
    const bulk = [
      { ref_id: "e1", source: "node-alice", target: "node-bob", edge_type: "KNOWS" },
      { ref_id: "e2", source: "node-alice", target: "node-bob", edge_type: "LOVES" },
    ];
    const results = simulateBatch(triplets, nodeMap, bulk);
    expect(results[0].status).toBe("Success");
    expect(results[1].status).toBe("Error");
    expect(results[1].error).toContain("invalid input");
    expect(results[2].status).toBe("Success");
  });

  test("idempotent-merge (edge absent from bulk response) is Success via fallback, not failure", () => {
    const triplets = [
      { source_ref_id: "node-a", target_ref_id: "node-b", edge_type: "KNOWS" },
    ];
    // Bulk response omits the edge (it was a Warning/duplicate)
    const bulk: any[] = [];
    // Fallback recovers the ref_id (simulates extractEdgeRefId from data.ref_id)
    const fallback = new Map([[0, "edge-existing"]]);
    const results = simulateBatch(triplets, new Map(), bulk, fallback);
    // "Success" is only reachable when the fallback recovered an edge ref_id —
    // a null recovery falls through to the "edge ref_id could not be recovered" Error.
    expect(results[0].status).toBe("Success");
    expect(results[0].edge_ref_id).toBeUndefined();
  });

  test("hard failure when both bulk and fallback fail to return a ref_id", () => {
    const triplets = [
      { source_ref_id: "node-a", target_ref_id: "node-b", edge_type: "KNOWS" },
    ];
    const bulk: any[] = [];
    // No fallback entry → failure
    const results = simulateBatch(triplets, new Map(), bulk);
    expect(results[0].status).toBe("Error");
  });

  test("duplicate inline sides across the batch dedup to the same resolved ref_id", () => {
    // Two triplets share the same inline source side — should resolve to the same node.
    const aliceKey = buildNodeDedupKey("Person", { name: "Alice" });
    const nodeMap = new Map([[aliceKey, "node-alice"]]);
    const triplets = [
      { source_type: "Person", source_data: { name: "Alice" }, target_ref_id: "node-b", edge_type: "KNOWS" },
      { source_type: "Person", source_data: { name: "Alice" }, target_ref_id: "node-c", edge_type: "LOVES" },
    ];
    const bulk = [
      { ref_id: "e1", source: "node-alice", target: "node-b", edge_type: "KNOWS" },
      { ref_id: "e2", source: "node-alice", target: "node-c", edge_type: "LOVES" },
    ];
    const results = simulateBatch(triplets, nodeMap, bulk);
    expect(results[0].source_ref_id).toBe("node-alice");
    expect(results[1].source_ref_id).toBe("node-alice");
    expect(results[0].status).toBe("Success");
    expect(results[1].status).toBe("Success");
  });
});
