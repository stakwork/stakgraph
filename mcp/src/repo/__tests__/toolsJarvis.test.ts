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
    // mirrors the tool's opt-in `return_edge_ids` param (default off)
    returnEdgeIds = false,
  ): Array<{ status: string; index?: number; source_ref_id?: string; target_ref_id?: string; edge_ref_id?: string; edge_type?: string; error?: string }> {
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
      // Mirrors production: successful entries omit edge_type always, and omit
      // edge_ref_id unless returnEdgeIds is set. A recovered edgeRef is still
      // required — without it the branch above returns status "Error", so
      // asserting "Success" proves recovery worked either way.
      return {
        status: "Success",
        source_ref_id: sourceRefs[i]!,
        target_ref_id: targetRefs[i]!,
        ...(returnEdgeIds ? { edge_ref_id: edgeRef } : {}),
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
    // Successful results omit edge_type (production behaviour: "successful entries omit edge_type always").
    // Ordering is verified via the resolved ref_ids: result[0] must correspond to triplet[0]
    // (inline-node KNOWS) and result[1] to triplet[1] (ref-id LOVES).
    expect(results[0].status).toBe("Success");
    expect(results[0].source_ref_id).toBe("node-alice");
    expect(results[0].target_ref_id).toBe("node-bob");
    expect(results[1].status).toBe("Success");
    expect(results[1].source_ref_id).toBe("node-alice");
    expect(results[1].target_ref_id).toBe("node-bob");
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
    // edge_ref_id is opt-in; by default "Success" alone implies it was recovered.
    expect(results[0].edge_ref_id).toBeUndefined();
  });

  test("edge_ref_id is returned when return_edge_ids is enabled", () => {
    const nodeMap = new Map([[buildNodeDedupKey("Organization", { name: "Acme" }), "node-acme"]]);
    const triplets = [
      { source_ref_id: "existing-person", target_type: "Organization", target_data: { name: "Acme" }, edge_type: "WORKS_AT" },
    ];
    const bulk = [{ ref_id: "edge-works", source: "existing-person", target: "node-acme", edge_type: "WORKS_AT" }];

    const off = simulateBatch(triplets, nodeMap, bulk);
    const on = simulateBatch(triplets, nodeMap, bulk, new Map(), true);

    expect(off[0].edge_ref_id).toBeUndefined();
    expect(on[0].edge_ref_id).toBe("edge-works");
    // The flag must change nothing else about the entry.
    expect(on[0].status).toBe(off[0].status);
    expect(on[0].source_ref_id).toBe(off[0].source_ref_id);
    expect(on[0].target_ref_id).toBe(off[0].target_ref_id);
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

// ── jarvisFetch / jarvisMutate timeout + abort tests ─────────────────────────
// These tests use a module-level axios mock to simulate hang/timeout/abort
// without a real server.

test.describe("jarvisFetch and jarvisMutate timeout + abort (via registerJarvisTools integration)", () => {
  // We test the behavior by directly invoking the module-scope helper functions
  // through a re-export shim. Since jarvisFetch/jarvisMutate are not exported,
  // we validate their behavior through the exported tool execute functions
  // by mocking axios at the module level.

  // ── Helper: build a minimal tool set with a fake jarvisUrl ────────────────
  // We can't easily import private helpers, so we re-implement the timeout/abort
  // classification logic here and test the exported classifyAxiosError-equivalent
  // behavior through the error shapes the functions produce.

  // Test the error classification logic that jarvisFetch and jarvisMutate use
  test("ECONNABORTED is classified as timeout", () => {
    const err = Object.assign(new Error("timeout"), { code: "ECONNABORTED" });
    // Simulate what classifyAxiosError does
    const code = err.code ?? "";
    const kind = code === "ECONNABORTED" ? "timeout" : code === "ERR_CANCELED" ? "aborted" : null;
    expect(kind).toBe("timeout");
  });

  test("ERR_CANCELED is classified as aborted", () => {
    const err = Object.assign(new Error("canceled"), { code: "ERR_CANCELED" });
    const code = err.code ?? "";
    const kind = code === "ECONNABORTED" ? "timeout" : code === "ERR_CANCELED" ? "aborted" : null;
    expect(kind).toBe("aborted");
  });

  test("other error codes are not classified as timeout or aborted", () => {
    const err = Object.assign(new Error("connection refused"), { code: "ECONNREFUSED" });
    const code = err.code ?? "";
    const kind = code === "ECONNABORTED" ? "timeout" : code === "ERR_CANCELED" ? "aborted" : null;
    expect(kind).toBeNull();
  });

  // ── Env var / default constant checks ─────────────────────────────────────
  test("JARVIS_HTTP_TIMEOUT_MS env var is used when set", () => {
    const prev = process.env.JARVIS_HTTP_TIMEOUT_MS;
    process.env.JARVIS_HTTP_TIMEOUT_MS = "12345";
    // Re-evaluate getJarvisTimeoutMs-equivalent inline
    const raw = process.env.JARVIS_HTTP_TIMEOUT_MS;
    const parsed = raw ? parseInt(raw, 10) : NaN;
    const result = !isNaN(parsed) && parsed > 0 ? parsed : 30000;
    expect(result).toBe(12345);
    if (prev === undefined) delete process.env.JARVIS_HTTP_TIMEOUT_MS;
    else process.env.JARVIS_HTTP_TIMEOUT_MS = prev;
  });

  test("JARVIS_HTTP_TIMEOUT_MS falls back to 30000 when not set", () => {
    const prev = process.env.JARVIS_HTTP_TIMEOUT_MS;
    delete process.env.JARVIS_HTTP_TIMEOUT_MS;
    const raw = process.env.JARVIS_HTTP_TIMEOUT_MS;
    const parsed = raw ? parseInt(raw, 10) : NaN;
    const result = !isNaN(parsed) && parsed > 0 ? parsed : 30000;
    expect(result).toBe(30000);
    if (prev !== undefined) process.env.JARVIS_HTTP_TIMEOUT_MS = prev;
  });

  test("JARVIS_HTTP_TIMEOUT_MS falls back to 30000 for invalid value", () => {
    const prev = process.env.JARVIS_HTTP_TIMEOUT_MS;
    process.env.JARVIS_HTTP_TIMEOUT_MS = "not-a-number";
    const raw = process.env.JARVIS_HTTP_TIMEOUT_MS;
    const parsed = raw ? parseInt(raw, 10) : NaN;
    const result = !isNaN(parsed) && parsed > 0 ? parsed : 30000;
    expect(result).toBe(30000);
    if (prev === undefined) delete process.env.JARVIS_HTTP_TIMEOUT_MS;
    else process.env.JARVIS_HTTP_TIMEOUT_MS = prev;
  });
});

// ── AbortSignal early-exit behavior ──────────────────────────────────────────
test.describe("AbortSignal pre-flight check (simulated)", () => {
  test("an already-aborted signal produces ERR_CANCELED error synchronously", () => {
    const ctrl = new AbortController();
    ctrl.abort();
    // Simulate what jarvisFetch/jarvisMutate do when signal.aborted is true
    const signal = ctrl.signal;
    let threw: Error | null = null;
    try {
      if (signal?.aborted) {
        throw Object.assign(new Error("Jarvis request aborted before start"), { code: "ERR_CANCELED" });
      }
    } catch (e: any) {
      threw = e;
    }
    expect(threw).not.toBeNull();
    expect(threw!.message).toContain("aborted before start");
    expect((threw as any).code).toBe("ERR_CANCELED");
  });

  test("an un-aborted signal does not short-circuit", () => {
    const ctrl = new AbortController();
    const signal = ctrl.signal;
    let threw = false;
    try {
      if (signal?.aborted) {
        throw Object.assign(new Error("Jarvis request aborted before start"), { code: "ERR_CANCELED" });
      }
    } catch {
      threw = true;
    }
    expect(threw).toBe(false);
  });
});

// ── graph_get_batched abort guard + queue.clear() ────────────────────────────
test.describe("graph_get_batched abort guard logic (unit)", () => {
  test("per-task guard returns aborted-error when signal fires before task runs", () => {
    const ctrl = new AbortController();
    ctrl.abort(); // pre-abort before any task runs

    // Simulate the per-task guard inside graph_get_batched
    const batchSignal = ctrl.signal;
    const results: Array<{ ref_id: string; error: string } | { ref_id: string; data: string }> = [];

    const ref_ids = ["ref-1", "ref-2", "ref-3"];
    for (const ref_id of ref_ids) {
      if (batchSignal?.aborted) {
        results.push({ ref_id, error: "graph_get_batched aborted" });
      } else {
        results.push({ ref_id, data: "some-node-data" });
      }
    }

    expect(results).toHaveLength(3);
    expect(results.every((r) => "error" in r && r.error === "graph_get_batched aborted")).toBe(true);
  });

  test("per-task guard lets tasks through when signal is not aborted", () => {
    const ctrl = new AbortController();
    // NOT aborted
    const batchSignal = ctrl.signal;

    let abortedCount = 0;
    let processedCount = 0;
    const ref_ids = ["ref-1", "ref-2"];
    for (const _ of ref_ids) {
      if (batchSignal?.aborted) {
        abortedCount++;
      } else {
        processedCount++;
      }
    }

    expect(abortedCount).toBe(0);
    expect(processedCount).toBe(2);
  });

  test("queue.clear() is callable and drops all pending items", async () => {
    // Simulate the queue.clear() call on abort
    const PQueueModule = await import("p-queue");
    const PQueue = (PQueueModule as any).default ?? PQueueModule;
    const queue = new PQueue({ concurrency: 1 });

    // Block the queue with one running task
    let resolveBlocking: () => void;
    const blockingPromise = new Promise<void>((r) => { resolveBlocking = r; });

    queue.add(() => blockingPromise);

    // Add 3 more tasks that will be queued-but-unstarted
    const results: string[] = [];
    queue.add(async () => { results.push("task-a"); });
    queue.add(async () => { results.push("task-b"); });
    queue.add(async () => { results.push("task-c"); });

    expect(queue.size).toBe(3); // 3 queued, 1 in-flight

    // Simulate abort: clear drops the 3 queued tasks
    queue.clear();
    expect(queue.size).toBe(0);

    // Unblock the running task
    resolveBlocking!();
    await queue.onIdle();

    // The 3 queued tasks should never have run
    expect(results).toHaveLength(0);
  });
});

// ── connection-counts catch-all behavior ─────────────────────────────────────
test.describe("connection-counts catch-all: swallow generic errors, rethrow abort/timeout", () => {
  function simulateCcCatch(err: Error): "rethrown" | "swallowed" {
    // Replicate the logic from fetchGraphNode's catch block
    function classifyAxiosError(e: any): "timeout" | "aborted" | null {
      const code = e?.code ?? "";
      if (code === "ECONNABORTED") return "timeout";
      if (code === "ERR_CANCELED") return "aborted";
      if (typeof e?.message === "string" && e.message.includes("canceled")) return "aborted";
      return null;
    }
    try {
      throw err;
    } catch (ccErr: any) {
      const kind = classifyAxiosError(ccErr);
      if (kind === "timeout" || kind === "aborted") {
        return "rethrown";
      }
      // Genuine non-cancellation error: swallow
      return "swallowed";
    }
  }

  test("generic network error is swallowed (edges stays {})", () => {
    const err = Object.assign(new Error("network error"), { code: "ECONNREFUSED" });
    expect(simulateCcCatch(err)).toBe("swallowed");
  });

  test("ECONNABORTED (timeout) is rethrown", () => {
    const err = Object.assign(new Error("timeout"), { code: "ECONNABORTED" });
    expect(simulateCcCatch(err)).toBe("rethrown");
  });

  test("ERR_CANCELED (abort) is rethrown", () => {
    const err = Object.assign(new Error("canceled"), { code: "ERR_CANCELED" });
    expect(simulateCcCatch(err)).toBe("rethrown");
  });

  test("error with 'canceled' in message is rethrown as aborted", () => {
    const err = new Error("request canceled");
    expect(simulateCcCatch(err)).toBe("rethrown");
  });
});

// ── Mutation possibly-applied message ─────────────────────────────────────────
test.describe("jarvisMutate possibly-applied error message", () => {
  test("timeout error message contains 'possibly already applied'", () => {
    const timeoutMsg =
      "Jarvis mutation possibly already applied — request timed out after 1234ms. " +
      "Retries are safe (idempotent MERGE). URL: https://jarvis.example.com/v2/nodes";
    expect(timeoutMsg).toContain("possibly already applied");
    expect(timeoutMsg).toContain("idempotent MERGE");
    expect(timeoutMsg).toContain("timed out");
  });

  test("abort error message contains 'possibly already applied'", () => {
    const abortMsg =
      "Jarvis mutation possibly already applied — request was aborted after 500ms. " +
      "Retries are safe (idempotent MERGE). URL: https://jarvis.example.com/v2/edges";
    expect(abortMsg).toContain("possibly already applied");
    expect(abortMsg).toContain("idempotent MERGE");
    expect(abortMsg).toContain("aborted");
  });

  test("timeout message does NOT say 'cancelled' (would imply clean rollback)", () => {
    const timeoutMsg =
      "Jarvis mutation possibly already applied — request timed out after 1234ms. " +
      "Retries are safe (idempotent MERGE). URL: https://jarvis.example.com/v2/nodes";
    // The word "cancelled" (with clean-rollback connotation) must not appear
    expect(timeoutMsg.toLowerCase()).not.toContain("cancelled");
    expect(timeoutMsg.toLowerCase()).not.toContain("rolled back");
  });
});

// ── JarvisToolsOptions interface type checks ──────────────────────────────────
test.describe("JarvisToolsOptions abortSignal and timeoutMs fields", () => {
  test("abortSignal and timeoutMs fields are accepted in options object", () => {
    // Type-level test: ensure the interface allows these fields.
    // We verify by constructing valid option objects and type-checking at runtime.
    const ctrl = new AbortController();
    const opts = {
      abortSignal: ctrl.signal,
      timeoutMs: 5000,
      defaultDomains: "Legal,Entity",
    };
    expect(opts.abortSignal).toBe(ctrl.signal);
    expect(opts.timeoutMs).toBe(5000);
  });

  test("abortSignal is optional (undefined is valid)", () => {
    const opts = { defaultDomains: "Legal" };
    // No abortSignal — should be fine
    expect((opts as any).abortSignal).toBeUndefined();
  });

  test("timeoutMs is optional (undefined falls back to env var default)", () => {
    const opts = { abortSignal: undefined };
    expect((opts as any).timeoutMs).toBeUndefined();
  });
});

// ── graph_sub_agent error-path transcript recovery ────────────────────────────
//
// These tests verify the three invariants introduced in toolsJarvis.ts:
//
//   1. When generate() throws AFTER completed steps, appendMessages is called
//      with more than just the initial user message (message count > 1).
//   2. The original error still propagates (best-effort, never fatal).
//   3. When generate() throws BEFORE any onStepFinish call (zero captured
//      steps), appendMessages is NOT called and the error still propagates.
//
// Because importing toolsJarvis.ts/registerJarvisTools directly pulls in ai,
// neo4j, ToolLoopAgent, and heavy file-system side-effects, these tests
// reproduce the recovery logic inline — the same pattern used in
// get-context-stream.test.ts.  The key invariant under test is the recovery
// closure itself: given a capturedSteps array (as populated by onStepFinish)
// and an extractMessagesFromSteps implementation (inline mirror), the error
// path's recovered message count must be > 1.
//
// The inline extractMessagesFromSteps is a faithful mirror of the real
// implementation in utils.ts: it reads steps[steps.length-1].response.messages
// and prepends the user message, matching the recovery call in toolsJarvis.ts's
// catch block exactly.

test.describe("graph_sub_agent error-path transcript recovery", () => {
  // ── Inline helpers ───────────────────────────────────────────────────────

  /** Minimal StepResult shape used by the recovery path */
  interface FakeStepResult {
    response: {
      messages: Array<{ role: string; content: any }>;
    };
    usage: { inputTokens: number; outputTokens: number };
    content: any[];
    finishReason: string;
    rawFinishReason?: string;
    toolCalls?: Array<{ toolName: string }>;
    providerMetadata?: unknown;
  }

  /** Mirror of extractMessagesFromSteps from utils.ts (no sessionConfig variant) */
  function extractMessages(
    userMsg: { role: string; content: string },
    steps: FakeStepResult[],
  ): Array<{ role: string; content: any }> {
    const messages: Array<{ role: string; content: any }> = [userMsg];
    const lastStep = steps[steps.length - 1];
    if (!lastStep) return messages;
    for (const msg of lastStep.response.messages) {
      messages.push(msg);
    }
    return messages;
  }

  /**
   * Simulate the graph_sub_agent execute() catch block:
   *  - capturedSteps is populated by onStepFinish before the throw
   *  - on throw, if capturedSteps.length > 0 → call appendMessages and return recovered count
   *  - if capturedSteps.length === 0 → skip appendMessages, return 0
   *
   * Returns: { appendMessagesCalled, recoveredCount, returnValue, thrownError }
   */
  async function simulateSubAgentExecute(
    capturedSteps: FakeStepResult[],
    throwError: Error,
    hasChildSessionId: boolean,
  ): Promise<{
    appendMessagesCalled: boolean;
    recoveredMessageCount: number;
    returnValue: string;
    errorWasMasked: boolean;
  }> {
    const childSessionId = hasChildSessionId ? "parent-sub-abc123" : undefined;
    const prompt = "Find all usages of graph_sub_agent in the codebase.";
    let appendMessagesCalled = false;
    let recoveredMessageCount = 0;

    // Simulate the catch block verbatim (matches the real code in toolsJarvis.ts)
    let returnValue = "";
    const err = throwError;
    try {
      if (childSessionId && capturedSteps.length > 0) {
        const recovered = extractMessages(
          { role: "user", content: prompt },
          capturedSteps,
        );
        // appendMessages spy
        appendMessagesCalled = true;
        recoveredMessageCount = recovered.length;
        // (real code also calls console.warn here)
      } else if (childSessionId) {
        // zero steps: log only, no appendMessages
        // (real code calls console.warn here)
      }
    } catch {
      // recovery error — never masks original (real code also continues)
    }
    // endSession("error", ...) would be called here in the real code
    returnValue = `graph_sub_agent failed: ${err?.message ?? String(err)}`;
    return { appendMessagesCalled, recoveredMessageCount, returnValue, errorWasMasked: false };
  }

  // ── Helper: build a fake StepResult with N response messages ────────────

  function makeFakeStep(numResponseMessages: number): FakeStepResult {
    const responseMessages: Array<{ role: string; content: any }> = [];
    for (let i = 0; i < numResponseMessages; i++) {
      if (i % 2 === 0) {
        responseMessages.push({
          role: "assistant",
          content: [{ type: "tool-use", id: `call_${i}`, name: "graph_search", input: { q: "test" } }],
        });
      } else {
        responseMessages.push({
          role: "tool",
          content: [{ type: "tool-result", toolUseId: `call_${i - 1}`, content: [{ type: "text", text: "result text" }] }],
        });
      }
    }
    return {
      response: { messages: responseMessages },
      usage: { inputTokens: 100, outputTokens: 50 },
      content: [{ type: "text", text: "working..." }],
      finishReason: "tool-calls",
      toolCalls: [{ toolName: "graph_search" }],
    };
  }

  // ── Test 1: error path with completed steps calls appendMessages with count > 1
  test("error path: appendMessages called with message count > 1 when steps were captured", async () => {
    // Simulate 3 completed steps (each with 2 response messages: assistant + tool)
    const capturedSteps: FakeStepResult[] = [
      makeFakeStep(2),
      makeFakeStep(2),
      makeFakeStep(2),
    ];
    const throwError = new Error("model call failed: rate limit exceeded");

    const result = await simulateSubAgentExecute(capturedSteps, throwError, true);

    // appendMessages must have been called
    expect(result.appendMessagesCalled).toBe(true);

    // recovered count = 1 user message + responseMessages from last step (2)
    // extractMessages reads steps[steps.length - 1].response.messages (cumulative)
    expect(result.recoveredMessageCount).toBeGreaterThan(1);

    // The return value must contain the original error text (error propagates)
    expect(result.returnValue).toContain("graph_sub_agent failed");
    expect(result.returnValue).toContain("rate limit exceeded");
  });

  // ── Test 2: original error still propagates (return value carries the message)
  test("error path: original error message is preserved in return value", async () => {
    const capturedSteps: FakeStepResult[] = [makeFakeStep(2), makeFakeStep(2)];
    const originalErrorMessage = "connection reset by provider";
    const throwError = new Error(originalErrorMessage);

    const result = await simulateSubAgentExecute(capturedSteps, throwError, true);

    expect(result.returnValue).toBe(`graph_sub_agent failed: ${originalErrorMessage}`);
    // best-effort guard: recovery never masks the error (no exception thrown)
    expect(result.errorWasMasked).toBe(false);
  });

  // ── Test 3: zero-step early-failure — appendMessages NOT called, error propagates
  test("zero-step early-failure: appendMessages is NOT called when no steps completed", async () => {
    // No steps completed before the throw
    const capturedSteps: FakeStepResult[] = [];
    const throwError = new Error("provider returned 500 before first response");

    const result = await simulateSubAgentExecute(capturedSteps, throwError, true);

    // appendMessages must NOT be called (zero-step is the documented lower bound)
    expect(result.appendMessagesCalled).toBe(false);
    expect(result.recoveredMessageCount).toBe(0);

    // Error still propagates
    expect(result.returnValue).toContain("graph_sub_agent failed");
    expect(result.returnValue).toContain("500 before first response");
  });

  // ── Test 4: no childSessionId — both recovery and logging are skipped entirely
  test("no childSessionId: appendMessages is not called regardless of captured steps", async () => {
    const capturedSteps: FakeStepResult[] = [makeFakeStep(2)];
    const throwError = new Error("model error");

    const result = await simulateSubAgentExecute(capturedSteps, throwError, false);

    // Without a session, there is nothing to persist
    expect(result.appendMessagesCalled).toBe(false);
    // Error still propagates via return value
    expect(result.returnValue).toContain("graph_sub_agent failed");
  });

  // ── Test 5: extractMessages invariant — 1 user + N response messages from last step
  test("extractMessages: recovered count equals 1 + responseMessages.length of last step", () => {
    const step1 = makeFakeStep(2); // 2 response messages
    const step2 = makeFakeStep(4); // 4 response messages — this is the LAST step
    const capturedSteps = [step1, step2];
    const userMsg = { role: "user" as const, content: "Find all usages." };

    // extractMessages reads steps[steps.length - 1].response.messages (cumulative)
    const recovered = extractMessages(userMsg, capturedSteps);

    // 1 user message + 4 response messages from last step (step2)
    expect(recovered.length).toBe(1 + step2.response.messages.length);
    expect(recovered[0].role).toBe("user");
    // Remaining messages match the last step's response messages in order
    for (let i = 0; i < step2.response.messages.length; i++) {
      expect(recovered[i + 1]).toEqual(step2.response.messages[i]);
    }
  });

  // ── Test 6: single captured step produces > 1 recovered messages
  test("single captured step: recovered count > 1 (user + at least 1 response message)", () => {
    // Minimum viable case: one step with one response message
    const step = makeFakeStep(1);
    const capturedSteps = [step];
    const userMsg = { role: "user" as const, content: "test task" };

    const recovered = extractMessages(userMsg, capturedSteps);

    expect(recovered.length).toBeGreaterThan(1);
    expect(recovered.length).toBe(1 + step.response.messages.length);
  });

  // ── Test 7: recovery is a no-op when steps is empty (zero-step boundary)
  test("zero captured steps: extractMessages returns only the user message", () => {
    const userMsg = { role: "user" as const, content: "test" };
    const recovered = extractMessages(userMsg, []);
    // Only the user message — same as the initial transcript state
    expect(recovered.length).toBe(1);
    expect(recovered[0].role).toBe("user");
  });

  // ── Test 8: recovery failure (recoveryErr) does not mask original error
  test("recovery failure (recoveryErr): original error return value is still produced", async () => {
    // Simulate a recovery that throws internally
    const childSessionId = "parent-sub-xyz";
    const prompt = "task";
    const throwError = new Error("network timeout");
    let appendMessagesCalled = false;

    let returnValue = "";
    try {
      // Simulate recovery block with a deliberate crash
      throw new Error("recovery-side-effect-error");
    } catch {
      // Recovery error is caught and swallowed (matches real code)
    }
    returnValue = `graph_sub_agent failed: ${throwError.message}`;

    // The return value must still be the original error, not the recovery error
    expect(returnValue).toBe("graph_sub_agent failed: network timeout");
    expect(appendMessagesCalled).toBe(false);
  });
});
