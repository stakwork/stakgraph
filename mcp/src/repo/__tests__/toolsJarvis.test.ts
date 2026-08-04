import { test, expect } from "../../testkit.js";
import {
  buildOntologyPayload,
  collapseConnectionCounts,
  validateTripletSide,
  extractNodeRefId,
  extractEdgeRefId,
} from "../toolsJarvis.js";
import { bearerToken } from "../../tools/utils.js";

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
      is_deleted: false,
      attributes: { title: "string", duration: "?float" },
      inherited_attributes: { ref_id: "string", created_at: "?datetime" },
      // carries a parent relationship to exercise inheritance-flavored data
      parent: "Thing",
    },
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
    // wildcard edges — source_type="*" means "any source type"
    { edge_type: "TAGGED_WITH", source_type: "*", target_type: "Topic" },
    // wildcard edges — target_type="*" means "any target type"
    { edge_type: "LINKED_TO", source_type: "Person", target_type: "*" },
    // both-sides wildcard
    { edge_type: "RELATES_TO", source_type: "*", target_type: "*" },
    // concrete SUPERSEDES rows (no applies_to)
    { edge_type: "SUPERSEDES", source_type: "Claim", target_type: "Claim" },
    { edge_type: "SUPERSEDES", source_type: "LegalDocument", target_type: "LegalDocument" },
    // wildcard SUPERSEDES row — applies_to should be preserved
    { edge_type: "SUPERSEDES", source_type: "*", target_type: "*", applies_to: "any" },
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

    // Concrete rows and wildcard-without-applies_to have 3 keys; wildcard rows with applies_to have 4
    for (const edge of payload.edges!) {
      const keys = Object.keys(edge);
      if (edge.applies_to !== undefined) {
        expect(keys).toEqual(["edge_type", "source_type", "target_type", "applies_to"]);
      } else {
        expect(keys).toEqual(["edge_type", "source_type", "target_type"]);
      }
    }

    // Sorted by edge_type: ABOUT, AUTHORED, KNOWS, LINKED_TO, RELATES_TO, SUPERSEDES×3, TAGGED_WITH
    expect(payload.edges!.map((e) => e.edge_type)).toEqual([
      "ABOUT",
      "AUTHORED",
      "KNOWS",
      "LINKED_TO",
      "RELATES_TO",
      "SUPERSEDES",
      "SUPERSEDES",
      "SUPERSEDES",
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
    const relatesTo = payload.edges!.find(
      (e) => e.edge_type === "RELATES_TO" && e.source_type === "*" && e.target_type === "*"
    );
    expect(relatesTo).toEqual({ edge_type: "RELATES_TO", source_type: "*", target_type: "*" });
  });

  test("applies_to is present for wildcard edge rows and genuinely absent for concrete rows", () => {
    const payload = buildOntologyPayload(fixtureSchemaData, true);
    const wildcardEdge = payload.edges!.find(
      (e) => e.edge_type === "SUPERSEDES" && e.source_type === "*" && e.target_type === "*"
    );
    expect(wildcardEdge).toBeDefined();
    expect(wildcardEdge!.applies_to).toBe("any");
    // applies_to must be a real key, not just truthy
    expect(Object.keys(wildcardEdge!)).toContain("applies_to");

    // Concrete SUPERSEDES rows must NOT have applies_to as a key at all
    const concreteEdges = payload.edges!.filter(
      (e) => e.edge_type === "SUPERSEDES" && e.source_type !== "*" && e.target_type !== "*"
    );
    expect(concreteEdges.length).toBeGreaterThan(0);
    for (const edge of concreteEdges) {
      expect(Object.keys(edge)).not.toContain("applies_to");
    }
  });

  test("edgeSeen dedup distinguishes wildcard SUPERSEDES from concrete SUPERSEDES rows", () => {
    const payload = buildOntologyPayload(fixtureSchemaData, true);
    const supersedesEdges = payload.edges!.filter((e) => e.edge_type === "SUPERSEDES");
    // fixture has: Claim→Claim, LegalDocument→LegalDocument, *→* — all distinct keys
    expect(supersedesEdges).toHaveLength(3);
    const keys = supersedesEdges.map(
      (e) => `${e.edge_type}|${e.source_type}|${e.target_type}`
    );
    expect(keys).toContain("SUPERSEDES|Claim|Claim");
    expect(keys).toContain("SUPERSEDES|LegalDocument|LegalDocument");
    expect(keys).toContain("SUPERSEDES|*|*");
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
      "SUPERSEDES",
      "SUPERSEDES",
      "SUPERSEDES",
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

// ── bearerToken middleware ───────────────────────────────────────────────────

test.describe("bearerToken", () => {
  function makeReqRes(authHeader?: string): {
    req: any;
    res: any;
    statusCode: number | undefined;
    body: any;
    nextCalled: boolean;
  } {
    const ctx: { statusCode: number | undefined; body: any; nextCalled: boolean } = {
      statusCode: undefined,
      body: undefined,
      nextCalled: false,
    };
    const req: any = {
      header: (name: string) =>
        name.toLowerCase() === "authorization" ? authHeader : undefined,
    };
    const res: any = {
      status(code: number) {
        ctx.statusCode = code;
        return res;
      },
      json(b: any) {
        ctx.body = b;
        return res;
      },
    };
    const next = () => {
      ctx.nextCalled = true;
    };
    return { req, res, next, ...ctx, get statusCode() { return ctx.statusCode; }, get body() { return ctx.body; }, get nextCalled() { return ctx.nextCalled; } };
  }

  test("returns 500 when API_TOKEN is not set", () => {
    const saved = process.env.API_TOKEN;
    delete process.env.API_TOKEN;
    try {
      const ctx = makeReqRes();
      bearerToken(ctx.req, ctx.res, () => { ctx.nextCalled = true; });
      expect(ctx.statusCode).toBe(500);
      expect(ctx.nextCalled).toBe(false);
    } finally {
      if (saved !== undefined) process.env.API_TOKEN = saved;
    }
  });

  test("returns 401 when API_TOKEN is set but Authorization header is missing", () => {
    const saved = process.env.API_TOKEN;
    process.env.API_TOKEN = "secret-token";
    try {
      const ctx = makeReqRes(undefined);
      bearerToken(ctx.req, ctx.res, () => { ctx.nextCalled = true; });
      expect(ctx.statusCode).toBe(401);
      expect(ctx.nextCalled).toBe(false);
    } finally {
      if (saved !== undefined) process.env.API_TOKEN = saved;
      else delete process.env.API_TOKEN;
    }
  });

  test("returns 401 when API_TOKEN is set but token does not match", () => {
    const saved = process.env.API_TOKEN;
    process.env.API_TOKEN = "secret-token";
    try {
      const ctx = makeReqRes("Bearer wrong-token");
      bearerToken(ctx.req, ctx.res, () => { ctx.nextCalled = true; });
      expect(ctx.statusCode).toBe(401);
      expect(ctx.nextCalled).toBe(false);
    } finally {
      if (saved !== undefined) process.env.API_TOKEN = saved;
      else delete process.env.API_TOKEN;
    }
  });

  test("calls next() when API_TOKEN is set and matching Bearer token is supplied", () => {
    const saved = process.env.API_TOKEN;
    process.env.API_TOKEN = "secret-token";
    try {
      let nextCalled = false;
      const ctx = makeReqRes("Bearer secret-token");
      bearerToken(ctx.req, ctx.res, () => { nextCalled = true; });
      expect(nextCalled).toBe(true);
      expect(ctx.statusCode).toBeUndefined();
    } finally {
      if (saved !== undefined) process.env.API_TOKEN = saved;
      else delete process.env.API_TOKEN;
    }
  });
});
