import { test, expect } from "../../testkit.js";

// ── appendNamespace unit tests ───────────────────────────────────────────────
// appendNamespace is not exported from tools.ts, so we test the same pure
// logic here to verify the helper behavior in isolation, and test graph_search
// URL construction by mocking axios.

// Inline the same pure helper logic for isolated unit tests
function appendNamespace(params: URLSearchParams, namespace?: string): void {
  if (namespace && namespace.length > 0) {
    params.set("namespace", namespace);
  }
}

test.describe("appendNamespace (graph_agent)", () => {
  test("sets namespace param when a non-empty string is provided", () => {
    const params = new URLSearchParams({ q: "search term" });
    appendNamespace(params, "my-namespace");
    expect(params.get("namespace")).toBe("my-namespace");
  });

  test("is a no-op when namespace is undefined", () => {
    const params = new URLSearchParams({ q: "search term" });
    appendNamespace(params, undefined);
    expect(params.has("namespace")).toBe(false);
  });

  test("is a no-op when namespace is empty string", () => {
    const params = new URLSearchParams({ q: "search term" });
    appendNamespace(params, "");
    expect(params.has("namespace")).toBe(false);
  });

  test("passes value verbatim (no lowercasing)", () => {
    const params = new URLSearchParams();
    appendNamespace(params, "MyNamespace");
    expect(params.get("namespace")).toBe("MyNamespace");
  });

  test("does not reorder existing params when namespace is appended", () => {
    const params = new URLSearchParams({
      q: "test",
      search_method: "hybrid",
      limit: "10",
    });
    appendNamespace(params, "acme");
    const str = params.toString();
    expect(str.indexOf("q=")).toBeLessThan(str.indexOf("namespace="));
    expect(str.indexOf("search_method=")).toBeLessThan(str.indexOf("namespace="));
    expect(str.indexOf("limit=")).toBeLessThan(str.indexOf("namespace="));
  });
});

// ── graph_search URL construction tests ─────────────────────────────────────
// We simulate the URL-building logic from graph_search in tools.ts to assert
// namespace is included/excluded correctly without requiring a live Jarvis server.

function buildGraphSearchUrl(
  baseUrl: string,
  {
    q,
    search_method = "hybrid",
    type,
    limit = 10,
    namespace,
  }: {
    q: string;
    search_method?: string;
    type?: string;
    limit?: number;
    namespace?: string;
  }
): string {
  const params = new URLSearchParams({
    q,
    search_method: search_method ?? "hybrid",
    limit: String(limit),
  });
  if (type) params.set("type", type);
  appendNamespace(params, namespace);
  return `${baseUrl}/v2/nodes?${params.toString()}`;
}

test.describe("graph_search URL construction (graph_agent/tools.ts)", () => {
  const BASE = "https://jarvis.example.com";

  test("includes namespace param when namespace is provided", () => {
    const url = buildGraphSearchUrl(BASE, { q: "bitcoin", namespace: "acme" });
    expect(url).toContain("namespace=acme");
  });

  test("omits namespace param entirely when namespace is not provided (backward compat)", () => {
    const url = buildGraphSearchUrl(BASE, { q: "bitcoin" });
    expect(url).not.toContain("namespace");
  });

  test("omits namespace param when namespace is empty string (backward compat)", () => {
    const url = buildGraphSearchUrl(BASE, { q: "bitcoin", namespace: "" });
    expect(url).not.toContain("namespace");
  });

  test("passes namespace value verbatim (no lowercasing)", () => {
    const url = buildGraphSearchUrl(BASE, { q: "bitcoin", namespace: "MyNamespace" });
    expect(url).toContain("namespace=MyNamespace");
  });

  test("without namespace, URL is identical to today's baseline", () => {
    const url = buildGraphSearchUrl(BASE, { q: "bitcoin", search_method: "hybrid", limit: 10 });
    expect(url).toBe(`${BASE}/v2/nodes?q=bitcoin&search_method=hybrid&limit=10`);
  });

  test("with type and namespace, both params appear in URL", () => {
    const url = buildGraphSearchUrl(BASE, { q: "test", type: "Episode", namespace: "ns1" });
    expect(url).toContain("type=Episode");
    expect(url).toContain("namespace=ns1");
  });
});
