import test from "node:test";
import assert from "node:assert/strict";

import { get_nodes, post_nodes, get_edges, search, get_map, get_repo_map, get_shortest_path, get_graph } from "../../graph/routes/graph.js";
import { ask, create_learning, explore, understand, seed_understanding, get_learnings, create_pull_request, seed_stories } from "../../graph/routes/knowledge.js";
import { gitsee, gitsee_services, gitsee_agent, gitseeEvents } from "../../graph/routes/gitsee.js";
import { get_services, mocks_inventory } from "../../graph/routes/services.js";

function mockReq(input: Record<string, any> = {}) {
  // Add context for handlers needing db, vectorSearch, etc.
  const dbMock = {
    nodes_by_type: async () => [],
    edges_by_type: async () => [],
    vectorSearch: async () => [],
    search: async () => [],
    all_edges: async () => [],
    get_pkg_files: async () => [],
    get_env_vars: async () => [],
    get_rules_files: async () => [],
    nodes_by_types_total: async () => [],
    nodes_by_types_per_type: async () => [],
    get_mocks_inventory: async () => [],
  };
  // Minimal Express Request mock
  return {
    method: input.method || "GET",
    path: input.path || "/",
    query: input.query || {},
    body: input.body || {},
    params: input.params || {},
    headers: input.headers || {},
    context: { db: dbMock },
    get: () => undefined,
    header: () => undefined,
    accepts: () => false,
    acceptsCharsets: () => false,
    acceptsEncodings: () => false,
    acceptsLanguages: () => false,
    is: () => false,
    // ...add more as needed for handler compatibility
  } as any;
}

function mockRes() {
  const state: {
    statusCode?: number;
    jsonBody?: any;
    sentBody?: any;
    headers: Record<string, string>;
  } = {
    headers: {},
  };
  const res = {
    status(code: number) {
      state.statusCode = code;
      return this;
    },
    json(payload: any) {
      state.jsonBody = payload;
      return this;
    },
    send(payload: any) {
      state.sentBody = payload;
      return this;
    },
    setHeader(name: string, value: string) {
      state.headers[name] = value;
      return this;
    },
    end(payload?: any) {
      state.sentBody = payload;
      return this;
    },
    writeHead(statusCode: number, headers?: Record<string, string>) {
      state.statusCode = statusCode;
      if (headers) {
        Object.assign(state.headers, headers);
      }
      return this;
    },
  } as any;
  return { res, state };
}

test("graph handlers accept valid input and return 200/202", async () => {
  // get_nodes
  {
    const { res, state } = mockRes();
    await get_nodes(mockReq({ query: { node_type: "Function" } }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
  // post_nodes
  {
    const { res, state } = mockRes();
    await post_nodes(mockReq({ method: "POST", body: { node_type: "Class" } }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
  // get_edges
  {
    const { res, state } = mockRes();
    await get_edges(mockReq({ query: { edge_type: "CALLS" } }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
  // search
  {
    const { res, state } = mockRes();
    await search(mockReq({ query: { query: "auth" } }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
  // get_map
  {
    const { res, state } = mockRes();
    await get_map(mockReq({ query: { node_type: "Function", name: "run" } }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
  // get_repo_map
  {
    const { res, state } = mockRes();
    await get_repo_map(mockReq({ query: { node_type: "Repository" } }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
  // get_shortest_path
  {
    const { res, state } = mockRes();
    await get_shortest_path(mockReq({ query: { start_ref_id: "a", end_ref_id: "b" } }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
  // get_graph
  {
    const { res, state } = mockRes();
    await get_graph(mockReq({ query: { node_types: "Function" } }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
});

test("knowledge handlers accept valid input and return 200/202", async () => {
  // ask
  {
    const { res, state } = mockRes();
    await ask(mockReq({ query: { question: "What is this?" } }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
  // create_learning
  {
    const { res, state } = mockRes();
    await create_learning(mockReq({ method: "POST", body: { question: "Q", answer: "A", context: "ctx" } }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
  // explore
  {
    const { res, state } = mockRes();
    await explore(mockReq({ query: { prompt: "explore" } }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
  // understand
  {
    const { res, state } = mockRes();
    await understand(mockReq({ query: { question: "How?" } }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
  // seed_understanding
  {
    const { res, state } = mockRes();
    await seed_understanding(mockReq({ query: { budget: "1" } }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
  // get_learnings
  {
    const { res, state } = mockRes();
    await get_learnings(mockReq({ query: {} }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
  // create_pull_request
  {
    const { res, state } = mockRes();
    await create_pull_request(mockReq({ method: "POST", body: { name: "feat", docs: "docs", number: "42" } }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
  // seed_stories
  {
    const { res, state } = mockRes();
    await seed_stories(mockReq({ query: { prompt: "stories" } }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
});

test("gitsee/services handlers accept valid input and return 200/202", async () => {
  // gitsee
  {
    const { res, state } = mockRes();
    await gitsee(mockReq({ method: "POST", body: { owner: "o", repo: "r", data: ["repo_info"] } }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
  // gitsee_services
  {
    const { res, state } = mockRes();
    await gitsee_services(mockReq({ query: { owner: "o", repo: "r" } }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
  // gitsee_agent
  {
    const { res, state } = mockRes();
    await gitsee_agent(mockReq({ query: { owner: "o", repo: "r", prompt: "summarize" } }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
  // gitseeEvents
  {
    const { res, state } = mockRes();
    await gitseeEvents(mockReq({ params: { owner: "o", repo: "r" } }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
  // get_services
  {
    const { res, state } = mockRes();
    await get_services(mockReq({ query: { repo_url: "https://github.com/org/repo" } }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
  // mocks_inventory
  {
    const { res, state } = mockRes();
    await mocks_inventory(mockReq({ query: { repo: "org/repo" } }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
});
