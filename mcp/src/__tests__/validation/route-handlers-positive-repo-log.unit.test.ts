import test from "node:test";
import assert from "node:assert/strict";



import { repo_agent, get_agent_session, get_leaks } from "../../repo/index.js";
import { logs_agent } from "../../log/index.js";

// Local mock implementations for external dependencies (no global assignment)
// If you need to mock, use dependency injection or patch the handler modules directly if possible.

function mockReq(input: Record<string, any> = {}) {
  // Minimal Express Request mock
  return {
    method: input.method || "GET",
    path: input.path || "/",
    query: input.query || {},
    body: input.body || {},
    params: input.params || {},
    headers: input.headers || {},
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
  } as any;
  return { res, state };
}

test("repo handlers accept valid input and return 200/202", async () => {
  // repo_agent
  {
    const { res, state } = mockRes();
    await repo_agent(mockReq({ method: "POST", body: { repo_url: "https://github.com/org/repo", prompt: "How does auth work?" }, params: {} }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
  // get_agent_session
  {
    const { res, state } = mockRes();
    await get_agent_session(mockReq({ query: { session_id: "abc123" }, params: {} }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
  // get_leaks
  {
    const { res, state } = mockRes();
    await get_leaks(mockReq({ query: { repo_url: "https://github.com/org/repo" }, params: {} }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
});

test("log handlers accept valid input and return 200/202", async () => {
  // logs_agent
  {
    const { res, state } = mockRes();
    await logs_agent(mockReq({ method: "POST", body: { prompt: "Show logs" }, params: {} }), res);
    assert.ok(state.statusCode === undefined || state.statusCode < 400);
  }
});
