import test, { after } from "node:test";
import assert from "node:assert/strict";

import { get_nodes, post_nodes, get_map } from "../../graph/routes/graph.js";
import { ask, create_learning } from "../../graph/routes/knowledge.js";
import { gitsee, gitsee_services } from "../../graph/routes/gitsee.js";
import { get_services } from "../../graph/routes/services.js";
import { repo_agent, get_agent_session } from "../../repo/index.js";
import { logs_agent } from "../../log/index.js";
import {
  gitree_process,
  gitree_get_pr,
  gitree_get_feature,
  gitree_analyze_clues,
} from "../../gitree/routes.js";
import { db } from "../../graph/neo4j.js";

after(async () => {
  if (db && typeof (db as any).close === "function") {
    await db.close();
  }
});

type ReqShape = {
  method?: string;
  path?: string;
  query?: Record<string, any>;
  body?: Record<string, any>;
  params?: Record<string, any>;
};

function mockReq(input: ReqShape = {}) {
  return {
    method: input.method || "GET",
    path: input.path || "/",
    query: input.query || {},
    body: input.body || {},
    params: input.params || {},
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

function assertValidation400(state: { statusCode?: number; jsonBody?: any }, source: "query" | "body" | "params") {
  assert.equal(state.statusCode, 400);
  assert.equal(state.jsonBody?.error, "ValidationError");
  assert.equal(state.jsonBody?.message, `Invalid request ${source}`);
  assert.ok(Array.isArray(state.jsonBody?.details));
  assert.ok(state.jsonBody.details.length > 0);
}

test("graph handlers return 400 validation payloads", async () => {
  {
    const { res, state } = mockRes();
    await get_nodes(mockReq({ query: {} }), res);
    assertValidation400(state, "query");
  }

  {
    const { res, state } = mockRes();
    await post_nodes(mockReq({ method: "POST", body: {} }), res);
    assertValidation400(state, "body");
  }

  {
    const { res, state } = mockRes();
    await get_map(mockReq({ query: {} }), res);
    assertValidation400(state, "query");
  }
});

test("knowledge handlers return 400 validation payloads", async () => {
  {
    const { res, state } = mockRes();
    await ask(mockReq({ query: {} }), res);
    assertValidation400(state, "query");
  }

  {
    const { res, state } = mockRes();
    await create_learning(
      mockReq({
        method: "POST",
        body: { question: "Q", answer: "A" },
      }),
      res
    );
    assertValidation400(state, "body");
  }
});

test("gitsee/services handlers return 400 validation payloads", async () => {
  {
    const { res, state } = mockRes();
    await gitsee(mockReq({ method: "POST", body: {} }), res);
    assertValidation400(state, "body");
  }

  {
    const { res, state } = mockRes();
    await gitsee_services(mockReq({ query: {} }), res);
    assertValidation400(state, "query");
  }

  {
    const { res, state } = mockRes();
    await get_services(mockReq({ query: { clone: "true" } }), res);
    assertValidation400(state, "query");
  }
});

test("repo/log handlers return 400 validation payloads", async () => {
  {
    const { res, state } = mockRes();
    await repo_agent(mockReq({ method: "POST", body: {} }), res);
    assertValidation400(state, "body");
  }

  {
    const { res, state } = mockRes();
    await get_agent_session(mockReq({ query: {} }), res);
    assertValidation400(state, "query");
  }

  {
    const { res, state } = mockRes();
    await logs_agent(mockReq({ method: "POST", body: {} }), res);
    assertValidation400(state, "body");
  }
});

test("gitree handlers return 400 validation payloads", async () => {
  {
    const { res, state } = mockRes();
    await gitree_process(mockReq({ method: "POST", query: { summarize: "invalid" } }), res);
    assertValidation400(state, "query");
  }

  {
    const { res, state } = mockRes();
    await gitree_get_pr(mockReq({ params: { number: "abc" }, query: {} }), res);
    assertValidation400(state, "params");
  }

  {
    const { res, state } = mockRes();
    await gitree_get_feature(mockReq({ params: { id: "" }, query: {} }), res);
    assertValidation400(state, "params");
  }

  {
    const { res, state } = mockRes();
    await gitree_analyze_clues(mockReq({ query: { owner: "stakwork" } }), res);
    assertValidation400(state, "query");
  }
});
