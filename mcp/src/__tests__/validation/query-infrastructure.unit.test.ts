import test from "node:test";
import assert from "node:assert/strict";
import { z } from "zod";

import { sendValidationError } from "../../validation.js";
import {
  booleanLikeSchema,
  csvStringArraySchema,
  parseQuery,
} from "../../graph/routes/validation.js";
import {
  getNodesQuerySchema,
  shortestPathQuerySchema,
} from "../../graph/routes/schemas/graph.js";
import {
  repoAgentBodySchema,
  getAgentSessionQuerySchema,
} from "../../repo/schemas/routes.js";
import { logsAgentBodySchema } from "../../log/schemas/routes.js";
import {
  gitreeProcessQuerySchema,
  gitreeProvenanceBodySchema,
} from "../../gitree/schemas/routes.js";

function createMockRes() {
  const state: { statusCode?: number; payload?: any } = {};
  return {
    state,
    res: {
      status(code: number) {
        state.statusCode = code;
        return this;
      },
      json(payload: any) {
        state.payload = payload;
        return this;
      },
    },
  };
}

test("sendValidationError returns standardized payload", () => {
  const parsed = z.object({ q: z.string().min(1) }).safeParse({ q: "" });
  assert.equal(parsed.success, false);

  const { res, state } = createMockRes();
  sendValidationError(res as any, "query", parsed.error);

  assert.equal(state.statusCode, 400);
  assert.equal(state.payload.error, "ValidationError");
  assert.equal(state.payload.message, "Invalid request query");
  assert.ok(Array.isArray(state.payload.details));
  assert.ok(state.payload.details.length > 0);
});

test("booleanLikeSchema coerces string/boolean inputs", () => {
  assert.equal(booleanLikeSchema.parse("1"), true);
  assert.equal(booleanLikeSchema.parse("false"), false);
  assert.equal(booleanLikeSchema.parse(true), true);
});

test("csvStringArraySchema normalizes csv and array input", () => {
  assert.deepEqual(csvStringArraySchema.parse("a, b ,, c"), ["a", "b", "c"]);
  assert.deepEqual(csvStringArraySchema.parse([" a ", "", "b"]), ["a", "b"]);
});

test("parseQuery validates and returns parsed query", () => {
  const { res } = createMockRes();
  const req = {
    query: {
      node_type: "Function",
      concise: "true",
      ref_ids: "r1,r2",
    },
  };

  const parsed = parseQuery(req as any, res as any, getNodesQuerySchema);
  assert.ok(parsed);
  assert.equal(parsed?.node_type, "Function");
  assert.equal(parsed?.concise, true);
  assert.deepEqual(parsed?.ref_ids, ["r1", "r2"]);
});

test("shortestPathQuerySchema enforces pair requirement", () => {
  const invalid = shortestPathQuerySchema.safeParse({ start_node_key: "a" });
  assert.equal(invalid.success, false);

  const valid = shortestPathQuerySchema.safeParse({
    start_node_key: "a",
    end_node_key: "b",
  });
  assert.equal(valid.success, true);
});

test("repoAgentBodySchema requires repo_url and prompt", () => {
  const invalid = repoAgentBodySchema.safeParse({ repo_url: "https://github.com/stakwork/hive" });
  assert.equal(invalid.success, false);

  const valid = repoAgentBodySchema.safeParse({
    repo_url: "https://github.com/stakwork/hive",
    prompt: "How does auth work?",
  });
  assert.equal(valid.success, true);
});

test("getAgentSessionQuerySchema accepts either session_id or sessionId", () => {
  assert.equal(getAgentSessionQuerySchema.safeParse({ session_id: "abc" }).success, true);
  assert.equal(getAgentSessionQuerySchema.safeParse({ sessionId: "abc" }).success, true);
  assert.equal(getAgentSessionQuerySchema.safeParse({}).success, false);
});

test("logsAgentBodySchema requires prompt", () => {
  assert.equal(logsAgentBodySchema.safeParse({}).success, false);
  assert.equal(logsAgentBodySchema.safeParse({ prompt: "investigate logs" }).success, true);
});

test("gitree process/provenance schemas parse booleans and arrays", () => {
  const processParsed = gitreeProcessQuerySchema.parse({
    owner: "stakwork",
    repo: "hive",
    summarize: "true",
    link: "0",
  });

  assert.equal(processParsed.summarize, true);
  assert.equal(processParsed.link, false);

  const provenance = gitreeProvenanceBodySchema.safeParse({ conceptIds: ["f1", "f2"] });
  assert.equal(provenance.success, true);
  assert.equal(gitreeProvenanceBodySchema.safeParse({ conceptIds: "f1" }).success, false);
});
