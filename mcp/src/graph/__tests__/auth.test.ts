/**
 * Tests for authMiddleware — mounted globally at index.ts, so it is the gate
 * in front of the entire API surface.
 */
import { test, expect } from "../../testkit.js";
import type { Request, Response } from "express";
import jwt from "jsonwebtoken";
import { authMiddleware } from "../routes.js";
import { signApiToken, signEventsToken } from "../../repo/events.js";

const TOKEN = "s3cret-api-token";

interface Captured {
  status?: number;
  body?: unknown;
  headers: Record<string, string>;
  nexted: boolean;
}

function run(opts: {
  headers?: Record<string, string>;
  query?: Record<string, string>;
  /** What req.accepts(["html","json"]) returns. */
  accepts?: string | false;
}): Captured {
  const headers: Record<string, string> = {};
  for (const [k, v] of Object.entries(opts.headers ?? {})) {
    headers[k.toLowerCase()] = v;
  }

  const captured: Captured = { headers: {}, nexted: false };

  const req = {
    header: (name: string) => headers[name.toLowerCase()],
    query: opts.query ?? {},
    accepts: () => opts.accepts ?? "json",
  } as unknown as Request;

  const res = {
    set: (k: string, v: string) => {
      captured.headers[k] = v;
      return res;
    },
    status: (n: number) => {
      captured.status = n;
      return res;
    },
    json: (b: unknown) => {
      captured.body = b;
      return res;
    },
    send: (b: unknown) => {
      captured.body = b;
      return res;
    },
  } as unknown as Response;

  authMiddleware(req, res, () => {
    captured.nexted = true;
  });

  return captured;
}

function basic(user: string, pass: string): string {
  return "Basic " + Buffer.from(`${user}:${pass}`).toString("base64");
}

test.describe("authMiddleware", () => {
  let prev: string | undefined;

  test.beforeEach(() => {
    prev = process.env.API_TOKEN;
    process.env.API_TOKEN = TOKEN;
  });

  test.afterEach(() => {
    if (prev === undefined) delete process.env.API_TOKEN;
    else process.env.API_TOKEN = prev;
  });

  test("dev mode: no API_TOKEN configured allows everything through", () => {
    delete process.env.API_TOKEN;
    const r = run({});
    expect(r.nexted).toBe(true);
    expect(r.status).toBeUndefined();
  });

  test("no credentials is rejected once API_TOKEN is set", () => {
    const r = run({});
    expect(r.nexted).toBe(false);
    expect(r.status).toBe(401);
    expect(r.body).toEqual({ error: "Unauthorized: Invalid API token" });
  });

  test("matching x-api-token header passes", () => {
    const r = run({ headers: { "x-api-token": TOKEN } });
    expect(r.nexted).toBe(true);
  });

  test("wrong x-api-token header is rejected", () => {
    const r = run({ headers: { "x-api-token": "wrong" } });
    expect(r.nexted).toBe(false);
    expect(r.status).toBe(401);
  });

  test("valid Bearer api JWT passes", () => {
    const r = run({ headers: { Authorization: `Bearer ${signApiToken()}` } });
    expect(r.nexted).toBe(true);
  });

  test("Bearer JWT signed with a different secret is rejected", () => {
    const forged = jwt.sign({ scope: "api" }, "not-the-real-secret", {
      expiresIn: "1h",
    });
    const r = run({ headers: { Authorization: `Bearer ${forged}` } });
    expect(r.nexted).toBe(false);
    expect(r.status).toBe(401);
  });

  test("expired Bearer JWT is rejected", () => {
    const r = run({
      headers: { Authorization: `Bearer ${signApiToken("-1s")}` },
    });
    expect(r.nexted).toBe(false);
    expect(r.status).toBe(401);
  });

  test("valid ?token= api JWT passes (iframe embed path)", () => {
    const r = run({ query: { token: signApiToken() } });
    expect(r.nexted).toBe(true);
  });

  test("garbage ?token= falls through to 401 rather than throwing", () => {
    const r = run({ query: { token: "not-a-jwt" } });
    expect(r.nexted).toBe(false);
    expect(r.status).toBe(401);
  });

  test("Basic auth with the api token as password passes", () => {
    const r = run({ headers: { Authorization: basic("admin", TOKEN) } });
    expect(r.nexted).toBe(true);
  });

  test("Basic auth with the wrong password is rejected", () => {
    const r = run({ headers: { Authorization: basic("admin", "nope") } });
    expect(r.nexted).toBe(false);
    expect(r.status).toBe(401);
  });

  test("malformed Basic auth is rejected without throwing", () => {
    const r = run({ headers: { Authorization: "Basic !!!not-base64!!!" } });
    expect(r.nexted).toBe(false);
    expect(r.status).toBe(401);
  });

  test("html requests get a WWW-Authenticate challenge, not a json 401", () => {
    const r = run({ accepts: "html" });
    expect(r.nexted).toBe(false);
    expect(r.status).toBe(401);
    expect(r.headers["WWW-Authenticate"]).toBe(
      'Basic realm="stakgraph", charset="UTF-8"'
    );
  });

  // Events tokens and api tokens are signed with the same secret, so only the
  // scope check in verifyApiToken keeps an events token off the API surface.
  test("an events-scoped token is rejected as a Bearer credential", () => {
    const r = run({
      headers: { Authorization: `Bearer ${signEventsToken("req-1")}` },
    });
    expect(r.nexted).toBe(false);
    expect(r.status).toBe(401);
  });

  test("an events-scoped token is rejected as a ?token= credential", () => {
    const r = run({ query: { token: signEventsToken("req-1") } });
    expect(r.nexted).toBe(false);
    expect(r.status).toBe(401);
  });
});
