/**
 * Tests for the two JWT scopes in events.ts. Both are signed with the same
 * secret (API_TOKEN), so the scope boundary is enforced by code, not by keys.
 */
import { test, expect } from "../../testkit.js";
import jwt from "jsonwebtoken";
import {
  signApiToken,
  verifyApiToken,
  signEventsToken,
  verifyEventsToken,
} from "../events.js";

const TOKEN = "s3cret-api-token";

test.describe("events.ts JWT scopes", () => {
  let prev: string | undefined;

  test.beforeEach(() => {
    prev = process.env.API_TOKEN;
    process.env.API_TOKEN = TOKEN;
  });

  test.afterEach(() => {
    if (prev === undefined) delete process.env.API_TOKEN;
    else process.env.API_TOKEN = prev;
  });

  test("api token round-trips with its scope intact", () => {
    const payload = verifyApiToken(signApiToken());
    expect(payload.scope).toBe("api");
  });

  test("events token round-trips with its request_id intact", () => {
    const payload = verifyEventsToken(signEventsToken("req-abc"));
    expect(payload.request_id).toBe("req-abc");
  });

  test("verifyApiToken rejects a token signed with another secret", () => {
    const forged = jwt.sign({ scope: "api" }, "other-secret", {
      expiresIn: "1h",
    });
    expect(() => verifyApiToken(forged)).toThrow();
  });

  test("verifyApiToken rejects an expired token", () => {
    expect(() => verifyApiToken(signApiToken("-1s"))).toThrow();
  });

  test("verifyEventsToken rejects a token signed with another secret", () => {
    const forged = jwt.sign({ request_id: "req-abc" }, "other-secret", {
      expiresIn: "1h",
    });
    expect(() => verifyEventsToken(forged)).toThrow();
  });

  test("verifyApiToken rejects an events token — wrong scope", () => {
    expect(() => verifyApiToken(signEventsToken("req-abc") as never)).toThrow(
      "Invalid token scope"
    );
  });

  /**
   * verifyEventsToken performs no scope check, so an api token passes
   * signature verification. The /events/:request_id handler is safe only
   * because it then compares payload.request_id against the route param —
   * and api tokens carry no request_id. Guard that assumption.
   */
  test("an api token carries no request_id to match an events route against", () => {
    const payload = verifyEventsToken(signApiToken() as never);
    expect(payload.request_id).toBeUndefined();
  });

  test("signing throws when API_TOKEN is unset", () => {
    delete process.env.API_TOKEN;
    expect(() => signEventsToken("req-abc")).toThrow("API_TOKEN is required");
    expect(() => signApiToken()).toThrow("API_TOKEN is required");
  });
});
