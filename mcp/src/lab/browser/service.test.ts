/** The service's offline surface: what boot accepts (no browser is touched). */
import { test } from "node:test";
import assert from "node:assert/strict";
import { BrowserService } from "./service.js";

test("BROWSER_WS_URL and BROWSER_CDP_URL together are refused at boot; either alone is a backend", () => {
  assert.throws(() => BrowserService.fromEnv({ BROWSER_WS_URL: "ws://browser:3000", BROWSER_CDP_URL: "http://127.0.0.1:9222" }), /not both/);
  assert.equal(BrowserService.fromEnv({ BROWSER_WS_URL: "ws://browser:3000" }).cdp, false);
  assert.equal(BrowserService.fromEnv({ BROWSER_CDP_URL: "http://127.0.0.1:9222" }).cdp, true);
  assert.equal(BrowserService.fromEnv({}).cdp, false, "unset: a local launch");
  assert.throws(() => BrowserService.fromEnv({ BROWSER_VIEWPORT: "wide" }), /WIDTHxHEIGHT/);
});
