/** The service's offline surface: what boot accepts (no browser is touched). */
import { test } from "node:test";
import assert from "node:assert/strict";
import { BrowserService, strictHint } from "./service.js";

test("BROWSER_WS_URL and BROWSER_CDP_URL together are refused at boot; either alone is a backend", () => {
  assert.throws(() => BrowserService.fromEnv({ BROWSER_WS_URL: "ws://browser:3000", BROWSER_CDP_URL: "http://127.0.0.1:9222" }), /not both/);
  assert.equal(BrowserService.fromEnv({ BROWSER_WS_URL: "ws://browser:3000" }).cdp, false);
  assert.equal(BrowserService.fromEnv({ BROWSER_CDP_URL: "http://127.0.0.1:9222" }).cdp, true);
  assert.equal(BrowserService.fromEnv({}).cdp, false, "unset: a local launch");
  assert.throws(() => BrowserService.fromEnv({ BROWSER_VIEWPORT: "wide" }), /WIDTHxHEIGHT/);
});

test("a strict-mode violation keeps the matches, drops the call log and names the way out", () => {
  const pw = "locator.click: Error: strict mode violation: locator('text=algorithms') resolved to 2 elements:\n    1) <p>…</p>\n    2) <strong>algorithms</strong>\n\nCall log:\n  - waiting for locator('text=algorithms')\n";
  const m = strictHint(pw);
  assert.match(m, /resolved to 2 elements:\n    1\) <p>…<\/p>\n    2\) <strong>algorithms<\/strong>\n→ act by ref instead/);
  assert.doesNotMatch(m, /Call log/);
  assert.equal(strictHint("locator.click: Timeout 5000ms exceeded."), "locator.click: Timeout 5000ms exceeded.", "anything else is untouched");
});
