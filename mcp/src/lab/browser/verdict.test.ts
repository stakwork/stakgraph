/**
 * browser/verify's verdict (verdict.ts), offline: what a page did → the
 * check contract, every failure named, the one line a person reads.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { judgeVerify, urlMatches } from "./verdict.js";

const AT = "https://pod-1.example.com/settings";

test("everything holds: supports, and the line says where the page got to", () => {
  const v = judgeVerify({
    status: 200,
    pageUrl: AT,
    expectations: [
      { label: 'text "Notification preferences"', holds: true },
      { label: "url **/settings", holds: true },
      { label: "no errors", holds: true },
    ],
  });
  assert.deepEqual(v, { supports: true, content: "/settings reached; all 3 expectations hold", failed: [] });
  assert.equal(judgeVerify({ status: 200, pageUrl: AT, expectations: [{ label: "evaluate", holds: true }] }).content, "/settings reached; the expectation holds");
  assert.equal(judgeVerify({ status: 200, pageUrl: "https://pod-1.example.com/", expectations: [] }).content, "pod-1.example.com reached", "no expectations: the page opened");
});

test("an expectation that does not hold refutes, named in failed and in the line", () => {
  const v = judgeVerify({
    status: 200,
    pageUrl: AT,
    expectations: [
      { label: 'text "Notification preferences"', holds: false, detail: "not visible" },
      { label: "url **/settings", holds: true },
      { label: "no errors", holds: false, detail: "2 on the page, first console: boom" },
    ],
  });
  assert.equal(v.supports, false);
  assert.deepEqual(v.failed, ['text "Notification preferences": not visible', "no errors: 2 on the page, first console: boom"]);
  assert.equal(v.content, '/settings reached; 2 of 3 expectations failed: text "Notification preferences" not visible; no errors 2 on the page, first console: boom');
});

test("a step whose target matched nothing is a result: the line says where the list stopped", () => {
  const v = judgeVerify({
    status: 200,
    stopped: { step: 2, verb: "click", args: { selector: "text=Sign in", timeoutMs: 1000 }, message: "browser/click: locator.click: Timeout 1000ms exceeded.\nCall log:\n  - waiting for locator('text=Sign in')" },
    pageUrl: "https://pod-1.example.com/login",
    expectations: [{ label: "url **/settings", holds: false, detail: "the page is at https://pod-1.example.com/login" }],
  });
  assert.equal(v.supports, false);
  assert.equal(v.failed[0], "step 2 (click text=Sign in) failed: browser/click: locator.click: Timeout 1000ms exceeded.");
  assert.match(v.content, /^stopped at step 2 \(click text=Sign in\): browser\/click: locator\.click: Timeout 1000ms exceeded\.; 1 of 1 expectation failed: url \*\*\/settings the page is at/);
  assert.doesNotMatch(v.content, /Call log/, "Playwright's call log stays out of the line");
});

test("a strict-mode violation keeps the elements it matched and the way out in failed; the line stays one line", () => {
  const message =
    "browser/fill: locator.fill: Error: strict mode violation: locator('input') resolved to 2 elements:\n" +
    "    1) <input type=\"text\" placeholder=\"dev-user\"/> aka getByPlaceholder('dev-user')\n" +
    "    2) <input type=\"hidden\" name=\"csrfToken\"/> aka locator('input[name=\"csrfToken\"]')\n" +
    "→ act by ref instead: browser/snapshot, then the element's [ref=eN] (or narrow the selector, e.g. `>> nth=1`)";
  const v = judgeVerify({ status: 200, stopped: { step: 1, verb: "fill", args: { selector: "input", value: "x" }, message }, pageUrl: AT, expectations: [] });
  assert.equal(v.failed[0], `step 1 (fill input) failed: ${message}`);
  assert.equal(v.content, "stopped at step 1 (fill input): browser/fill: locator.fill: Error: strict mode violation: locator('input') resolved to 2 elements:");
});

test("a document that answered HTTP 400 or more refutes, even with no expectations", () => {
  const v = judgeVerify({ status: 500, pageUrl: AT, expectations: [] });
  assert.deepEqual(v, { supports: false, content: "/settings reached; HTTP 500", failed: ["the page answered HTTP 500"] });
  assert.equal(judgeVerify({ status: 0, pageUrl: "data:text/html,<p>x</p>", expectations: [] }).content, "the page reached", "a data: URL fetches nothing");
});

test("the line is bounded", () => {
  const many = Array.from({ length: 40 }, (_, i) => ({ label: `text "a fairly long expected phrase number ${i}"`, holds: false, detail: "not visible" }));
  const v = judgeVerify({ status: 200, pageUrl: AT, expectations: many });
  assert.ok(v.content.length <= 600);
  assert.equal(v.failed.length, 40, "failed keeps every one");
});

test("urlMatches: a glob when there is a star (** crosses /, * does not), else a substring", () => {
  assert.equal(urlMatches(AT, "**/settings"), true);
  assert.equal(urlMatches(AT, "https://*.example.com/settings"), true);
  assert.equal(urlMatches(AT, "https://*/settings"), true);
  assert.equal(urlMatches(`${AT}/notifications`, "**/settings"), false, "a glob matches the whole URL");
  assert.equal(urlMatches(`${AT}/notifications`, "**/settings/**"), true);
  assert.equal(urlMatches("https://a.example.com/x/settings", "https://a.example.com/*"), false, "* stays inside a segment");
  assert.equal(urlMatches(`${AT}?tab=email`, "**/settings?tab=*"), true, "? is a literal, as in a query");
  assert.equal(urlMatches(AT, "/settings"), true, "no star: a substring");
  assert.equal(urlMatches(AT, "/profile"), false);
  assert.equal(urlMatches("https://x.example.com/a.b", "**/a.b"), true);
  assert.equal(urlMatches("https://x.example.com/axb", "**/a.b"), false, "a dot is a dot");
});
