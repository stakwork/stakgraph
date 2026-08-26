import { describe, it } from "node:test";
import assert from "node:assert/strict";
import type { HttpResponse } from "./capabilities.js";
import extract, { extractTitle } from "./steps/lib/html/extract.js";

// ── Fake ctx ────────────────────────────────────────────────────────────────
//
// html/extract fetches through ctx.services.http. Stub it with a canned
// response and record the calls so we can assert what was requested.

interface Call {
  url: string;
  headers?: Record<string, string>;
}

function makeCtx(body: unknown, opts: { status?: number } = {}) {
  const calls: Call[] = [];
  const http = async (
    url: string,
    o: { headers?: Record<string, string> } = {},
  ): Promise<HttpResponse> => {
    calls.push({ url, headers: o.headers });
    const status = opts.status ?? 200;
    return { status, ok: status < 400, headers: {}, body };
  };
  const ctx = {
    runId: "t",
    path: "t",
    scope: {},
    input: {},
    emit: async () => {},
    services: { http, secrets: { get: async () => undefined } },
  } as never;
  return { ctx, calls };
}

const cfg = (over: Record<string, unknown>) => extract.input.parse(over);

// ── extraction ──────────────────────────────────────────────────────────────

describe("html/extract", () => {
  it("strips scripts/styles, keeps text, decodes entities", async () => {
    const html = `<html><head><title>Q3 &amp; Q4 Report</title>
      <style>body { color: red }</style></head>
      <body><script>alert("nope")</script>
      <h1>Revenue &amp; Margin</h1>
      <p>Total revenue was &#36;5,000 &ndash; up 10%.</p></body></html>`;
    const out = await extract.run(cfg({ html }), makeCtx(null).ctx);
    assert.equal(out.title, "Q3 & Q4 Report");
    assert.match(out.text, /Revenue & Margin/);
    assert.match(out.text, /\$5,000 – up 10%/);
    assert.doesNotMatch(out.text, /alert|color: red/);
    assert.equal(out.status, null);
    assert.equal(out.truncated, false);
  });

  it("renders tables as readable rows (the filings case)", async () => {
    const html = `<table>
      <tr><th>Segment</th><th>FY23</th><th>FY22</th></tr>
      <tr><td>iPhone</td><td>200,583</td><td>205,489</td></tr>
      <tr><td>Services</td><td>85,200</td><td>78,129</td></tr>
    </table>`;
    const out = await extract.run(cfg({ html }), makeCtx(null).ctx);
    // Each row survives as one line with all its cells.
    const iphone = out.text.split("\n").find((l) => l.includes("iPhone"));
    assert.ok(iphone, `no iPhone row in:\n${out.text}`);
    assert.match(iphone!, /200,583/);
    assert.match(iphone!, /205,489/);
    assert.match(out.text, /Services\s+85,200\s+78,129/);
  });

  it("fetches via ctx.services.http with the given headers", async () => {
    const { ctx, calls } = makeCtx("<html><body><p>Hello SEC</p></body></html>");
    const out = await extract.run(
      cfg({
        url: "https://www.sec.gov/some/filing.htm",
        headers: { "User-Agent": "vein test test@example.com" },
      }),
      ctx,
    );
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, "https://www.sec.gov/some/filing.htm");
    assert.equal(calls[0].headers?.["User-Agent"], "vein test test@example.com");
    assert.match(out.text, /Hello SEC/);
    assert.equal(out.status, 200);
  });

  it("throws on a non-ok fetch", async () => {
    const { ctx } = makeCtx("nope", { status: 403 });
    await assert.rejects(
      () => extract.run(cfg({ url: "https://example.com/x" }), ctx),
      /failed with 403/,
    );
  });

  it("truncates at maxChars and reports the full length", async () => {
    const html = `<p>${"word ".repeat(200)}</p>`;
    const out = await extract.run(cfg({ html, maxChars: 50 }), makeCtx(null).ctx);
    assert.equal(out.truncated, true);
    assert.equal(out.text.length, 50);
    assert.ok(out.length > 900, `length was ${out.length}`);
  });

  it("rejects config with neither url nor html", () => {
    assert.throws(() => cfg({}), /needs either/);
  });

  it("extractTitle decodes entities and collapses whitespace", () => {
    assert.equal(extractTitle("<title>  A&nbsp;&amp;\n B  </title>"), "A & B");
    assert.equal(extractTitle("<p>no title</p>"), null);
    assert.equal(extractTitle("<title>   </title>"), null);
  });
});
