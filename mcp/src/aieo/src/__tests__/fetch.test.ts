import {
  createWebFetch,
  validateFetchUrl,
  isPrivateAddress,
  htmlToText,
  resolveFetchBackend,
  captureNativeFetchResults,
  WEB_FETCH_TOOL_NAME,
  type WebFetchResult,
} from "../fetch.js";

type TestCase = { label: string; run: () => Promise<void> | void };

function assert(cond: unknown, msg: string): void {
  if (!cond) throw new Error(msg);
}

async function rejects(fn: () => Promise<unknown>, pattern: RegExp, label: string): Promise<void> {
  try {
    await fn();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    assert(pattern.test(msg), `${label}: rejected, but message "${msg}" !~ ${pattern}`);
    return;
  }
  throw new Error(`${label}: expected a rejection`);
}

const PUBLIC = async () => ["93.184.216.34"];
const PRIVATE = async () => ["10.0.0.5"];
const MIXED = async () => ["93.184.216.34", "10.0.0.5"];

/** Stub global fetch; returns the list of URLs it was called with. */
function stubFetch(handler: (url: string) => Response | Promise<Response>): string[] {
  const calls: string[] = [];
  (globalThis as any).fetch = async (url: unknown) => {
    calls.push(String(url));
    return handler(String(url));
  };
  return calls;
}

const html = (body: string, headers: Record<string, string> = {}) =>
  new Response(body, {
    status: 200,
    headers: { "content-type": "text/html; charset=utf-8", ...headers },
  });

const redirect = (to: string, status = 302) =>
  new Response(null, { status, headers: { location: to } });

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const exec = (t: any, input: any) => t.execute(input, { toolCallId: "t", messages: [] });

const PAGE = `<!doctype html>
<html><head>
  <title> Hello &amp; World </title>
  <style>body { color: red }</style>
  <script>alert("nope")</script>
</head>
<body>
  <!-- a comment -->
  <h1>Heading</h1>
  <p>Tom &amp; Jerry &mdash; &#8220;quoted&#8221; &#x27;x&#x27;&nbsp;done</p>
  <ul><li>one</li><li>two</li></ul>
  <svg><text>vector junk</text></svg>
  <p>Last<br>line</p>
</body></html>`;

const tests: TestCase[] = [
  {
    label: "anthropic keeps the native tool; others fall to http",
    run() {
      assert(resolveFetchBackend("anthropic") === "anthropic", "anthropic → anthropic");
      for (const p of ["openai", "google", "openrouter", "xai"] as const) {
        assert(resolveFetchBackend(p) === "http", `${p} → http`);
      }
    },
  },
  {
    label: "anthropic path: no key → tool undefined; with key → native tool",
    run() {
      delete process.env.ANTHROPIC_API_KEY;
      const none = createWebFetch({ provider: "anthropic" });
      assert(none.tool === undefined, "tool should be undefined without a key");
      assert(none.backend === undefined, "backend should be undefined without a key");
      const native = createWebFetch({ provider: "anthropic", apiKey: "sk-ant-fake" });
      assert(!!native.tool, "tool present with a key");
      assert(native.native === true, "anthropic is native");
      assert(native.backend === "anthropic", "backend is anthropic");
    },
  },
  {
    label: "http path needs no key",
    run() {
      delete process.env.ANTHROPIC_API_KEY;
      const wf = createWebFetch({ provider: "openai" });
      assert(!!wf.tool, "tool present");
      assert(wf.native === false, "shim is not native");
      assert(wf.backend === "http", "backend is http");
      assert(WEB_FETCH_TOOL_NAME === "web_fetch", "tool name matches Anthropic's native name");
    },
  },
  {
    label: "isPrivateAddress classifies v4, v6 and mapped literals",
    run() {
      const blocked = [
        "127.0.0.1", "10.1.2.3", "172.16.0.1", "172.31.255.255", "192.168.1.1",
        "169.254.169.254", "100.64.0.1", "0.0.0.0", "255.255.255.255", "224.0.0.1",
        "::1", "::", "fe80::1", "fc00::1", "fd12:3456::1", "ff02::1",
        "::ffff:127.0.0.1", "::ffff:10.0.0.1", "[::1]", "64:ff9b::a00:1", "2002:a00:1::1",
        "not-an-ip", "",
      ];
      for (const ip of blocked) assert(isPrivateAddress(ip) === true, `${ip} should be private`);
      const allowed = ["8.8.8.8", "93.184.216.34", "172.32.0.1", "100.128.0.1", "2606:4700::1111", "::ffff:8.8.8.8"];
      for (const ip of allowed) assert(isPrivateAddress(ip) === false, `${ip} should be public`);
    },
  },
  {
    label: "validateFetchUrl rejects bad schemes, credentials, local names and private literals",
    async run() {
      const cases: Array<[string, RegExp]> = [
        ["file:///etc/passwd", /http\(s\)/],
        ["ftp://example.com/x", /http\(s\)/],
        ["javascript:alert(1)", /http\(s\)|valid absolute/],
        ["not a url", /valid absolute/],
        ["https://user:pw@example.com/", /credentials/],
        ["http://localhost:3000/", /local/],
        ["http://api.localhost/", /local/],
        ["http://127.0.0.1/", /private or reserved/],
        ["http://[::1]/", /private or reserved/],
        ["http://2130706433/", /private or reserved/],
        ["http://0x7f000001/", /private or reserved/],
        ["http://169.254.169.254/latest/meta-data", /private or reserved/],
        ["http://10.0.0.1:8080/", /private or reserved/],
      ];
      for (const [url, pattern] of cases) {
        await rejects(() => validateFetchUrl(url, { lookup: PUBLIC }), pattern, url);
      }
    },
  },
  {
    label: "validateFetchUrl resolves hostnames and refuses private or mixed answers",
    async run() {
      const ok = await validateFetchUrl("https://example.com/page", { lookup: PUBLIC });
      assert(ok.hostname === "example.com", "public host passes");
      await rejects(
        () => validateFetchUrl("https://intranet.example.com/", { lookup: PRIVATE }),
        /resolves to a private/,
        "private resolution",
      );
      await rejects(
        () => validateFetchUrl("https://mixed.example.com/", { lookup: MIXED }),
        /resolves to a private/,
        "mixed resolution",
      );
      await rejects(
        () => validateFetchUrl("https://nope.example.com/", { lookup: async () => [] }),
        /Could not resolve/,
        "empty resolution",
      );
      await rejects(
        () => validateFetchUrl("https://nx.example.com/", { lookup: async () => { throw new Error("ENOTFOUND"); } }),
        /Could not resolve .*ENOTFOUND/,
        "resolver error",
      );
      const literal = await validateFetchUrl("http://93.184.216.34/", {
        lookup: async () => { throw new Error("should not resolve a literal"); },
      });
      assert(literal.hostname === "93.184.216.34", "public literal passes without DNS");
    },
  },
  {
    label: "allowed and blocked domains match the host and its subdomains",
    async run() {
      const allow = { lookup: PUBLIC, allowedDomains: ["example.com", "https://docs.other.org/"] };
      await validateFetchUrl("https://example.com/", allow);
      await validateFetchUrl("https://deep.docs.example.com/", allow);
      await validateFetchUrl("https://docs.other.org/x", allow);
      await rejects(() => validateFetchUrl("https://notexample.com/", allow), /not in the allowed/, "suffix trick");
      await rejects(() => validateFetchUrl("https://other.org/", allow), /not in the allowed/, "parent of allowed sub");
      const block = { lookup: PUBLIC, blockedDomains: ["evil.com"] };
      await validateFetchUrl("https://fine.com/", block);
      await rejects(() => validateFetchUrl("https://sub.evil.com/", block), /blocked/, "blocked subdomain");
    },
  },
  {
    label: "http path converts html to text and records the page",
    async run() {
      stubFetch(() => html(PAGE));
      const wf = createWebFetch({ provider: "openai", lookup: PUBLIC });
      const out = (await exec(wf.tool, { url: "https://example.com/page" })) as WebFetchResult;
      assert(out.type === "web_fetch_result", `type: ${JSON.stringify(out)}`);
      assert(out.title === "Hello & World", `title: ${out.title}`);
      assert(out.text!.includes("Tom & Jerry — “quoted” 'x' done"), `entities: ${out.text}`);
      assert(!out.text!.includes("alert"), "script removed");
      assert(!out.text!.includes("color"), "style removed");
      assert(!out.text!.includes("vector junk"), "svg removed");
      assert(!out.text!.includes("comment"), "comment removed");
      assert(out.text!.includes("- one\n- two"), `list items: ${out.text}`);
      assert(out.text!.includes("Last\nline"), `br: ${out.text}`);
      assert(out.text!.startsWith("Heading"), `leading: ${out.text!.slice(0, 20)}`);
      assert(out.mediaType === "text/html", `mediaType: ${out.mediaType}`);
      assert(out.truncated === false, "not truncated");
      assert(out.url === "https://example.com/page", `url: ${out.url}`);
      assert(!Number.isNaN(Date.parse(out.retrievedAt!)), "retrievedAt is a date");
      assert(wf.results.length === 1 && wf.results[0] === out, "recorded in results");
    },
  },
  {
    label: "htmlToText shapes blocks, lists and breaks",
    run() {
      const { title, text } = htmlToText("<h1>Title</h1><p>a<br>b</p><ul><li>x</li><li>y</li></ul>");
      assert(title === null, "no title");
      assert(text === "Title\n\na\nb\n\n- x\n- y", `got: ${JSON.stringify(text)}`);
      assert(htmlToText("&amp;lt; &#0; &#xZZ;").text === "&lt; &#0; &#xZZ;", "double-encoded and invalid entities survive");
    },
  },
  {
    label: "text budget truncates and flags; maxContentTokens alone derives it",
    async run() {
      const body = `<html><body><p>${"x".repeat(500)}</p></body></html>`;
      stubFetch(() => html(body));
      const chars = createWebFetch({ provider: "openai", lookup: PUBLIC, maxCharacters: 50 });
      const a = (await exec(chars.tool, { url: "https://example.com/" })) as WebFetchResult;
      assert(a.text!.length === 50 && a.truncated === true, `chars: len=${a.text!.length} truncated=${a.truncated}`);
      const tokens = createWebFetch({ provider: "openai", lookup: PUBLIC, maxContentTokens: 10 });
      const b = (await exec(tokens.tool, { url: "https://example.com/" })) as WebFetchResult;
      assert(b.text!.length === 40 && b.truncated === true, `tokens: len=${b.text!.length}`);
    },
  },
  {
    label: "json and plain text pass through; html served as text/plain is sniffed",
    async run() {
      stubFetch(() => new Response('{"a":1}', { status: 200, headers: { "content-type": "application/json" } }));
      const wf = createWebFetch({ provider: "google", lookup: PUBLIC });
      const j = (await exec(wf.tool, { url: "https://api.example.com/x" })) as WebFetchResult;
      assert(j.text === '{"a":1}' && j.title === null && j.mediaType === "application/json", `json: ${JSON.stringify(j)}`);

      stubFetch(() => new Response("<!doctype html><html><head><title>T</title></head><body>hi</body></html>", {
        status: 200, headers: { "content-type": "text/plain" },
      }));
      const s = (await exec(wf.tool, { url: "https://example.com/raw" })) as WebFetchResult;
      assert(s.title === "T" && s.text === "hi", `sniffed: ${JSON.stringify(s)}`);

      stubFetch(() => new Response(new Uint8Array([0xe9, 0x74, 0xe9]), {
        status: 200, headers: { "content-type": "text/plain; charset=iso-8859-1" },
      }));
      const l = (await exec(wf.tool, { url: "https://example.com/latin" })) as WebFetchResult;
      assert(l.text === "été", `charset honored: ${l.text}`);
    },
  },
  {
    label: "redirects are followed hop by hop and every hop is re-validated",
    async run() {
      const calls = stubFetch((url) =>
        url === "https://example.com/start" ? redirect("/next")
        : url === "https://example.com/next" ? redirect("https://final.example.org/page", 301)
        : html("<title>Final</title>"),
      );
      const wf = createWebFetch({ provider: "xai", lookup: PUBLIC });
      const out = (await exec(wf.tool, { url: "https://example.com/start" })) as WebFetchResult;
      assert(out.url === "https://final.example.org/page", `final url: ${out.url}`);
      assert(out.title === "Final", "landed on the final page");
      assert(calls.length === 3, `hops: ${calls.length}`);

      stubFetch(() => redirect("http://169.254.169.254/latest/meta-data"));
      const ssrf = (await exec(wf.tool, { url: "https://example.com/open-redirect" })) as { error?: string };
      assert(/private or reserved/.test(ssrf.error ?? ""), `redirect to metadata blocked: ${ssrf.error}`);

      const resolver = async (host: string) => (host === "internal.example.com" ? ["10.0.0.9"] : ["93.184.216.34"]);
      stubFetch(() => redirect("https://internal.example.com/"));
      const wf2 = createWebFetch({ provider: "xai", lookup: resolver });
      const hop = (await exec(wf2.tool, { url: "https://example.com/" })) as { error?: string };
      assert(/resolves to a private/.test(hop.error ?? ""), `redirect to private host blocked: ${hop.error}`);

      let n = 0;
      stubFetch(() => redirect(`https://example.com/loop${n++}`));
      const loop = (await exec(wf.tool, { url: "https://example.com/loop" })) as { error?: string };
      assert(/Too many redirects/.test(loop.error ?? ""), `loop: ${loop.error}`);

      stubFetch(() => new Response(null, { status: 302 }));
      const bare = (await exec(wf.tool, { url: "https://example.com/bare" })) as { error?: string };
      assert(/without a Location/.test(bare.error ?? ""), `bare redirect: ${bare.error}`);
    },
  },
  {
    label: "http errors, unsupported types and pdfs come back as errors, not throws",
    async run() {
      // Seven failing calls below; keep them under the budget so the last
      // ones don't trip the maxUses guard instead of the case under test.
      const wf = createWebFetch({ provider: "openai", lookup: PUBLIC, maxUses: 20 });
      stubFetch(() => new Response("gone", { status: 404 }));
      const nf = (await exec(wf.tool, { url: "https://example.com/missing" })) as { error?: string };
      assert(/HTTP 404/.test(nf.error ?? ""), `404: ${nf.error}`);

      stubFetch(() => new Response(new Uint8Array([1, 2, 3]), { status: 200, headers: { "content-type": "image/png" } }));
      const img = (await exec(wf.tool, { url: "https://example.com/a.png" })) as { error?: string };
      assert(/Unsupported content type: image\/png/.test(img.error ?? ""), `png: ${img.error}`);

      stubFetch(() => new Response("%PDF-1.4", { status: 200, headers: { "content-type": "application/pdf" } }));
      const pdf = (await exec(wf.tool, { url: "https://example.com/a.pdf" })) as { error?: string };
      assert(/PDF/.test(pdf.error ?? ""), `pdf: ${pdf.error}`);

      (globalThis as any).fetch = async () => {
        throw Object.assign(new TypeError("fetch failed"), {
          cause: Object.assign(new Error("certificate has expired"), { code: "CERT_HAS_EXPIRED" }),
        });
      };
      const down = (await exec(wf.tool, { url: "https://example.com/" })) as { error?: string };
      assert(
        /Could not connect to example\.com: certificate has expired \(CERT_HAS_EXPIRED\)/.test(down.error ?? ""),
        `transport cause surfaced: ${down.error}`,
      );

      (globalThis as any).fetch = async () => {
        throw new DOMException("The operation was aborted due to timeout", "TimeoutError");
      };
      const slow = (await exec(wf.tool, { url: "https://example.com/" })) as { error?: string };
      assert(/timed out/.test(slow.error ?? ""), `timeout: ${slow.error}`);

      (globalThis as any).fetch = async () => { throw new TypeError("fetch failed"); };
      const bare = (await exec(wf.tool, { url: "https://example.com/" })) as { error?: string };
      assert(/fetch failed/.test(bare.error ?? ""), `no cause falls back to message: ${bare.error}`);

      const bad = (await exec(wf.tool, { url: "file:///etc/passwd" })) as { error?: string };
      assert(/http\(s\)/.test(bad.error ?? ""), `guard surfaces to the model: ${bad.error}`);
      assert(wf.results.length === 0, "failures record nothing");
    },
  },
  {
    label: "maxUses is enforced in-process on the http path",
    async run() {
      stubFetch(() => html("<title>a</title>"));
      const wf = createWebFetch({ provider: "openai", lookup: PUBLIC, maxUses: 1 });
      await exec(wf.tool, { url: "https://example.com/1" });
      const blocked = (await exec(wf.tool, { url: "https://example.com/2" })) as { error?: string };
      assert(/budget exhausted/.test(blocked.error ?? ""), `second call refused: ${blocked.error}`);
      assert(wf.results.length === 1, "refused call adds no results");
    },
  },
  {
    label: "capture() is a no-op on the http path (no double-count)",
    async run() {
      stubFetch(() => html("<title>a</title>"));
      const wf = createWebFetch({ provider: "openai", lookup: PUBLIC });
      await exec(wf.tool, { url: "https://example.com/" });
      wf.capture([
        {
          type: "tool-result",
          toolName: WEB_FETCH_TOOL_NAME,
          output: { type: "web_fetch_result", url: "https://example.com/", content: { title: "a", source: { type: "text", data: "a" } } },
        },
      ]);
      assert(wf.results.length === 1, `expected 1 result, got ${wf.results.length}`);
    },
  },
  {
    label: "captureNativeFetchResults walks output and result shapes, skips errors and junk",
    run() {
      const target: WebFetchResult[] = [];
      captureNativeFetchResults(
        [
          {
            type: "tool-result",
            toolName: WEB_FETCH_TOOL_NAME,
            output: {
              type: "web_fetch_result",
              url: "https://a.com",
              retrievedAt: "2026-09-10T00:00:00Z",
              content: { type: "document", title: "A", source: { type: "text", mediaType: "text/plain", data: "page text" } },
            },
          },
          {
            type: "tool-result",
            toolName: WEB_FETCH_TOOL_NAME,
            result: {
              type: "web_fetch_result",
              url: "https://b.com/doc.pdf",
              retrieved_at: null,
              content: { type: "document", title: null, source: { type: "base64", media_type: "application/pdf", data: "JVBERi0=" } },
            },
          },
          {
            type: "tool-result",
            toolName: WEB_FETCH_TOOL_NAME,
            isError: true,
            output: { type: "web_fetch_tool_result_error", errorCode: "url_not_accessible" },
          },
          { type: "tool-result", toolName: "other_tool", output: { type: "web_fetch_result", url: "https://nope.com" } },
          { type: "tool-result", toolName: WEB_FETCH_TOOL_NAME, output: "not an object" },
          { type: "text", text: "hi" },
        ],
        target,
      );
      assert(target.length === 2, `expected 2, got ${target.length}`);
      assert(target[0].text === "page text" && target[0].title === "A", "text source captured");
      assert(target[0].mediaType === "text/plain" && target[0].retrievedAt === "2026-09-10T00:00:00Z", "fields mapped");
      assert(target[1].text === undefined, "pdf base64 is not surfaced as text");
      assert(target[1].mediaType === "application/pdf" && target[1].title === null, "snake_case shape tolerated");
      captureNativeFetchResults("not an array", target);
      assert(target.length === 2, "non-array content ignored");
    },
  },
];

let passed = 0;
let failed = 0;

for (const tc of tests) {
  try {
    await tc.run();
    console.log(`✅ PASS: ${tc.label}`);
    passed++;
  } catch (err: any) {
    console.error(`❌ FAIL: ${tc.label}`);
    console.error(`   ${err.message}`);
    failed++;
  }
}

console.log(`\nResults: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
