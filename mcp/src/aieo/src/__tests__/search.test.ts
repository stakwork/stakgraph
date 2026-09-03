import {
  createWebSearch,
  stripCitations,
  linkifyCitations,
  resolveSearchBackend,
  captureNativeResults,
  WEB_SEARCH_TOOL_NAME,
  type WebSearchResult,
} from "../search.js";

type TestCase = { label: string; run: () => Promise<void> | void };

function assert(cond: unknown, msg: string): void {
  if (!cond) throw new Error(msg);
}

/** Stub Exa's HTTP endpoint with a fixed page of results. */
function stubExa(pages: Array<Array<{ url: string; title?: string }>>) {
  let call = 0;
  (globalThis as any).fetch = async () => ({
    ok: true,
    json: async () => ({
      results: (pages[call++] ?? []).map((r) => ({
        url: r.url,
        title: r.title ?? null,
        publishedDate: null,
        text: "body text",
      })),
    }),
  });
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const exec = (t: any, input: any) => t.execute(input, { toolCallId: "t", messages: [] });

const tests: TestCase[] = [
  {
    label: "anthropic keeps the native tool; others fall to exa",
    run() {
      assert(resolveSearchBackend("anthropic") === "anthropic", "anthropic → anthropic");
      for (const p of ["openai", "google", "openrouter", "xai"] as const) {
        assert(resolveSearchBackend(p) === "exa", `${p} → exa`);
      }
    },
  },
  {
    label: "no EXA key → tool is undefined, not a throw",
    run() {
      delete process.env.EXA_API_KEY;
      const ws = createWebSearch({ provider: "openai" });
      assert(ws.tool === undefined, "tool should be undefined");
      assert(ws.backend === undefined, "backend should be undefined");
      assert(ws.promptSnippet === "", "no snippet without a tool");
    },
  },
  {
    label: "exa path assigns flat 1-based indices across successive calls",
    async run() {
      process.env.EXA_API_KEY = "test-key";
      stubExa([
        [{ url: "https://a.com" }, { url: "https://b.com" }],
        [{ url: "https://c.com" }],
      ]);
      const ws = createWebSearch({ provider: "openai", citations: true });
      const first = (await exec(ws.tool, { query: "one" })) as WebSearchResult[];
      const second = (await exec(ws.tool, { query: "two" })) as WebSearchResult[];
      assert(first.map((r) => r.index).join() === "1,2", `first indices: ${first.map((r) => r.index)}`);
      assert(second[0].index === 3, `second index: ${second[0].index}`);
      assert(ws.results.length === 3, `results length: ${ws.results.length}`);
      assert(ws.results[2].url === "https://c.com", "results in citation order");
      assert(ws.native === false, "exa path is not native");
      assert(ws.promptSnippet.includes("index"), "shim ships citation instructions");
    },
  },
  {
    label: "capture() is a no-op on the exa path (no double-count)",
    async run() {
      process.env.EXA_API_KEY = "test-key";
      stubExa([[{ url: "https://a.com" }]]);
      const ws = createWebSearch({ provider: "openai" });
      await exec(ws.tool, { query: "one" });
      ws.capture([
        { type: "tool-result", toolName: WEB_SEARCH_TOOL_NAME, output: [{ url: "https://a.com" }] },
      ]);
      assert(ws.results.length === 1, `expected 1 result, got ${ws.results.length}`);
    },
  },
  {
    label: "maxUses is enforced in-process on the exa path",
    async run() {
      process.env.EXA_API_KEY = "test-key";
      stubExa([[{ url: "https://a.com" }], [{ url: "https://b.com" }]]);
      const ws = createWebSearch({ provider: "openai", maxUses: 1 });
      await exec(ws.tool, { query: "one" });
      const blocked = (await exec(ws.tool, { query: "two" })) as { error?: string };
      assert(!!blocked.error, "second call should be refused");
      assert(ws.results.length === 1, "refused call adds no results");
    },
  },
  {
    label: "search failures return an error to the model, not a throw",
    async run() {
      process.env.EXA_API_KEY = "test-key";
      (globalThis as any).fetch = async () => ({ ok: false, status: 429, text: async () => "slow down" });
      const ws = createWebSearch({ provider: "openai" });
      const out = (await exec(ws.tool, { query: "one" })) as { error?: string };
      assert(out.error?.includes("429"), `expected 429 in error, got: ${out.error}`);
    },
  },
  {
    label: "captureNativeResults walks both output and result shapes, skips junk",
    run() {
      const target: WebSearchResult[] = [];
      captureNativeResults(
        [
          { type: "tool-result", toolName: WEB_SEARCH_TOOL_NAME, output: [{ url: "https://a.com", title: "A" }] },
          { type: "tool-result", toolName: WEB_SEARCH_TOOL_NAME, result: [{ url: "https://b.com" }] },
          { type: "tool-result", toolName: "other_tool", output: [{ url: "https://nope.com" }] },
          { type: "tool-result", toolName: WEB_SEARCH_TOOL_NAME, output: { notAnArray: true } },
          { type: "text", text: "hi" },
        ],
        target,
      );
      assert(target.length === 2, `expected 2, got ${target.length}`);
      assert(target[0].title === "A" && target[1].title === null, "title normalized to null");
      assert(target[0].index === 1 && target[1].index === 2, "native results get indices too");
    },
  },
  {
    label: "both backends ship citation instructions when asked",
    run() {
      process.env.EXA_API_KEY = "test-key";
      const native = createWebSearch({ provider: "anthropic", apiKey: "sk-ant-fake", citations: true });
      assert(native.native === true, "anthropic is native");
      assert(native.promptSnippet.includes("cite index"), "native needs instructions too");
      const shim = createWebSearch({ provider: "xai", searchApiKey: "exa-fake", citations: true });
      assert(shim.promptSnippet.includes("index"), "shim needs instructions");
      assert(shim.promptSnippet.includes("REQUIRED"), "anchor text is demanded");
    },
  },
  {
    label: "citations off by default: no instructions, output is plain prose",
    run() {
      process.env.EXA_API_KEY = "test-key";
      const ws = createWebSearch({ provider: "xai" });
      assert(ws.promptSnippet === "", "no citation instructions by default");
      const out = ws.formatOutput(
        'Sphinx runs on Lightning. <cite index="1">per the docs</cite> Also fast. <cite index="2"></cite>',
      );
      assert(
        out.content === "Sphinx runs on Lightning. per the docs Also fast.",
        `got: ${out.content}`,
      );
      assert(!out.content.includes("cite"), "no markup survives");
    },
  },
  {
    label: "citations: true restores instructions and markdown links",
    run() {
      process.env.EXA_API_KEY = "test-key";
      const ws = createWebSearch({ provider: "xai", citations: true });
      assert(ws.promptSnippet.includes("cite index"), "instructions restored");
      ws.results.push({
        url: "https://a.com",
        title: "A",
        pageAge: null,
        index: 1,
        type: "web_search_result",
      });
      const out = ws.formatOutput('claim <cite index="1">source</cite>');
      assert(out.content === "claim [source](https://a.com)", `got: ${out.content}`);
      assert(out.converted === 1, "counted");
    },
  },
  {
    label: "stripCitations handles empty markers, anchors and truncated tags",
    run() {
      // Trailing empty marker takes its leading space with it.
      assert(
        stripCitations('a claim. <cite index="4"></cite>') === "a claim.",
        "empty marker leaves no trailing space",
      );
      // Anthropic's multi-part index form.
      assert(
        stripCitations('<cite index="2-1,3-4">the words</cite> rest') === "the words rest",
        "multi-part anchor collapses",
      );
      // Stream cut mid-tag: no half-tag survives.
      assert(
        stripCitations('a claim <cite index="3">trailing') === "a claim trailing",
        "orphan open tag swept",
      );
      assert(stripCitations("plain text") === "plain text", "untouched when no tags");
    },
  },
  {
    label: "empty anchor degrades to a [N] footnote, not an empty link",
    run() {
      const results: WebSearchResult[] = [
        { url: "https://a.com", title: "A", pageAge: null, type: "web_search_result" },
        { url: "https://b.com", title: "B", pageAge: null, type: "web_search_result" },
      ];
      // grok-4 emits exactly this shape: a trailing marker, no anchor.
      const out = linkifyCitations('a claim. <cite index="2"></cite>', results);
      assert(out.content === "a claim. [[2]](https://b.com)", `got: ${out.content}`);
      assert(out.converted === 1, "still counts as converted");
    },
  },
  {
    label: "linkifyCitations handles shim and anthropic multi-part indices",
    run() {
      const results: WebSearchResult[] = [
        { url: "https://a.com", title: "A", pageAge: null, type: "web_search_result" },
        { url: "https://b.com", title: "B", pageAge: null, type: "web_search_result" },
      ];
      const out = linkifyCitations(
        'see <cite index="1">first</cite> and <cite index="2-1,3-4">second</cite> and <cite index="9">gone</cite>',
        results,
      );
      assert(out.content.includes("[first](https://a.com)"), `shim form: ${out.content}`);
      assert(out.content.includes("[second](https://b.com)"), `multi-part form: ${out.content}`);
      assert(out.content.includes(" and gone"), "out-of-range collapses to anchor");
      assert(!out.content.includes("<cite"), "no raw cite markup survives");
      assert(out.converted === 2 && out.skipped === 1, `counts: ${out.converted}/${out.skipped}`);
    },
  },
  {
    label: "linkifyCitations escapes ] in model-generated anchor text",
    run() {
      const results: WebSearchResult[] = [
        { url: "https://a.com", title: "A", pageAge: null, type: "web_search_result" },
      ];
      const out = linkifyCitations('<cite index="1">a]b</cite>', results);
      assert(out.content === "[a\\]b](https://a.com)", `got: ${out.content}`);
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
