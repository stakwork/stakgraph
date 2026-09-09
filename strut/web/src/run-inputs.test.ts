import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { deriveInputBindings, refsInExpr, stepTypesIn } from "./run-inputs";
import type { FieldDesc } from "./api";
import type { StepData } from "./flow-to-canvas";

const str = (name: string, required = true): FieldDesc => ({ name, kind: "string", required });
const schemas: Record<string, FieldDesc[]> = {
  "media/detect-type": [str("url")],
  "html/extract": [str("url"), { name: "maxChars", kind: "number", required: false, default: 40000 }],
  http: [str("url"), { name: "method", kind: "enum", required: false, enumValues: ["GET", "POST"] }],
};
const schemaFor = (t: string) => schemas[t];

const keys = (b: ReturnType<typeof deriveInputBindings>) => b.map((x) => x.inputKey);
const req = (b: ReturnType<typeof deriveInputBindings>) => b.filter((x) => x.field.required).map((x) => x.inputKey);

describe("refsInExpr", () => {
  it("bare reference is required", () => {
    assert.deepEqual(refsInExpr(" input.media_url "), [{ key: "media_url", optional: false }]);
  });
  it("guards make a reference optional", () => {
    for (const expr of [
      "input.ns || 'default'",
      "input.ns ?? 'default'",
      "input?.ns",
      "input.ns ? 'a' : 'b'",
      "input.ns?.deep",
      "other.x || input.ns",
      "other.x ?? input.ns",
    ]) {
      assert.deepEqual(refsInExpr(expr), [{ key: "ns", optional: true }], expr);
    }
  });
  it("reads bracket access", () => {
    assert.deepEqual(refsInExpr('input["media url"]'), [{ key: "media url", optional: false }]);
    assert.deepEqual(refsInExpr("input['x'] || 1"), [{ key: "x", optional: true }]);
  });
  it("does not match other identifiers ending in input", () => {
    assert.deepEqual(refsInExpr("evalInput.owner + subinput.x"), []);
  });
  it("collects every reference in a chain, in order", () => {
    assert.deepEqual(
      refsInExpr("input.title || yt?.title || input.media_url"),
      [
        { key: "title", optional: true },
        { key: "media_url", optional: true },
      ],
    );
  });
});

describe("deriveInputBindings", () => {
  // The shape of content-ingest-agentic-graph v6: an artifacts/dir step with no
  // config leads, and every input is consumed further down.
  const v6: StepData[] = [
    { id: "dir", type: "artifacts/dir", config: {} },
    { id: "register_ns", type: "graph/register-namespace", depends: [], config: { namespace: "{{ input.namespace || 'default' }}" } },
    { id: "detect", type: "media/detect-type", depends: [], config: { url: "{{ input.media_url }}" } },
    {
      id: "page",
      type: "html/extract",
      config: { url: "{{ input.media_url }}", maxChars: "{{ params.maxContentChars }}" },
      options: { onError: { id: "page_failed", type: "pack", config: { fetch_error: "{{ $error.message }}" } } },
    },
    {
      id: "content",
      type: "pack",
      config: { title: "{{ input.title || yt_meta?.body?.title || page?.title || input.media_url }}" },
    },
    {
      id: "docnode",
      type: "graph/create-node",
      config: { node_data: { source_link: "{{ input.media_url }}" }, namespace: "{{ input.namespace || 'default' }}" },
    },
    {
      id: "extract",
      type: "agent",
      config: {
        prompt:
          "media_url: {{ input.media_url }}\nDomain: {{ input.domain || '' }}\nHint: {{ input.context_hint || '(none provided)' }}\n" +
          "Use namespace \"{{ input.namespace || 'default' }}\" on every write.",
        model: "{{ params.model }}",
      },
    },
  ];

  it("finds inputs in every step, not just the first", () => {
    const b = deriveInputBindings(v6, schemaFor);
    assert.deepEqual(keys(b), ["media_url", "namespace", "title", "domain", "context_hint"]);
  });

  it("orders required keys first and marks guarded keys optional", () => {
    const b = deriveInputBindings(v6, schemaFor);
    assert.deepEqual(req(b), ["media_url"]);
    assert.equal(b.find((x) => x.inputKey === "namespace")!.field.required, false);
  });

  it("borrows the schema of the first exact-slot match", () => {
    const b = deriveInputBindings(v6, schemaFor);
    const url = b.find((x) => x.inputKey === "media_url")!.field;
    assert.equal(url.kind, "string");
    assert.equal(url.name, "media_url");
  });

  it("keeps a typed field's kind, default and enum values", () => {
    const steps: StepData[] = [
      { id: "a", type: "html/extract", config: { url: "https://x", maxChars: "{{ input.limit }}" } },
      { id: "b", type: "http", config: { url: "https://x", method: "{{ input.verb }}" } },
    ];
    const b = deriveInputBindings(steps, schemaFor);
    assert.deepEqual(b.map((x) => [x.inputKey, x.field.kind, x.field.required]), [
      ["limit", "number", false],
      ["verb", "enum", false],
    ]);
    assert.equal(b[0]!.field.default, 40000);
    assert.deepEqual(b[1]!.field.enumValues, ["GET", "POST"]);
  });

  it("an optional step slot becomes required if another step needs the key bare", () => {
    const steps: StepData[] = [
      { id: "a", type: "html/extract", config: { url: "https://x", maxChars: "{{ input.limit }}" } },
      { id: "b", type: "pack", config: { note: "limit is {{ input.limit }}" } },
    ];
    assert.deepEqual(req(deriveInputBindings(steps, schemaFor)), ["limit"]);
  });

  it("an unknown schema still surfaces the key, untyped and required when bare", () => {
    const steps: StepData[] = [{ id: "a", type: "custom/thing", config: { url: "{{ input.target }}" } }];
    const b = deriveInputBindings(steps, () => undefined);
    assert.deepEqual(b, [{ inputKey: "target", field: { name: "target", kind: "string", required: true } }]);
  });

  it("ignores prose outside {{ }} and params/step references", () => {
    const steps: StepData[] = [
      { id: "a", type: "agent", config: { prompt: "The workflow's input.media_url field. {{ params.model }} {{ detect.media_type }}" } },
    ];
    assert.deepEqual(deriveInputBindings(steps, schemaFor), []);
  });

  it("walks onError handlers and loop bodies", () => {
    const steps: StepData[] = [
      {
        id: "a",
        type: "http",
        config: { url: "https://x" },
        options: { onError: { id: "a_failed", type: "pack", config: { who: "{{ input.owner }}" } } },
      },
      {
        id: "l",
        type: "loop",
        config: { maxIterations: 3, body: { id: "poll", type: "http", config: { url: "{{ input.pollUrl }}" } } },
      },
    ];
    const b = deriveInputBindings(steps, schemaFor);
    assert.deepEqual(keys(b), ["owner", "pollUrl"]);
    assert.equal(b[1]!.field.kind, "string");
    assert.deepEqual(stepTypesIn(steps), ["http", "pack", "loop"]);
  });

  it("dedupes a key referenced by many steps", () => {
    const steps: StepData[] = [
      { id: "a", type: "pack", config: { x: "{{ input.k }}" } },
      { id: "b", type: "pack", config: { y: "{{ input.k || 1 }}", z: ["{{ input.k }}"] } },
    ];
    assert.deepEqual(keys(deriveInputBindings(steps, schemaFor)), ["k"]);
  });

  it("returns nothing for a flow without input references", () => {
    assert.deepEqual(deriveInputBindings([{ id: "dir", type: "artifacts/dir", config: {} }], schemaFor), []);
  });
});
