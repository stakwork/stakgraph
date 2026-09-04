import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { z } from "zod";
import { defineStep } from "./core.js";
import { createRegistry } from "./steps/registry.js";
import { exprRoots, templateExprs } from "./expr.js";
import { validateWorkflowYaml, type ValidationResult } from "./validate.js";

describe("exprRoots", () => {
  it("returns scope roots, not member names or lambda params", () => {
    assert.deepEqual(exprRoots(" fetch.body.name "), ["fetch"]);
    assert.deepEqual(exprRoots("search.map(n => n.ref_id)"), ["search"]);
    assert.deepEqual(exprRoots("prs.filter(p => p.merged && input.strict).map(p => p.title)").sort(), ["input", "prs"]);
    assert.deepEqual(exprRoots("a?.b ?? params.x"), ["a", "params"]);
    assert.deepEqual(exprRoots("$current.status === 'done' && true"), ["$current"]);
    assert.deepEqual(exprRoots("items[0].x + 1"), ["items"]);
    assert.deepEqual(exprRoots("null"), []);
  });
  it("templateExprs pulls every {{ }} body", () => {
    assert.deepEqual(templateExprs("a {{ x.y }} b {{z}}"), [" x.y ", "z"]);
    assert.deepEqual(templateExprs("plain"), []);
  });
});

describe("validateWorkflowYaml", () => {
  const registryP = createRegistry([
    defineStep({
      type: "echo",
      description: "echo",
      input: z.object({ message: z.string(), times: z.number().optional() }),
      output: z.any(),
      async run(cfg) { return cfg; },
    }),
  ]);
  async function v(yaml: string, workflows?: Array<{ name: string; versions: string[] }>): Promise<ValidationResult> {
    return validateWorkflowYaml(yaml, { registry: await registryP, workflows });
  }
  const msgs = (r: ValidationResult) => ({ errors: r.errors.map((e) => e.message), warnings: r.warnings.map((w) => w.message) });
  const has = (xs: string[], re: RegExp) => xs.some((m) => re.test(m));

  it("accepts a well-formed workflow (with if-gate, subflow, loop, onError)", async () => {
    const r = await v(`
name: good
params: { greeting: hi }
steps:
  - id: fetch
    type: http
    config: { url: "https://example.com/{{ input.id }}" }
  - id: check
    type: if
    config: { cond: "{{ fetch.status === 200 }}" }
  - id: yes_branch
    type: echo
    config: { message: "{{ params.greeting }} {{ fetch.body.name }}" }
    depends: check
    when: true
  - id: retry_loop
    type: loop
    config:
      until: "{{ $current.message === 'x' }}"
      maxIterations: 3
      body: { id: body, type: echo, config: { message: "{{ $current }}" } }
    depends: check
    when: false
  - id: each
    type: foreach
    config:
      items: "{{ fetch.body.items }}"
      body: { id: one, type: echo, config: { message: "{{ $index }}: {{ $current.name }}" } }
    depends: [yes_branch, retry_loop]
  - id: child
    type: subflow
    config: { workflow: other, version: v1, input: { x: "{{ each }}" } }
    options:
      retry: { max: 2, delayMs: 10 }
      onError: { id: oops, type: log, config: { message: "failed: {{ $error.message }}" } }
`, [{ name: "other", versions: ["v1"] }]);
    assert.deepEqual(msgs(r), { errors: [], warnings: [] });
    assert.equal(r.ok, true);
    assert.equal(r.summary.steps, 6);
    assert.ok(r.summary.stepTypes.includes("log")); // nested onError counted
  });

  it("rejects YAML / shape problems", async () => {
    assert.match((await v("name: x\nsteps:\n  - id: a\n    type: echo\n    config:\n      message: {{ input.m }}\n")).errors[0]!.message, /unquoted template/);
    assert.match((await v("just a string")).errors[0]!.message, /mapping/);
    assert.match((await v("steps: []")).errors.map((e) => e.message).join("|"), /`name` is required.*`steps` must be a non-empty/);
  });

  it("flags ids: missing, duplicate, unreferenceable", async () => {
    const r = await v(`
name: ids
steps:
  - type: echo
    config: { message: a }
  - id: my-step
    type: echo
    config: { message: b }
  - id: dup
    type: echo
    config: { message: c }
  - id: dup
    type: echo
    config: { message: d }
`);
    const { errors } = msgs(r);
    assert.ok(has(errors, /`id` is required/));
    assert.ok(has(errors, /"my-step" must be alphanumeric/));
    assert.ok(has(errors, /Duplicate step id "dup"/));
  });

  it("flags unknown step types, unknown depends, self-deps and cycles", async () => {
    const r = await v(`
name: deps
steps:
  - id: a
    type: nope
    config: {}
  - id: b
    type: echo
    config: { message: x }
    depends: [a, ghost]
  - id: c
    type: echo
    config: { message: x }
    depends: [d, c]
  - id: d
    type: echo
    config: { message: x }
    depends: c
`);
    const { errors } = msgs(r);
    assert.ok(has(errors, /Unknown step type "nope"/));
    assert.ok(has(errors, /depends on unknown step "ghost"/));
    assert.ok(has(errors, /"c" depends on itself/));
    assert.ok(has(errors, /Dependency cycle: c → d → c|Dependency cycle: d → c → d/));
    assert.equal(r.ok, false);
  });

  it("checks template roots and syntax, warns on non-upstream refs", async () => {
    const r = await v(`
name: tpl
steps:
  - id: a
    type: echo
    config: { message: "{{ input.x }}" }
  - id: b
    type: echo
    config: { message: "{{ c.out }} and {{ typo.x }} and {{ a.b # }}" }
    depends: []
  - id: c
    type: echo
    config: { message: "{{ a.message }}" }
    depends: []
`);
    const { errors, warnings } = msgs(r);
    assert.ok(has(errors, /unknown root "typo"/));
    assert.ok(has(errors, /Template syntax error/));
    assert.ok(has(warnings, /references step "c" which is not an upstream dependency/));
    assert.ok(has(warnings, /references step "a" which is not an upstream dependency/));
    assert.ok(!has(errors, /unknown root "c"/));
  });

  it("loop/error variables are only valid in their nested context", async () => {
    const r = await v(`
name: vars
steps:
  - id: top
    type: echo
    config: { message: "{{ $current }} {{ $error.message }}" }
  - id: each
    type: foreach
    config:
      items: "{{ input.items }}"
      body: { id: one, type: echo, config: { message: "{{ $index }} {{ $error }}" } }
`);
    const { errors } = msgs(r);
    assert.ok(has(errors, /unknown root "\$current"/));
    assert.ok(has(errors, /unknown root "\$error"/));
    assert.ok(!has(errors, /unknown root "\$index"/));
  });

  it("checks config against the schema, skipping template-valued fields; warns on unknown fields", async () => {
    const r = await v(`
name: cfg
steps:
  - id: a
    type: echo
    config: { times: "{{ input.n }}", mesage: oops }
  - id: b
    type: echo
    config: { message: fine, times: "3" }
  - id: c
    type: echo
    config: { message: "{{ input.m }}", times: "{{ params.n }}" }
`);
    const { errors, warnings } = msgs(r);
    // a: `message` missing (an error); `times` is templated (skipped)
    assert.ok(r.errors.some((e) => e.path === "steps[0].config.message"), JSON.stringify(errors));
    assert.ok(!r.errors.some((e) => e.path === "steps[0].config.times"));
    assert.ok(has(warnings, /Unknown config field "mesage"/));
    // b: literal "3" for a number field is an error
    assert.ok(r.errors.some((e) => e.path === "steps[1].config.times"));
    // c: fully templated → no config errors
    assert.ok(!r.errors.some((e) => e.path.startsWith("steps[2]")));
  });

  it("when: errors without depends, warns without an if gate; loop/foreach/subflow requirements", async () => {
    const r = await v(`
name: gates
steps:
  - id: lonely
    type: echo
    config: { message: x }
    depends: []
    when: true
  - id: gated
    type: echo
    config: { message: x }
    depends: lonely
    when: false
  - id: l
    type: loop
    config: { body: { id: b, type: echo, config: { message: x } } }
  - id: f
    type: foreach
    config: { body: { id: b2, type: echo, config: { message: x } } }
  - id: s
    type: subflow
    config: { workflow: missing }
  - id: s2
    type: subflow
    config: { workflow: other, version: v9 }
  - id: bad_retry
    type: echo
    config: { message: x }
    options: { retry: { max: "3" } }
`, [{ name: "other", versions: ["v1", "v2"] }]);
    const { errors, warnings } = msgs(r);
    assert.ok(has(errors, /"lonely" has `when` but no `depends`/));
    assert.ok(has(warnings, /"gated" has `when` but none of its depends .* is an `if` gate/));
    assert.ok(has(errors, /loop step requires an `until`/));
    assert.ok(has(errors, /foreach step requires `items`/));
    assert.ok(has(errors, /Subflow target "missing" is not a published workflow/));
    assert.ok(has(errors, /Subflow "other" has no version "v9". Available: v1, v2/));
    assert.ok(has(errors, /`retry` must be/));
  });

  it("skips the subflow-existence check when no workflow list is supplied", async () => {
    const r = await v(`
name: nowf
steps:
  - id: s
    type: subflow
    config: { workflow: whatever, input: {} }
`);
    assert.deepEqual(msgs(r).errors, []);
  });
});
