import { describe, it, beforeEach, afterEach } from "node:test";
import http from "node:http";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync, readFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import { coreRegistry } from "../registry.js";
import { defineStep, type StepContext, type StepRegistry } from "../../core.js";
import agent, {
  repoTree,
  textEdit,
  buildRegistryTools,
  expandAgentTools,
  wrapToolsWithEmit,
  maskSecretValues,
  maskDeep,
  wrapToolsWithMask,
  classifyFinalAnswerStop,
  isTransientStreamError,
} from "./agent.js";

// These tests are OFFLINE: they exercise registration, the input schema, and the
// config-validation guards in run() that fire BEFORE any model call. The actual
// generation loop needs a provider key + network and is covered by lab smokes.

describe("core agent step", () => {
  it("is registered in the core registry", () => {
    const reg = coreRegistry();
    assert.ok(reg["agent"], "agent should be a core step");
    assert.equal(reg["agent"]!.type, "agent");
  });

  it("applies input defaults (maxSteps, toolFilter)", () => {
    const cfg = (agent.input as any).parse({
      cwd: "/tmp/x",
      system: "you are a tester",
      prompt: "do the thing",
    });
    assert.equal(cfg.maxSteps, 40);
    assert.deepEqual(cfg.toolFilter, []);
  });

  it("rejects setting BOTH schema and finalAnswer", async () => {
    const cfg = (agent.input as any).parse({
      cwd: "/tmp/x",
      system: "s",
      prompt: "p",
      finalAnswer: "return the answer",
      schema: { type: "object", properties: {} },
    });
    await assert.rejects(() => (agent.run as any)(cfg, {}), /EITHER `schema`.*OR `finalAnswer`/);
  });

  it("rejects an unknown provider", async () => {
    const cfg = (agent.input as any).parse({
      cwd: "/tmp/x",
      system: "s",
      prompt: "p",
      provider: "not-a-provider",
    });
    await assert.rejects(() => (agent.run as any)(cfg, {}), /Unknown LLM provider/);
  });
});

describe("agentTools (buildRegistryTools — tools are steps)", () => {
  // A fake `tool()` factory: identity, so we can inspect the produced tool def
  // (description/inputSchema/execute) without importing the AI SDK.
  const fakeTool = (def: any) => def;

  const echoStep = defineStep({
    type: "demo/echo",
    description: "Echo the message back.",
    input: z.object({ msg: z.string() }),
    output: z.string(),
    async run(cfg) {
      return `got:${cfg.msg}`;
    },
  });
  const registry = { "demo/echo": echoStep } as StepRegistry;

  it("builds a tool per registry step, sanitizing the slash in the tool name", () => {
    const tools = buildRegistryTools(["demo/echo"], registry, undefined, fakeTool);
    assert.ok(tools["demo_echo"], "slash sanitized to underscore");
    assert.equal((tools["demo_echo"] as any).description, "Echo the message back.");
    assert.equal((tools["demo_echo"] as any).inputSchema, echoStep.input);
  });

  it("skips unknown step types (no throw)", () => {
    const tools = buildRegistryTools(["nope/missing", "demo/echo"], registry, undefined, fakeTool);
    assert.deepEqual(Object.keys(tools), ["demo_echo"]);
  });

  it("returns {} with no names or no registry", () => {
    assert.deepEqual(buildRegistryTools([], registry, undefined, fakeTool), {});
    assert.deepEqual(buildRegistryTools(["demo/echo"], undefined, undefined, fakeTool), {});
  });

  it("executes the step and returns its output (no emit here)", async () => {
    const ctx = {
      runId: "r1", path: "wf/diagnose", scope: {}, input: undefined,
      emit: async () => {}, services: undefined, registry,
    } as unknown as StepContext;
    const tools = buildRegistryTools(["demo/echo"], registry, ctx, fakeTool);
    const out = await (tools["demo_echo"] as any).execute({ msg: "hi" });
    assert.equal(out, "got:hi");
  });

  it("returns an Error string (not throw) on invalid tool input", async () => {
    const tools = buildRegistryTools(["demo/echo"], registry, undefined, fakeTool);
    const out = await (tools["demo_echo"] as any).execute({ wrong: 1 });
    assert.match(String(out), /Error: invalid input for "demo\/echo"/);
  });

  describe("glob expansion (expandAgentTools)", () => {
    const shoutStep = defineStep({
      type: "demo/shout",
      input: z.object({ msg: z.string() }),
      output: z.string(),
      async run(cfg) {
        return cfg.msg.toUpperCase();
      },
    });
    const otherStep = defineStep({
      type: "other/thing",
      input: z.object({}),
      output: z.any(),
      async run() {
        return null;
      },
    });
    const globReg = {
      "demo/echo": echoStep,
      "demo/shout": shoutStep,
      "other/thing": otherStep,
    } as StepRegistry;

    it("expands a namespace glob to every matching step type, sorted", () => {
      assert.deepEqual(expandAgentTools(["demo/*"], globReg), ["demo/echo", "demo/shout"]);
    });

    it("mixes globs and plain names, deduping (first occurrence wins)", () => {
      assert.deepEqual(
        expandAgentTools(["demo/echo", "demo/*", "other/thing"], globReg),
        ["demo/echo", "demo/shout", "other/thing"],
      );
    });

    it("a glob matching nothing expands to nothing (no throw)", () => {
      assert.deepEqual(expandAgentTools(["nope/*"], globReg), []);
    });

    it("does not treat regex metacharacters in the pattern as regex", () => {
      // "demo/e.ho" must NOT match "demo/echo" — dots are literal.
      assert.deepEqual(
        expandAgentTools(["demo/e.ho"], globReg),
        ["demo/e.ho"], // passes through as a plain (unknown) name
      );
    });

    it("buildRegistryTools consumes globs end-to-end", () => {
      const tools = buildRegistryTools(["demo/*"], globReg, undefined, fakeTool);
      assert.deepEqual(Object.keys(tools).sort(), ["demo_echo", "demo_shout"]);
    });
  });

  it("exposes the agent step itself as a tool (sub-agent recursion seam)", () => {
    const reg = coreRegistry();
    const tools = buildRegistryTools(["agent"], reg, undefined, fakeTool);
    assert.ok(tools["agent"], "the core agent step is grantable as an agentTool");
    assert.equal((tools["agent"] as any).inputSchema, reg["agent"]!.input);
  });

  it("nests a registry step's own emits under the tool-call span (child ctx path)", async () => {
    // A step that emits an event itself — stands in for a nested agent whose
    // own tool calls emit at `${ctx.path}/NNN-<tool>`.
    const emittingStep = defineStep({
      type: "demo/emitter",
      input: z.object({}),
      output: z.any(),
      async run(_cfg, sctx) {
        await (sctx.emit as any)({ type: "step.start", path: `${sctx.path}/001-inner`, stepType: "tool:inner" });
        return "ok";
      },
    });
    const reg = { "demo/emitter": emittingStep } as StepRegistry;
    const events: any[] = [];
    const ctx = {
      runId: "r1", path: "wf/parent-agent", scope: {}, input: undefined,
      emit: async (e: any) => { events.push(e); }, services: undefined, registry: reg,
    } as unknown as StepContext;

    const tools = buildRegistryTools(["demo/emitter"], reg, ctx, (d: any) => d);
    wrapToolsWithEmit(tools, ctx);
    await (tools["demo_emitter"] as any).execute({}, {});

    const paths = events.map((e) => e.path);
    // outer span from wrapToolsWithEmit…
    assert.ok(paths.includes("wf/parent-agent/001-demo_emitter"), `outer span missing: ${paths}`);
    // …and the step's own emit nests UNDER it (not as a flat sibling).
    assert.ok(
      paths.includes("wf/parent-agent/001-demo_emitter/001-inner"),
      `inner emit not nested: ${paths}`,
    );
  });

  it("registry tool keeps the parent ctx path when called without wrap options", async () => {
    const events: any[] = [];
    const probeStep = defineStep({
      type: "demo/probe",
      input: z.object({}),
      output: z.any(),
      async run(_cfg, sctx) {
        return sctx.path;
      },
    });
    const reg = { "demo/probe": probeStep } as StepRegistry;
    const ctx = {
      runId: "r1", path: "wf/agent", scope: {}, input: undefined,
      emit: async (e: any) => { events.push(e); }, services: undefined, registry: reg,
    } as unknown as StepContext;
    const tools = buildRegistryTools(["demo/probe"], reg, ctx, (d: any) => d);
    // Unwrapped (no wrapToolsWithEmit): no veinToolPath → parent path unchanged.
    assert.equal(await (tools["demo_probe"] as any).execute({}), "wf/agent");
  });
});

describe("wrapToolsWithEmit (per-call nested run events)", () => {
  function makeCtx(events: any[]): StepContext {
    return {
      runId: "r1", path: "wf/agent", scope: {}, input: undefined,
      emit: async (e: any) => { events.push(e); }, services: undefined,
    } as unknown as StepContext;
  }

  it("emits step.start/step.end around every tool, with a shared ordered counter", async () => {
    const events: any[] = [];
    const tools: Record<string, any> = {
      bash: { execute: async (i: any) => `ran:${i.command}` },
      assess: { execute: async () => ({ working: true }) },
    };
    wrapToolsWithEmit(tools, makeCtx(events));

    const a = await tools.bash.execute({ command: "ls" });
    const b = await tools.assess.execute({});
    assert.equal(a, "ran:ls");
    assert.deepEqual(b, { working: true }); // model still gets the REAL output

    assert.deepEqual(events.map((e) => [e.type, e.path, e.stepType]), [
      ["step.start", "wf/agent/001-bash", "tool:bash"],
      ["step.end", "wf/agent/001-bash", "tool:bash"],
      ["step.start", "wf/agent/002-assess", "tool:assess"],
      ["step.end", "wf/agent/002-assess", "tool:assess"],
    ]);
    assert.deepEqual(events[0].input, { command: "ls" });
    assert.equal(events[1].output, "ran:ls"); // event output is the summarized string
  });

  it("skips final_answer and provider-executed tools (no execute)", async () => {
    const events: any[] = [];
    const tools: Record<string, any> = {
      final_answer: { execute: async (i: any) => i.answer },
      web_search: { type: "provider-defined" }, // no execute
    };
    wrapToolsWithEmit(tools, makeCtx(events));
    await tools.final_answer.execute({ answer: "done" });
    assert.equal(events.length, 0);
  });

  it("emits step.error and rethrows when a tool throws", async () => {
    const events: any[] = [];
    const tools: Record<string, any> = { boom: { execute: async () => { throw new Error("nope"); } } };
    wrapToolsWithEmit(tools, makeCtx(events));
    await assert.rejects(() => tools.boom.execute({}), /nope/);
    assert.equal(events[0].type, "step.start");
    assert.equal(events[1].type, "step.error");
    assert.equal(events[1].error.message, "nope");
  });

  it("is a no-op without a runner ctx (in-code/test)", async () => {
    const tools: Record<string, any> = { bash: { execute: async () => "ok" } };
    const orig = tools.bash.execute;
    wrapToolsWithEmit(tools, undefined);
    assert.equal(tools.bash.execute, orig, "execute is left untouched");
  });
});

describe("repo_overview adaptive tree (repoTree)", () => {
  it("always shows every root entry and collapses noise dirs", () => {
    const files = [
      "package.json",
      "src/index.ts",
      "src/lib/a.ts",
      "prisma/migrations/0001_init/migration.sql",
      "prisma/migrations/0002_next/migration.sql",
      "node_modules/foo/index.js",
      "dist/bundle.js",
    ];
    const { text } = repoTree(files, { maxLines: 1000, maxDepth: 8 });
    // root entries present
    assert.ok(text.includes("package.json"), "root file shown");
    assert.ok(text.includes("src/"), "root dir shown");
    // noise dirs are shown but NOT expanded
    assert.ok(text.includes("migrations/"), "migrations dir shown");
    assert.ok(!text.includes("migration.sql"), "migration files collapsed away");
    assert.ok(text.includes("node_modules/") && !text.includes("foo"), "node_modules collapsed");
    assert.ok(text.includes("dist/") && !text.includes("bundle.js"), "dist collapsed");
  });

  it("deepens while under the line budget and steps back when it busts", () => {
    // 50 top-level dirs, each with a nested file: depth 1 = 50 lines, depth 2 = 100.
    const files: string[] = [];
    for (let i = 0; i < 50; i++) files.push(`dir${i}/sub/file${i}.ts`);

    const tight = repoTree(files, { maxLines: 60, maxDepth: 8 });
    assert.equal(tight.depth, 1, "depth-2 (100 lines) busts a 60-line budget → stay at depth 1");

    const roomy = repoTree(files, { maxLines: 1000, maxDepth: 8 });
    assert.ok(roomy.depth >= 2, "with room, it deepens past the root");
  });

  it("respects the hard depth cap", () => {
    const files = ["a/b/c/d/e/f/g/h/i/j/deep.ts"];
    const { depth } = repoTree(files, { maxLines: 10000, maxDepth: 3 });
    assert.equal(depth, 3, "never deeper than maxDepth even with budget to spare");
  });
});

describe("textEdit (str_replace_based_edit_tool handler)", () => {
  let cwd: string;
  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), "vein-textedit-"));
  });
  afterEach(() => {
    rmSync(cwd, { recursive: true, force: true });
  });

  it("views a file with 1-indexed line numbers", () => {
    writeFileSync(join(cwd, "a.txt"), "one\ntwo\nthree");
    const out = textEdit({ command: "view", path: "a.txt" }, cwd);
    assert.equal(out, "1: one\n2: two\n3: three");
  });

  it("views a line range", () => {
    writeFileSync(join(cwd, "a.txt"), "one\ntwo\nthree\nfour");
    assert.equal(textEdit({ command: "view", path: "a.txt", view_range: [2, 3] }, cwd), "2: two\n3: three");
    assert.equal(textEdit({ command: "view", path: "a.txt", view_range: [3, -1] }, cwd), "3: three\n4: four");
  });

  it("lists a directory on view", () => {
    writeFileSync(join(cwd, "z.txt"), "");
    writeFileSync(join(cwd, "a.txt"), "");
    const out = textEdit({ command: "view", path: "." }, cwd);
    assert.equal(out, "a.txt\nz.txt");
  });

  it("creates a new file (including nested dirs)", () => {
    const out = textEdit({ command: "create", path: "sub/dir/new.txt", file_text: "hi" }, cwd);
    assert.match(out, /Successfully created/);
    assert.equal(readFileSync(join(cwd, "sub/dir/new.txt"), "utf-8"), "hi");
  });

  it("str_replace replaces exactly one match", () => {
    writeFileSync(join(cwd, "a.txt"), "foo bar baz");
    const out = textEdit({ command: "str_replace", path: "a.txt", old_str: "bar", new_str: "QUX" }, cwd);
    assert.match(out, /Successfully replaced/);
    assert.equal(readFileSync(join(cwd, "a.txt"), "utf-8"), "foo QUX baz");
  });

  it("str_replace refuses zero matches", () => {
    writeFileSync(join(cwd, "a.txt"), "foo");
    const out = textEdit({ command: "str_replace", path: "a.txt", old_str: "nope", new_str: "x" }, cwd);
    assert.match(out, /No match found/);
    assert.equal(readFileSync(join(cwd, "a.txt"), "utf-8"), "foo");
  });

  it("str_replace refuses multiple matches", () => {
    writeFileSync(join(cwd, "a.txt"), "x x x");
    const out = textEdit({ command: "str_replace", path: "a.txt", old_str: "x", new_str: "y" }, cwd);
    assert.match(out, /Found 3 matches/);
    assert.equal(readFileSync(join(cwd, "a.txt"), "utf-8"), "x x x");
  });

  it("inserts text after a line (0 = top of file)", () => {
    writeFileSync(join(cwd, "a.txt"), "one\ntwo");
    textEdit({ command: "insert", path: "a.txt", insert_line: 0, insert_text: "ZERO" }, cwd);
    assert.equal(readFileSync(join(cwd, "a.txt"), "utf-8"), "ZERO\none\ntwo");
  });

  it("refuses paths that escape the working dir", () => {
    writeFileSync(join(cwd, "a.txt"), "secret");
    const out = textEdit({ command: "view", path: "../../../etc/passwd" }, cwd);
    assert.match(out, /escapes the working directory/);
  });

  it("returns File not found for missing files", () => {
    assert.match(textEdit({ command: "view", path: "nope.txt" }, cwd), /File not found/);
    assert.ok(!existsSync(join(cwd, "nope.txt")));
  });

  it("accepts relative path under repo root (unchanged behaviour)", () => {
    writeFileSync(join(cwd, "rel.txt"), "hello");
    const out = textEdit({ command: "view", path: "rel.txt" }, [cwd, tmpdir()]);
    assert.equal(out, "1: hello");
  });

  it("accepts absolute path under the repo root", () => {
    const abs = join(cwd, "abs.txt");
    writeFileSync(abs, "world");
    const out = textEdit({ command: "view", path: abs }, [cwd, tmpdir()]);
    assert.equal(out, "1: world");
  });

  it("accepts absolute path under os.tmpdir() — create then str_replace round-trip", () => {
    const scratchPath = join(tmpdir(), `vein-scratch-${Date.now()}.py`);
    try {
      // Create the scratch file
      const createOut = textEdit(
        { command: "create", path: scratchPath, file_text: "x = 1\n" },
        [cwd, tmpdir()]
      );
      assert.match(createOut, /Successfully created/);
      // Edit it (the original bug: this used to fail with "escapes the working directory")
      const replaceOut = textEdit(
        { command: "str_replace", path: scratchPath, old_str: "x = 1", new_str: "x = 42" },
        [cwd, tmpdir()]
      );
      assert.match(replaceOut, /Successfully replaced/);
      assert.equal(readFileSync(scratchPath, "utf-8"), "x = 42\n");
    } finally {
      rmSync(scratchPath, { force: true });
    }
  });

  it("refuses absolute path outside all roots (e.g. /etc/passwd)", () => {
    const out = textEdit({ command: "view", path: "/etc/passwd" }, [cwd, tmpdir()]);
    assert.match(out, /escapes the working directory/);
  });
});

describe("secretsEnv masking", () => {
  const VALUES = ["tok-abc123def", "sk-other-secret-9"];

  it("maskSecretValues replaces every occurrence of every value", () => {
    const out = maskSecretValues("a tok-abc123def b tok-abc123def c sk-other-secret-9", VALUES);
    assert.equal(out, "a [MASKED_SECRET] b [MASKED_SECRET] c [MASKED_SECRET]");
  });

  it("maskDeep masks string leaves in nested objects/arrays, preserves shape", () => {
    const out = maskDeep(
      { a: "x tok-abc123def", n: 7, ok: true, none: null, arr: ["sk-other-secret-9", { b: "clean" }] },
      VALUES,
    ) as any;
    assert.deepEqual(out, {
      a: "x [MASKED_SECRET]",
      n: 7,
      ok: true,
      none: null,
      arr: ["[MASKED_SECRET]", { b: "clean" }],
    });
  });

  it("maskDeep with no values is identity", () => {
    const v = { a: "tok-abc123def" };
    assert.equal(maskDeep(v, []), v);
  });

  it("wrapToolsWithMask masks tool results (the echo-$KEY path)", async () => {
    const tools: Record<string, any> = {
      bash: {
        execute: async ({ command }: { command: string }) =>
          command === "echo $KEY" ? "tok-abc123def\n" : "ok",
      },
      // provider-executed tool (no execute) must be skipped, not crash
      web_search: {},
    };
    wrapToolsWithMask(tools, VALUES);
    assert.equal(await tools.bash.execute({ command: "echo $KEY" }), "[MASKED_SECRET]\n");
    assert.equal(await tools.bash.execute({ command: "other" }), "ok");
  });

  it("run() fails loudly when secretsEnv is set but no secrets capability exists", async () => {
    const cfg = (agent.input as any).parse({
      cwd: tmpdir(),
      system: "s",
      prompt: "p",
      secretsEnv: ["SOME_KEY"],
    });
    await assert.rejects(
      () => agent.run(cfg, { runId: "r", path: "p", scope: {}, input: undefined, emit: async () => {}, services: {}, registry: {} } as any),
      /secretsEnv requires the secrets capability/,
    );
  });

  it("secretsEnv defaults to []", () => {
    const cfg = (agent.input as any).parse({ cwd: "/tmp/x", system: "s", prompt: "p" });
    assert.deepEqual(cfg.secretsEnv, []);
  });
});

describe("classifyFinalAnswerStop (premature text-only stop vs exhausted budget)", () => {
  it("done when final_answer was called, regardless of budget", () => {
    assert.equal(classifyFinalAnswerStop(true, 3, 40), "done");
    assert.equal(classifyFinalAnswerStop(true, 40, 40), "done");
  });

  it("nudge when the loop stopped tool-lessly with budget remaining", () => {
    // The live incident: a mid-task narration ended a 32-step session with an
    // 80-step budget — the loop must be resumed, not force-answered.
    assert.equal(classifyFinalAnswerStop(false, 32, 80), "nudge");
    // A single pure-text first turn is also premature.
    assert.equal(classifyFinalAnswerStop(false, 1, 40), "nudge");
  });

  it("exhausted at (or beyond) the step cap — only a no-tools forced turn is left", () => {
    assert.equal(classifyFinalAnswerStop(false, 40, 40), "exhausted");
    assert.equal(classifyFinalAnswerStop(false, 41, 40), "exhausted");
  });
});

describe("isTransientStreamError (resume a severed stream, not a real failure)", () => {
  it("treats undici's bare `terminated` as transient", () => {
    // The live incident: a 34-tool-call case-law step died ~12 minutes in when
    // its streaming response body dropped. undici raises exactly this.
    assert.equal(isTransientStreamError(new TypeError("terminated")), true);
  });

  it("matches connection faults by message or errno code", () => {
    for (const e of [
      new Error("fetch failed"),
      new Error("socket hang up"),
      new Error("Premature close"),
      new Error("other side closed"),
      Object.assign(new Error("read"), { code: "ECONNRESET" }),
      Object.assign(new Error("x"), { code: "UND_ERR_BODY_TIMEOUT" }),
      Object.assign(new Error("x"), { code: "UND_ERR_HEADERS_TIMEOUT" }),
    ]) {
      assert.equal(isTransientStreamError(e), true, `expected transient: ${e.message}`);
    }
  });

  it("unwraps a nested cause (the SDK wraps the socket error)", () => {
    const wrapped = new Error("API call failed", { cause: new TypeError("terminated") });
    assert.equal(isTransientStreamError(wrapped), true);
  });

  it("does NOT resume deterministic API failures", () => {
    for (const e of [
      new Error("401 Unauthorized: invalid x-api-key"),
      new Error("400 Bad Request: messages.0 invalid"),
      new Error("No object generated: could not parse the response."),
      new Error("model not found"),
    ]) {
      assert.equal(isTransientStreamError(e), false, `expected fatal: ${e.message}`);
    }
  });

  it("does NOT resume an abort — run control must win over recovery", () => {
    // A paused or cancelled run surfaces as an abort; resuming one would
    // defeat the cooperative pause/cancel boundary.
    assert.equal(isTransientStreamError(Object.assign(new Error("x"), { name: "AbortError" })), false);
    assert.equal(isTransientStreamError(new Error("The operation was aborted")), false);
    // Even when a transient-looking cause is wrapped underneath it.
    assert.equal(
      isTransientStreamError(Object.assign(new Error("aborted"), { cause: new TypeError("terminated") })),
      false,
    );
  });

  it("never resumes a cancelled run, by identity not by wording", () => {
    // Run control outranks recovery. Matched on the CancelledError marker so a
    // reworded cancel can't start looking transient.
    assert.equal(isTransientStreamError(Object.assign(new Error("stopped"), { isVeinCancelled: true })), false);
    assert.equal(isTransientStreamError(Object.assign(new Error("stopped"), { name: "CancelledError" })), false);
    // Even wrapping a genuinely transient cause must not make a cancel resumable.
    assert.equal(
      isTransientStreamError(
        Object.assign(new Error("stopped"), { isVeinCancelled: true, cause: new TypeError("terminated") }),
      ),
      false,
    );
  });

  it("terminates on a self-referential cause chain", () => {
    const a: any = new Error("weird");
    a.cause = a;
    assert.equal(isTransientStreamError(a), false);
  });

  it("is safe on null/undefined/non-errors", () => {
    assert.equal(isTransientStreamError(undefined), false);
    assert.equal(isTransientStreamError(null), false);
    assert.equal(isTransientStreamError("terminated"), true);
  });
});

// These stay OFFLINE in the sense that matters — nothing leaves the machine.
// They drive the real generation loop against a local server that speaks the
// Anthropic SSE wire format, because the failure being guarded is a TRANSPORT
// fault: only a genuinely severed socket reproduces `TypeError: terminated`.
describe("mid-stream socket death is resumed, not lost", () => {
  const sse = (o: any) => `event: ${o.type}\ndata: ${JSON.stringify(o)}\n\n`;
  const msgStart = () =>
    sse({
      type: "message_start",
      message: {
        id: "msg_1", type: "message", role: "assistant", model: "claude-sonnet-4-5",
        content: [], stop_reason: null, stop_sequence: null,
        usage: { input_tokens: 100, output_tokens: 1 },
      },
    });
  const toolUse = (id: string, name: string, input: unknown) =>
    sse({ type: "content_block_start", index: 0, content_block: { type: "tool_use", id, name, input: {} } }) +
    sse({ type: "content_block_delta", index: 0, delta: { type: "input_json_delta", partial_json: JSON.stringify(input) } }) +
    sse({ type: "content_block_stop", index: 0 }) +
    sse({ type: "message_delta", delta: { stop_reason: "tool_use", stop_sequence: null }, usage: { output_tokens: 20 } }) +
    sse({ type: "message_stop" });

  type Server = { port: number; bodies: string[]; calls: () => number; close: () => void };
  async function serve(handler: (call: number, res: http.ServerResponse) => void): Promise<Server> {
    const bodies: string[] = [];
    let call = 0;
    const server = http.createServer((req, res) => {
      let raw = "";
      req.on("data", (c) => (raw += c));
      req.on("end", () => { bodies.push(raw); handler(++call, res); });
    });
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    return {
      port: (server.address() as any).port,
      bodies, calls: () => call,
      close: () => server.close(),
    };
  }
  const severMidStream = (res: http.ServerResponse) => {
    res.writeHead(200, { "content-type": "text/event-stream" });
    res.write(msgStart());
    res.write(sse({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }));
    res.write(sse({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "partial" } }));
    setTimeout(() => res.socket?.destroy(), 5);
  };

  let cwd = "";
  let saved: Record<string, string | undefined> = {};
  const ENV = ["ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "VEIN_LLM_PROVIDER", "AI_SDK_LOG_WARNINGS"];
  beforeEach(() => {
    cwd = mkdtempSync(join(tmpdir(), "vein-stream-"));
    saved = Object.fromEntries(ENV.map((k) => [k, process.env[k]]));
    process.env["ANTHROPIC_API_KEY"] = "test-key";
    process.env["AI_SDK_LOG_WARNINGS"] = "false";
  });
  afterEach(() => {
    rmSync(cwd, { recursive: true, force: true });
    for (const k of ENV) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k]!;
    }
  });

  const run = (maxSteps = 10) =>
    agent.run(
      (agent.input as any).parse({
        cwd, system: "sys", prompt: "do the research",
        model: "claude-sonnet-4-5", maxSteps,
        finalAnswer: "Report what you found.", toolFilter: ["bash"],
      }),
      { runId: "r", path: "p", scope: {}, input: undefined, emit: async () => {}, services: {}, registry: {} } as any,
    ) as Promise<any>;

  it("resumes a severed stream and keeps the work done before it died", async () => {
    const s = await serve((call, res) => {
      if (call === 2) return severMidStream(res);
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.write(msgStart());
      res.write(
        call === 1
          ? toolUse("toolu_1", "bash", { command: "echo 'finding one' > research.md" })
          : toolUse("toolu_2", "final_answer", { answer: "resumed and finished" }),
      );
      res.end();
    });
    process.env["ANTHROPIC_BASE_URL"] = `http://127.0.0.1:${s.port}`;
    try {
      const out = await run();
      assert.equal(out.result, "resumed and finished");
      // The pre-error tool call's real side effect survived the socket death.
      assert.equal(readFileSync(join(cwd, "research.md"), "utf8").trim(), "finding one");
      // Banked + resumed steps are both counted, and usage is summed across
      // attempts (two message_starts at 100 input tokens each).
      assert.equal(out.steps, 2);
      assert.equal(out.usage.inputTokens, 200);
      // The resume replayed the banked conversation, the original task, and
      // told the model what actually happened.
      const resume = s.bodies[2] ?? "";
      assert.ok(resume.includes("toolu_1"), "resume must replay the banked tool call");
      assert.ok(resume.includes("do the research"), "resume must restate the task");
      assert.ok(resume.includes("interrupted mid-stream"), "resume must carry the nudge");
    } finally {
      s.close();
    }
  });

  it("gives up after a bounded number of resumes when the socket keeps dying", async () => {
    const s = await serve((call, res) => {
      if (call === 1) {
        res.writeHead(200, { "content-type": "text/event-stream" });
        res.write(msgStart());
        res.write(toolUse("toolu_1", "bash", { command: "echo hi > research.md" }));
        res.end();
        return;
      }
      severMidStream(res);
    });
    process.env["ANTHROPIC_BASE_URL"] = `http://127.0.0.1:${s.port}`;
    try {
      await assert.rejects(run(), /terminated/);
      // 1 good call + the first sever + MAX_STREAM_ERROR_CONTINUATIONS resumes.
      assert.equal(s.calls(), 7);
    } finally {
      s.close();
    }
  });

  it("does not burn resumes on a deterministic API failure", async () => {
    const s = await serve((_call, res) => {
      res.writeHead(401, { "content-type": "application/json" });
      res.end(JSON.stringify({ type: "error", error: { type: "authentication_error", message: "invalid x-api-key" } }));
    });
    process.env["ANTHROPIC_BASE_URL"] = `http://127.0.0.1:${s.port}`;
    try {
      await assert.rejects(run());
      assert.equal(s.calls(), 1, "a 401 must fail on the first call, not retry");
    } finally {
      s.close();
    }
  });
});
