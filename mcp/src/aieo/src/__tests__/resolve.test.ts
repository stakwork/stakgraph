// resolve.ts: keyless name resolution (the grammar table), the OpenRouter
// no-prefix trap, key precedence, and the one-call resolver. No network —
// getModel only constructs SDK clients.
export {};

// Env must be clean BEFORE provider.ts loads (LLM_GATEWAY_URL is read at
// module load) and stays clean so inference / key lookup are deterministic.
for (const k of [
  "LLM_GATEWAY_URL",
  "LLM_PROVIDER",
  "MAX_OUTPUT_TOKENS",
  "ANTHROPIC_API_KEY",
  "ANTHROPIC_BASE_URL",
  "ANTHROPIC_WORKSPACE_ID",
  "GOOGLE_API_KEY",
  "OPENAI_API_KEY",
  "OPENROUTER_API_KEY",
  "XAI_API_KEY",
  "MAPLE_API_KEY",
  "MAPLE_BASE_URL",
]) {
  delete process.env[k];
}

const {
  listModels,
  parseModelName,
  canonicalModelName,
  maxOutputTokensFor,
  resolveModel,
} = await import("../resolve.js");
const {
  API_KEY_ENV,
  KEY_HEADER_ENV,
  PROVIDERS,
  getModel,
  getModelDetails,
  getProviderOptions,
  computeSessionCost,
  mapleReasoningEffort,
} = await import("../provider.js");
const { generateText } = await import("ai");
const { createServer } = await import("node:http");

let passed = 0;
let failed = 0;

function eq(actual: unknown, expected: unknown, what: string) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a !== e) throw new Error(`${what}: expected ${e}, got ${a}`);
}

async function throwsWith(fn: () => unknown, needle: string, what: string) {
  try {
    await fn();
  } catch (err: any) {
    if (!String(err?.message).includes(needle)) {
      throw new Error(`${what}: error should mention "${needle}", got "${err?.message}"`);
    }
    return;
  }
  throw new Error(`${what}: expected a throw`);
}

async function test(label: string, fn: () => unknown | Promise<unknown>) {
  try {
    await fn();
    console.log(`✅ PASS: ${label}`);
    passed++;
  } catch (err: any) {
    console.error(`❌ FAIL: ${label}`);
    console.error(`   ${err.message}`);
    failed++;
  }
}

// ── listModels ──────────────────────────────────────────────────────────

await test("listModels: one entry per alias, PROVIDERS order, defaults flagged", () => {
  const all = listModels();
  eq(all.map((m) => m.alias), ["sonnet", "opus", "haiku", "gemini", "gpt", "kimi", "glm", "grok", "maple"], "aliases");
  eq(all.filter((m) => m.default).map((m) => m.provider), PROVIDERS, "exactly one default per provider");
  const sonnet = all.find((m) => m.alias === "sonnet")!;
  eq([sonnet.provider, sonnet.modelId, sonnet.default], ["anthropic", "claude-sonnet-5-5", true], "sonnet");
  eq(all.find((m) => m.alias === "opus")!.default, false, "opus is not the default");
  eq(all.find((m) => m.alias === "kimi")!.modelId, "moonshotai/kimi-k3", "kimi keeps its org/ id");
  eq(all.find((m) => m.alias === "opus")!.modelId, "claude-opus-5-5", "opus");
  eq(all.find((m) => m.alias === "grok")!.modelId, "grok-4.7", "grok");
  eq(all.find((m) => m.alias === "gpt")!.modelId, "gpt-5.6-luna", "gpt");
  eq(all.find((m) => m.alias === "gemini")!.modelId, "gemini-3.8-flash", "gemini");
  eq(all.find((m) => m.alias === "glm")!.modelId, "~z-ai/glm-flash-latest", "glm");
  eq(all.find((m) => m.alias === "glm")!.default, false, "kimi stays the openrouter default");
  eq(all.find((m) => m.alias === "maple")!.modelId, "gemma4-31b", "maple");
});

await test("API_KEY_ENV covers every provider", () => {
  eq(Object.keys(API_KEY_ENV).sort(), [...PROVIDERS].sort(), "keys");
  eq(API_KEY_ENV.openrouter, "OPENROUTER_API_KEY", "openrouter");
});

// ── parseModelName (pure syntax) ────────────────────────────────────────

const parseCases: [string | undefined, ReturnType<typeof parseModelName>][] = [
  [undefined, {}],
  ["", {}],
  ["sonnet", { modelId: "sonnet" }],
  ["claude-sonnet-5-5", { modelId: "claude-sonnet-5-5" }],
  ["anthropic/claude-sonnet-5-5", { provider: "anthropic", modelId: "claude-sonnet-5-5" }],
  ["openrouter/moonshotai/kimi-k2.6", { provider: "openrouter", modelId: "moonshotai/kimi-k2.6" }],
  ["openrouter/openrouter/auto", { provider: "openrouter", modelId: "openrouter/auto" }],
  ["openai/gpt-5", { provider: "openai", modelId: "gpt-5" }],
  ["moonshotai/kimi-k2.6", { modelId: "moonshotai/kimi-k2.6" }], // not a provider → untouched
  ["openrouter/", { provider: "openrouter" }],                     // bare prefix → default model
];
for (const [name, expected] of parseCases) {
  await test(`parseModelName(${JSON.stringify(name)})`, () => eq(parseModelName(name), expected, "parsed"));
}

// ── canonicalModelName (the grammar table, keyless) ─────────────────────

const canonCases: [string | undefined, string | undefined, [string, string, string]][] = [
  // [model, provider, [provider, modelId, name]]
  ["sonnet", undefined, ["anthropic", "claude-sonnet-5-5", "anthropic/claude-sonnet-5-5"]],
  ["claude-sonnet-5-5", undefined, ["anthropic", "claude-sonnet-5-5", "anthropic/claude-sonnet-5-5"]],
  ["anthropic/claude-sonnet-5-5", undefined, ["anthropic", "claude-sonnet-5-5", "anthropic/claude-sonnet-5-5"]],
  ["anthropic/sonnet", undefined, ["anthropic", "claude-sonnet-5-5", "anthropic/claude-sonnet-5-5"]],
  ["openrouter/moonshotai/kimi-k2.6", undefined, ["openrouter", "moonshotai/kimi-k2.6", "openrouter/moonshotai/kimi-k2.6"]],
  ["openrouter/openrouter/auto", undefined, ["openrouter", "openrouter/auto", "openrouter/openrouter/auto"]],
  ["openrouter/openai/gpt-5", undefined, ["openrouter", "openai/gpt-5", "openrouter/openai/gpt-5"]],
  ["openai/gpt-5", undefined, ["openai", "gpt-5", "openai/gpt-5"]],
  ["kimi", undefined, ["openrouter", "moonshotai/kimi-k3", "openrouter/moonshotai/kimi-k3"]],
  ["opus", undefined, ["anthropic", "claude-opus-5-5", "anthropic/claude-opus-5-5"]],
  ["grok", undefined, ["xai", "grok-4.7", "xai/grok-4.7"]],
  ["claude-opus-5", undefined, ["anthropic", "claude-opus-5", "anthropic/claude-opus-5"]],
  ["claude-opus-5-5", undefined, ["anthropic", "claude-opus-5-5", "anthropic/claude-opus-5-5"]],
  ["grok-4-fast", undefined, ["xai", "grok-4-fast", "xai/grok-4-fast"]],
  ["claude-opus-4-8", undefined, ["anthropic", "claude-opus-4-8", "anthropic/claude-opus-4-8"]], // unknown id passes through
  [undefined, undefined, ["anthropic", "claude-sonnet-5-5", "anthropic/claude-sonnet-5-5"]],       // nothing → anthropic default
  [undefined, "openai", ["openai", "gpt-5.6-luna", "openai/gpt-5.6-luna"]],                        // provider only → its default
  ["gpt", undefined, ["openai", "gpt-5.6-luna", "openai/gpt-5.6-luna"]],
  ["gpt-5.6-sol", undefined, ["openai", "gpt-5.6-sol", "openai/gpt-5.6-sol"]],                   // exact-match list
  ["gpt-5.4-mini", undefined, ["openai", "gpt-5.4-mini", "openai/gpt-5.4-mini"]],                 // vendor-prefix fallback
  ["gemini-3.1-pro-preview", undefined, ["google", "gemini-3.1-pro-preview", "google/gemini-3.1-pro-preview"]],
  ["gemini", undefined, ["google", "gemini-3.8-flash", "google/gemini-3.8-flash"]],
  ["glm", undefined, ["openrouter", "~z-ai/glm-flash-latest", "openrouter/~z-ai/glm-flash-latest"]],
  ["openrouter/~z-ai/glm-flash-latest", undefined, ["openrouter", "~z-ai/glm-flash-latest", "openrouter/~z-ai/glm-flash-latest"]],
  ["openrouter/", undefined, ["openrouter", "moonshotai/kimi-k3", "openrouter/moonshotai/kimi-k3"]],
  ["moonshotai/kimi-k2.6", "openrouter", ["openrouter", "moonshotai/kimi-k2.6", "openrouter/moonshotai/kimi-k2.6"]], // explicit provider
  ["anthropic/claude-sonnet-5-5", "openai", ["openai", "claude-sonnet-5-5", "openai/claude-sonnet-5-5"]], // explicit provider wins (mirrors getModel)
  ["maple/gemma4-31b", undefined, ["maple", "gemma4-31b", "maple/gemma4-31b"]],
  ["maple/gpt-oss-120b", undefined, ["maple", "gpt-oss-120b", "maple/gpt-oss-120b"]],
  ["maple", undefined, ["maple", "gemma4-31b", "maple/gemma4-31b"]],                              // the provider's own alias
  [undefined, "maple", ["maple", "gemma4-31b", "maple/gemma4-31b"]],
  ["kimi-k3", "maple", ["maple", "kimi-k3", "maple/kimi-k3"]],                                     // explicit provider
];
for (const [model, provider, [p, id, name]] of canonCases) {
  await test(`canonicalModelName(${JSON.stringify(model)}, ${JSON.stringify(provider)}) → ${name}`, () => {
    const ref = canonicalModelName(model, provider);
    eq([ref.provider, ref.modelId, ref.name], [p, id, name], "ref");
  });
}

await test("canonicalModelName: unknown provider throws the agent-step message", () =>
  throwsWith(() => canonicalModelName("gpt-5", "foo"), 'Unknown LLM provider: "foo"', "unknown provider"));

await test("canonicalModelName: alias of another provider throws", () =>
  throwsWith(() => canonicalModelName("sonnet", "openai"), 'not available for provider "openai"', "cross-provider alias"));

await test("canonicalModelName: the OpenRouter no-prefix trap throws with the fix", () =>
  throwsWith(
    () => canonicalModelName("moonshotai/kimi-k2.6"),
    'write "openrouter/moonshotai/kimi-k2.6"',
    "trap",
  ));

await test("canonicalModelName: trap also catches org-prefixed grok", () =>
  throwsWith(() => canonicalModelName("x-ai/grok-4"), '"x-ai" is not a provider', "trap"));

await test("canonicalModelName: LLM_PROVIDER outranks the vendor-prefix fallback for bare ids", () => {
  process.env.LLM_PROVIDER = "openrouter";
  try {
    eq(canonicalModelName("gpt-5.4-mini").provider, "openrouter", "bare gpt id under a gateway provider");
  } finally {
    delete process.env.LLM_PROVIDER;
  }
});

await test("canonicalModelName: LLM_PROVIDER env disarms the trap (it names the provider)", () => {
  process.env.LLM_PROVIDER = "openrouter";
  try {
    const ref = canonicalModelName("moonshotai/kimi-k2.6");
    eq(ref.name, "openrouter/moonshotai/kimi-k2.6", "ref");
  } finally {
    delete process.env.LLM_PROVIDER;
  }
});

await test("canonicalModelName: a bare Maple id never picks Maple — it says how to", () =>
  throwsWith(() => canonicalModelName("gemma4-31b"), 'On Maple it is "maple/gemma4-31b"', "bare maple id"));

await test("canonicalModelName: LLM_PROVIDER=maple makes a bare Maple id Maple's", () => {
  process.env.LLM_PROVIDER = "maple";
  try {
    eq(canonicalModelName("gpt-oss-120b").name, "maple/gpt-oss-120b", "ref");
  } finally {
    delete process.env.LLM_PROVIDER;
  }
});

// ── maxOutputTokensFor ───────────────────────────────────────────────────

await test("maxOutputTokensFor: anthropic 128k, others 64k, env override wins", () => {
  eq(maxOutputTokensFor("anthropic"), 128_000, "anthropic");
  eq(maxOutputTokensFor("openai"), 64_000, "openai");
  eq(maxOutputTokensFor(undefined), 64_000, "undefined");
  process.env.MAX_OUTPUT_TOKENS = "9000";
  try {
    eq(maxOutputTokensFor("anthropic"), 9000, "env");
  } finally {
    delete process.env.MAX_OUTPUT_TOKENS;
  }
});

await test("maxOutputTokensFor: an Anthropic model's own limit, less an enabled thinking budget", () => {
  eq(maxOutputTokensFor("anthropic", "claude-haiku-4-5"), 64_000, "haiku-4-5");
  eq(maxOutputTokensFor("anthropic", "claude-haiku-4-5-20251001"), 64_000, "dated id");
  eq(maxOutputTokensFor("anthropic", "claude-sonnet-4-5"), 64_000, "sonnet-4-5");
  eq(maxOutputTokensFor("anthropic", "claude-sonnet-4-6"), 128_000, "sonnet-4-6");
  eq(maxOutputTokensFor("anthropic", "claude-sonnet-5-5"), 128_000, "sonnet-5-5");
  const enabled = { anthropic: { thinking: { type: "enabled", budgetTokens: 24_000 } } };
  const disabled = { anthropic: { thinking: { type: "disabled" } } };
  const adaptive = { anthropic: { thinking: { type: "adaptive" } } };
  eq(maxOutputTokensFor("anthropic", "claude-haiku-4-5", enabled), 40_000, "haiku-4-5 + 24k budget");
  eq(maxOutputTokensFor("anthropic", "claude-haiku-4-5", disabled), 64_000, "haiku-4-5, thinking off");
  eq(maxOutputTokensFor("anthropic", "claude-sonnet-5-5", adaptive), 128_000, "adaptive");
  process.env.MAX_OUTPUT_TOKENS = "200000";
  try {
    eq(maxOutputTokensFor("anthropic", "claude-haiku-4-5"), 64_000, "env within the limit");
    eq(maxOutputTokensFor("anthropic", "claude-haiku-4-5", enabled), 40_000, "env within the limit, with thinking");
    eq(maxOutputTokensFor("openai"), 200_000, "env elsewhere as given");
  } finally {
    delete process.env.MAX_OUTPUT_TOKENS;
  }
});

await test("maxOutputTokensFor: Maple's per model, under a tenth of the window", () => {
  eq(maxOutputTokensFor("maple", "gemma4-31b"), 24_576, "gemma4-31b");
  eq(maxOutputTokensFor("maple", "gpt-oss-120b"), 12_288, "gpt-oss-120b");
  eq(maxOutputTokensFor("maple", "deepseek-v4-1-flash"), 64_000, "deepseek-v4-1-flash");
  eq(maxOutputTokensFor("maple", "something-new"), 12_288, "unlisted");
});

// ── resolveModel ─────────────────────────────────────────────────────────

await test("resolveModel: alias → concrete model, context limit, output cap", async () => {
  const r = await resolveModel({ model: "sonnet", apiKey: "k" });
  eq([r.provider, r.modelId, r.name], ["anthropic", "claude-sonnet-5-5", "anthropic/claude-sonnet-5-5"], "ref");
  eq((r.model as { modelId?: string }).modelId, "claude-sonnet-5-5", "SDK model id");
  eq(r.contextLimit, 1_000_000, "contextLimit");
  eq(r.maxOutputTokens, 128_000, "maxOutputTokens");
  eq(r.apiKey, "k", "apiKey");
});

await test("resolveModel: context limits for the alias targets", async () => {
  const limit = async (model: string) => (await resolveModel({ model, apiKey: "k" })).contextLimit;
  eq(await limit("opus"), 1_000_000, "claude-opus-5-5");
  eq(await limit("gpt"), 1_050_000, "gpt-5.6-luna");
  eq(await limit("grok"), 500_000, "grok-4.7");
  eq(await limit("kimi"), 1_048_576, "moonshotai/kimi-k3");
  eq(await limit("gemini"), 1_048_576, "gemini-3.8-flash");
  eq(await limit("glm"), 1_310_720, "~z-ai/glm-flash-latest");
  eq(await limit("openrouter/anthropic/claude-opus-5"), 1_000_000, "opus via OpenRouter");
  eq(await limit("openrouter/x-ai/grok-4.7"), 500_000, "grok via OpenRouter");
});

await test("resolveModel: the ~rolling-alias id reaches the SDK intact", async () => {
  const r = await resolveModel({ model: "glm", apiKey: "k" });
  eq((r.model as { modelId?: string }).modelId, "~z-ai/glm-flash-latest", "SDK model id");
});

await test("resolveModel: OpenRouter own-namespace id survives the single prefix strip", async () => {
  const r = await resolveModel({ model: "openrouter/openrouter/auto", apiKey: "k" });
  eq((r.model as { modelId?: string }).modelId, "openrouter/auto", "SDK model id");
});

await test("resolveModel: OpenRouter org/model id keeps its org and finds its context limit", async () => {
  const r = await resolveModel({ model: "openrouter/moonshotai/kimi-k2.6", apiKey: "k" });
  eq((r.model as { modelId?: string }).modelId, "moonshotai/kimi-k2.6", "SDK model id");
  eq(r.contextLimit, 262_144, "contextLimit");
  eq(r.maxOutputTokens, 64_000, "maxOutputTokens");
});

await test("resolveModel: getSecret is asked by env-var NAME and beats process.env", async () => {
  const asked: string[] = [];
  process.env.OPENAI_API_KEY = "from-env";
  try {
    const r = await resolveModel({
      model: "gpt",
      getSecret: async (name) => {
        asked.push(name);
        return "from-store";
      },
    });
    eq(asked, ["OPENAI_API_KEY"], "asked");
    eq(r.apiKey, "from-store", "store wins over env");
  } finally {
    delete process.env.OPENAI_API_KEY;
  }
});

await test("resolveModel: explicit apiKey skips getSecret; blank apiKey counts as absent", async () => {
  let calls = 0;
  const secret = async () => {
    calls++;
    return "from-store";
  };
  const a = await resolveModel({ model: "gpt", apiKey: "explicit", getSecret: secret });
  eq([a.apiKey, calls], ["explicit", 0], "explicit key, getSecret untouched");
  const b = await resolveModel({ model: "gpt", apiKey: "   ", getSecret: secret });
  eq([b.apiKey, calls], ["from-store", 1], "blank key falls through to getSecret");
});

await test("resolveModel: getSecret miss falls back to process.env", async () => {
  process.env.XAI_API_KEY = "env-xai";
  try {
    const r = await resolveModel({ model: "grok", getSecret: async () => undefined });
    eq([r.provider, r.apiKey], ["xai", "env-xai"], "env fallback");
  } finally {
    delete process.env.XAI_API_KEY;
  }
});

await test("resolveModel: no key anywhere → error names the env var", () =>
  throwsWith(() => resolveModel({ model: "gemini" }), "set GOOGLE_API_KEY", "missing key"));

await test("resolveModel: the trap fires before any key lookup", async () => {
  let calls = 0;
  await throwsWith(
    () => resolveModel({ model: "moonshotai/kimi-k2.6", getSecret: async () => (calls++, "k") }),
    'write "openrouter/moonshotai/kimi-k2.6"',
    "trap",
  );
  eq(calls, 0, "getSecret never called");
});

// ── maple ────────────────────────────────────────────────────────────────

await test("resolveModel: maple is a chat-completions model with its window and cap", async () => {
  const r = await resolveModel({ model: "maple/gemma4-31b", apiKey: "k" });
  const m = r.model as { modelId?: string; provider?: string };
  eq([m.provider, m.modelId], ["openai.chat", "gemma4-31b"], "SDK model");
  eq([r.contextLimit, r.maxOutputTokens], [262_144, 24_576], "limits");
  eq((await resolveModel({ model: "maple/gpt-oss-120b", apiKey: "k" })).contextLimit, 131_072, "gpt-oss");
});

await test("mapleReasoningEffort: the nearest effort the model takes, ties to more thinking", () => {
  const cases: [string, string, string][] = [
    ["gemma4-31b", "none", "none"],
    ["gemma4-31b", "medium", "medium"],
    ["gpt-oss-120b", "none", "low"],
    ["gpt-oss-120b", "minimal", "low"],
    ["gpt-oss-120b", "medium", "medium"],
    ["gpt-oss-120b", "xhigh", "high"],
    ["gpt-oss-120b", "max", "high"],
    ["glm-5-3", "none", "low"],
    ["glm-5-3", "medium", "high"],
    ["glm-5-3-flash", "xhigh", "max"],
    ["kimi-k3", "none", "none"],
    ["kimi-k3", "minimal", "low"],
    ["kimi-k3", "medium", "high"],
    ["deepseek-v4-1-flash", "medium", "high"],
    ["deepseek-v4-1-flash", "xhigh", "xhigh"],
    ["llama3-3-70b", "high", "high"],
    ["unlisted", "medium", "medium"],
    ["gpt-oss-120b", "provider-default", "provider-default"],
  ];
  for (const [model, asked, sent] of cases) eq(mapleReasoningEffort(model, asked), sent, `${model} ${asked}`);
});

await test("getProviderOptions(maple): fast is the least thinking the model allows, thinking is high", () => {
  eq(getProviderOptions("maple", "fast", "maple/gemma4-31b"), { openai: { reasoningEffort: "none" } }, "gemma fast");
  eq(getProviderOptions("maple", "fast", "gpt-oss-120b"), { openai: { reasoningEffort: "low" } }, "gpt-oss fast");
  eq(getProviderOptions("maple", "thinking", "maple"), { openai: { reasoningEffort: "high" } }, "default thinking");
  eq(getProviderOptions("maple", undefined, "maple/kimi-k3"), { openai: {} }, "the model's default");
});

await test("computeSessionCost(maple): Maple's own rates, a cached prompt token as input", () => {
  const usage = { input: 600_000, cache_read: 400_000, cache_write: 0, output: 1_000_000 };
  eq(computeSessionCost("maple", usage, "gemma4-31b").toFixed(4), "2.8000", "gemma4-31b: 1M in at 0.80, 1M out at 2.00");
  eq(computeSessionCost("maple", usage, "gpt-oss-120b").toFixed(4), "1.5000", "gpt-oss-120b");
  eq(computeSessionCost("maple", usage, "unlisted").toFixed(4), "33.0000", "unlisted: the dearest listed rate");
});

await test("maple on the wire: MAPLE_BASE_URL via getSecret, /chat/completions, the effort the model takes", async () => {
  const seen: { path?: string; auth?: string; effort?: unknown; model?: unknown }[] = [];
  const server = createServer((req, res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      const json = JSON.parse(body);
      seen.push({ path: req.url, auth: req.headers.authorization, effort: json.reasoning_effort, model: json.model });
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({
        id: "x", object: "chat.completion", created: 0, model: json.model,
        choices: [{ index: 0, message: { role: "assistant", content: "391" }, finish_reason: "stop" }],
        usage: { prompt_tokens: 100, completion_tokens: 10, prompt_tokens_details: { cached_tokens: 40 } },
      }));
    });
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const root = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  const store = (entries: Record<string, string>) => async (name: string) => entries[name];
  try {
    const asked: string[] = [];
    const gemma = await resolveModel({
      model: "maple/gemma4-31b",
      getSecret: async (name) => (asked.push(name), ({ MAPLE_API_KEY: "mk", MAPLE_BASE_URL: `${root}/v1` } as Record<string, string>)[name]),
    });
    eq(asked, ["MAPLE_API_KEY", "MAPLE_BASE_URL"], "asked by name");
    const r = await generateText({ model: gemma.model, prompt: "hi", reasoning: "none", maxRetries: 0 } as any);
    eq(r.text, "391", "text");
    eq([r.usage.inputTokens, r.usage.outputTokens, r.usage.inputTokenDetails?.cacheReadTokens], [100, 10, 40], "usage");
    // gpt-oss takes low | medium | high: none is sent as low
    const oss = await resolveModel({ model: "maple/gpt-oss-120b", getSecret: store({ MAPLE_API_KEY: "mk", MAPLE_BASE_URL: `${root}/v1` }) });
    await generateText({ model: oss.model, prompt: "hi", reasoning: "none", maxRetries: 0 } as any);
    // providerOptions too, and a caller's baseUrl (a bare root gets /v1) wins over MAPLE_BASE_URL
    const glm = await resolveModel({ model: "maple/glm-5-3", baseUrl: root, getSecret: store({ MAPLE_API_KEY: "mk", MAPLE_BASE_URL: "http://127.0.0.1:1/v1" }) });
    await generateText({ model: glm.model, prompt: "hi", providerOptions: getProviderOptions("maple", undefined, "glm-5-3") as any, reasoning: "medium", maxRetries: 0 } as any);
    // the env-only path (getModel)
    process.env.MAPLE_API_KEY = "env-key";
    process.env.MAPLE_BASE_URL = `${root}/v1`;
    await generateText({ model: getModel("maple", { modelName: "maple/kimi-k3" }), prompt: "hi", maxRetries: 0 });
    eq(seen, [
      { path: "/v1/chat/completions", auth: "Bearer mk", effort: "none", model: "gemma4-31b" },
      { path: "/v1/chat/completions", auth: "Bearer mk", effort: "low", model: "gpt-oss-120b" },
      { path: "/v1/chat/completions", auth: "Bearer mk", effort: "high", model: "glm-5-3" },
      { path: "/v1/chat/completions", auth: "Bearer env-key", model: "kimi-k3" },
    ], "requests");
  } finally {
    server.close();
    delete process.env.MAPLE_API_KEY;
    delete process.env.MAPLE_BASE_URL;
  }
});

await test("maple: no MAPLE_BASE_URL anywhere → the proxy on localhost:8080", async () => {
  const urls: string[] = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (input: any) => {
    urls.push(typeof input === "string" ? input : input.url);
    throw new Error("recorded");
  }) as typeof fetch;
  try {
    const r = await resolveModel({ model: "maple/gemma4-31b", apiKey: "k", getSecret: async () => undefined });
    await throwsWith(() => generateText({ model: r.model, prompt: "hi", maxRetries: 0 }), "recorded", "request");
  } finally {
    globalThis.fetch = realFetch;
  }
  eq(urls, ["http://localhost:8080/v1/chat/completions"], "default url");
});

// ── KEY_HEADER_ENV: an organization-wide Anthropic key names its workspace ──

await test("KEY_HEADER_ENV: Anthropic's workspace is listed beside its key", () => {
  eq(KEY_HEADER_ENV.anthropic, { ANTHROPIC_WORKSPACE_ID: "anthropic-workspace-id" }, "anthropic");
  eq(KEY_HEADER_ENV.openai, undefined, "openai has none");
});

await test("anthropic-workspace-id: sent with a looked-up key, never with a caller's own", async () => {
  const seen: (string | undefined)[] = [];
  const server = createServer((req, res) => {
    seen.push(req.headers["anthropic-workspace-id"] as string | undefined);
    res.writeHead(400, { "content-type": "application/json" });
    res.end(JSON.stringify({ type: "error", error: { type: "invalid_request_error", message: "recorded" } }));
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  process.env.ANTHROPIC_BASE_URL = `http://127.0.0.1:${(server.address() as { port: number }).port}/v1`;
  const send = async (model: any) => {
    await throwsWith(() => generateText({ model, prompt: "hi", maxRetries: 0 }), "recorded", "request");
  };
  const store = (entries: Record<string, string>) => async (name: string) => entries[name];
  try {
    // resolveModel, key and workspace from getSecret (trimmed)
    await send((await resolveModel({
      model: "sonnet",
      getSecret: store({ ANTHROPIC_API_KEY: "k", ANTHROPIC_WORKSPACE_ID: " wrkspc_store " }),
    })).model);
    // a workspace key: none set anywhere
    await send((await resolveModel({ model: "sonnet", getSecret: store({ ANTHROPIC_API_KEY: "k" }) })).model);
    // getSecret misses → env, for the workspace as for the key
    process.env.ANTHROPIC_API_KEY = "k";
    process.env.ANTHROPIC_WORKSPACE_ID = "wrkspc_env";
    await send((await resolveModel({ model: "sonnet", getSecret: store({}) })).model);
    // caller headers win
    await send((await resolveModel({ model: "sonnet", headers: { "anthropic-workspace-id": "wrkspc_caller" } })).model);
    // a caller's own key brings its own workspace
    await send((await resolveModel({ model: "sonnet", apiKey: "vk-user" })).model);
    // the env-only path: getModel / getModelDetails
    await send(getModel("anthropic"));
    await send(getModelDetails("sonnet").model);
    await send(getModel("anthropic", { apiKey: "vk-user" }));
    eq(
      seen,
      ["wrkspc_store", undefined, "wrkspc_env", "wrkspc_caller", undefined, "wrkspc_env", "wrkspc_env", undefined],
      "workspace header per request",
    );
  } finally {
    server.close();
    delete process.env.ANTHROPIC_BASE_URL;
    delete process.env.ANTHROPIC_API_KEY;
    delete process.env.ANTHROPIC_WORKSPACE_ID;
  }
});

console.log(`\nResults: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
