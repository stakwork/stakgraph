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
  "GOOGLE_API_KEY",
  "OPENAI_API_KEY",
  "OPENROUTER_API_KEY",
  "XAI_API_KEY",
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
const { API_KEY_ENV, PROVIDERS } = await import("../provider.js");

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
  eq(all.map((m) => m.alias), ["sonnet", "opus", "haiku", "gemini", "gpt", "kimi", "grok"], "aliases");
  eq(all.filter((m) => m.default).map((m) => m.provider), PROVIDERS, "exactly one default per provider");
  const sonnet = all.find((m) => m.alias === "sonnet")!;
  eq([sonnet.provider, sonnet.modelId, sonnet.default], ["anthropic", "claude-sonnet-5", true], "sonnet");
  eq(all.find((m) => m.alias === "opus")!.default, false, "opus is not the default");
  eq(all.find((m) => m.alias === "kimi")!.modelId, "moonshotai/kimi-k3", "kimi keeps its org/ id");
  eq(all.find((m) => m.alias === "opus")!.modelId, "claude-opus-5", "opus");
  eq(all.find((m) => m.alias === "grok")!.modelId, "grok-4.6", "grok");
  eq(all.find((m) => m.alias === "gpt")!.modelId, "gpt-5.6-luna", "gpt");
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
  ["claude-sonnet-5", { modelId: "claude-sonnet-5" }],
  ["anthropic/claude-sonnet-5", { provider: "anthropic", modelId: "claude-sonnet-5" }],
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
  ["sonnet", undefined, ["anthropic", "claude-sonnet-5", "anthropic/claude-sonnet-5"]],
  ["claude-sonnet-5", undefined, ["anthropic", "claude-sonnet-5", "anthropic/claude-sonnet-5"]],
  ["anthropic/claude-sonnet-5", undefined, ["anthropic", "claude-sonnet-5", "anthropic/claude-sonnet-5"]],
  ["anthropic/sonnet", undefined, ["anthropic", "claude-sonnet-5", "anthropic/claude-sonnet-5"]],
  ["openrouter/moonshotai/kimi-k2.6", undefined, ["openrouter", "moonshotai/kimi-k2.6", "openrouter/moonshotai/kimi-k2.6"]],
  ["openrouter/openrouter/auto", undefined, ["openrouter", "openrouter/auto", "openrouter/openrouter/auto"]],
  ["openrouter/openai/gpt-5", undefined, ["openrouter", "openai/gpt-5", "openrouter/openai/gpt-5"]],
  ["openai/gpt-5", undefined, ["openai", "gpt-5", "openai/gpt-5"]],
  ["kimi", undefined, ["openrouter", "moonshotai/kimi-k3", "openrouter/moonshotai/kimi-k3"]],
  ["opus", undefined, ["anthropic", "claude-opus-5", "anthropic/claude-opus-5"]],
  ["grok", undefined, ["xai", "grok-4.6", "xai/grok-4.6"]],
  ["claude-opus-5", undefined, ["anthropic", "claude-opus-5", "anthropic/claude-opus-5"]],
  ["grok-4-fast", undefined, ["xai", "grok-4-fast", "xai/grok-4-fast"]],
  ["claude-opus-4-8", undefined, ["anthropic", "claude-opus-4-8", "anthropic/claude-opus-4-8"]], // unknown id passes through
  [undefined, undefined, ["anthropic", "claude-sonnet-5", "anthropic/claude-sonnet-5"]],       // nothing → anthropic default
  [undefined, "openai", ["openai", "gpt-5.6-luna", "openai/gpt-5.6-luna"]],                        // provider only → its default
  ["gpt", undefined, ["openai", "gpt-5.6-luna", "openai/gpt-5.6-luna"]],
  ["gpt-5.6-sol", undefined, ["openai", "gpt-5.6-sol", "openai/gpt-5.6-sol"]],                   // exact-match list
  ["gpt-5.4-mini", undefined, ["openai", "gpt-5.4-mini", "openai/gpt-5.4-mini"]],                 // vendor-prefix fallback
  ["gemini-3.1-pro-preview", undefined, ["google", "gemini-3.1-pro-preview", "google/gemini-3.1-pro-preview"]],
  ["openrouter/", undefined, ["openrouter", "moonshotai/kimi-k3", "openrouter/moonshotai/kimi-k3"]],
  ["moonshotai/kimi-k2.6", "openrouter", ["openrouter", "moonshotai/kimi-k2.6", "openrouter/moonshotai/kimi-k2.6"]], // explicit provider
  ["anthropic/claude-sonnet-5", "openai", ["openai", "claude-sonnet-5", "openai/claude-sonnet-5"]], // explicit provider wins (mirrors getModel)
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

// ── resolveModel ─────────────────────────────────────────────────────────

await test("resolveModel: alias → concrete model, context limit, output cap", async () => {
  const r = await resolveModel({ model: "sonnet", apiKey: "k" });
  eq([r.provider, r.modelId, r.name], ["anthropic", "claude-sonnet-5", "anthropic/claude-sonnet-5"], "ref");
  eq((r.model as { modelId?: string }).modelId, "claude-sonnet-5", "SDK model id");
  eq(r.contextLimit, 1_000_000, "contextLimit");
  eq(r.maxOutputTokens, 128_000, "maxOutputTokens");
  eq(r.apiKey, "k", "apiKey");
});

await test("resolveModel: context limits for the alias targets", async () => {
  const limit = async (model: string) => (await resolveModel({ model, apiKey: "k" })).contextLimit;
  eq(await limit("opus"), 1_000_000, "claude-opus-5");
  eq(await limit("gpt"), 1_050_000, "gpt-5.6-luna");
  eq(await limit("grok"), 500_000, "grok-4.6");
  eq(await limit("kimi"), 1_048_576, "moonshotai/kimi-k3");
  eq(await limit("gemini"), 1_048_576, "gemini-3-pro-preview");
  eq(await limit("openrouter/anthropic/claude-opus-5"), 1_000_000, "opus via OpenRouter");
  eq(await limit("openrouter/x-ai/grok-4.6"), 500_000, "grok via OpenRouter");
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

console.log(`\nResults: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
