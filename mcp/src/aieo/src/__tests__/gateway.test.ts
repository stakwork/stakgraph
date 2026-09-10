// Gateway model-id routing. When getModel is handed a gateway baseUrl, the
// providers that ride Bifrost's OpenAI-compat route must ship a
// "<provider>/"-prefixed model id: Bifrost routes by a known prefix and
// otherwise defaults to OpenAI on /openai/v1. Direct calls keep the bare id.
// No network — getModel only constructs SDK clients.
export {};

delete process.env.LLM_GATEWAY_URL;

const { getModel } = await import("../provider.js");

type Case = {
  label: string;
  provider: Parameters<typeof getModel>[0];
  modelName: string;
  baseUrl?: string;
  expectModelId: string;
};

const GATEWAY = "https://swarm.example:8181";

const cases: Case[] = [
  {
    label: "openrouter via gateway → openrouter/ prefix",
    provider: "openrouter",
    modelName: "openrouter/moonshotai/kimi-k2-0905",
    baseUrl: GATEWAY,
    expectModelId: "openrouter/moonshotai/kimi-k2-0905",
  },
  {
    label: "openrouter via already-suffixed gateway url → openrouter/ prefix",
    provider: "openrouter",
    modelName: "openrouter/moonshotai/kimi-k2-0905",
    baseUrl: `${GATEWAY}/openai/v1`,
    expectModelId: "openrouter/moonshotai/kimi-k2-0905",
  },
  {
    label: "openrouter direct → bare id",
    provider: "openrouter",
    modelName: "openrouter/moonshotai/kimi-k2-0905",
    expectModelId: "moonshotai/kimi-k2-0905",
  },
  {
    label: "openrouter own-namespace id via gateway → one prefix per hop",
    provider: "openrouter",
    modelName: "openrouter/openrouter/auto",
    baseUrl: GATEWAY,
    expectModelId: "openrouter/openrouter/auto",
  },
  {
    label: "xai via gateway → xai/ prefix",
    provider: "xai",
    modelName: "xai/grok-4.3",
    baseUrl: GATEWAY,
    expectModelId: "xai/grok-4.3",
  },
  {
    label: "xai direct → bare id",
    provider: "xai",
    modelName: "xai/grok-4.3",
    expectModelId: "grok-4.3",
  },
  {
    label: "google via gateway → gemini/ prefix",
    provider: "google",
    modelName: "google/gemini-2.5-flash",
    baseUrl: GATEWAY,
    expectModelId: "gemini/gemini-2.5-flash",
  },
  {
    label: "anthropic via gateway → bare id",
    provider: "anthropic",
    modelName: "anthropic/claude-sonnet-4-6",
    baseUrl: GATEWAY,
    expectModelId: "claude-sonnet-4-6",
  },
  {
    label: "openai via gateway → bare id",
    provider: "openai",
    modelName: "openai/gpt-4o-mini",
    baseUrl: GATEWAY,
    expectModelId: "gpt-4o-mini",
  },
];

let passed = 0;
let failed = 0;

for (const tc of cases) {
  try {
    const model = getModel(tc.provider, {
      modelName: tc.modelName,
      apiKey: "test-key",
      baseUrl: tc.baseUrl,
    });
    const modelId = (model as { modelId?: string }).modelId;
    if (modelId !== tc.expectModelId) {
      throw new Error(`expected modelId "${tc.expectModelId}", got "${modelId}"`);
    }
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
