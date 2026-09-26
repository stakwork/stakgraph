import { getProviderOptions } from "../provider.js";

type TestCase = {
  label: string;
  provider: Parameters<typeof getProviderOptions>[0];
  thinkingSpeed: Parameters<typeof getProviderOptions>[1];
  modelName: Parameters<typeof getProviderOptions>[2];
  assert: (result: ReturnType<typeof getProviderOptions>) => void;
};

const tests: TestCase[] = [
  {
    label: 'thinkingSpeed:"thinking" + adaptive model → type:"adaptive", effort:"high"',
    provider: "anthropic",
    thinkingSpeed: "thinking",
    modelName: "claude-opus-4-7",
    assert(result) {
      const opts = (result as any).anthropic;
      if (opts.thinking.type !== "adaptive")
        throw new Error(`Expected type:"adaptive", got "${opts.thinking.type}"`);
      if (opts.thinking.display !== "summarized")
        throw new Error(`Expected display:"summarized"`);
      if (opts.effort !== "high")
        throw new Error(`Expected effort:"high", got "${opts.effort}"`);
    },
  },
  {
    label: 'thinkingSpeed:"thinking" + non-adaptive model → type:"enabled", budgetTokens:24000',
    provider: "anthropic",
    thinkingSpeed: "thinking",
    modelName: "claude-haiku-3",
    assert(result) {
      const opts = (result as any).anthropic;
      if (opts.thinking.type !== "enabled")
        throw new Error(`Expected type:"enabled", got "${opts.thinking.type}"`);
      if (opts.thinking.budgetTokens !== 24000)
        throw new Error(`Expected budgetTokens:24000, got ${opts.thinking.budgetTokens}`);
      if ("effort" in opts)
        throw new Error(`effort should not be present for non-adaptive model`);
    },
  },
  {
    label: 'thinkingSpeed:undefined + adaptive model → type:"adaptive", no effort',
    provider: "anthropic",
    thinkingSpeed: undefined,
    modelName: "claude-sonnet-5",
    assert(result) {
      const opts = (result as any).anthropic;
      if (opts.thinking.type !== "adaptive")
        throw new Error(`Expected type:"adaptive", got "${opts.thinking.type}"`);
      if (opts.thinking.display !== "summarized")
        throw new Error(`Expected display:"summarized"`);
      if ("effort" in opts)
        throw new Error(`effort should not be present when thinkingSpeed is not "thinking"`);
    },
  },
  {
    label: 'thinkingSpeed:"fast" + opus 5 → type:"disabled" (fast is no thinking wherever the API allows it)',
    provider: "anthropic",
    thinkingSpeed: "fast",
    modelName: "claude-opus-5",
    assert(result) {
      const opts = (result as any).anthropic;
      if (opts.thinking?.type !== "disabled")
        throw new Error(`Expected type:"disabled", got "${opts.thinking?.type}"`);
      if ("effort" in opts) throw new Error(`effort should not be present on the fast path`);
    },
  },
  {
    label: 'thinkingSpeed:"fast" + no modelName → type:"disabled" (callers pass the model so the reject branch can fire)',
    provider: "anthropic",
    thinkingSpeed: "fast",
    modelName: undefined,
    assert(result) {
      const opts = (result as any).anthropic;
      if (opts.thinking?.type !== "disabled")
        throw new Error(`Expected type:"disabled", got "${opts.thinking?.type}"`);
    },
  },
  {
    label: 'thinkingSpeed:"fast" + sonnet 4.5 → type:"disabled" (adaptive and effort arrived with 4.6)',
    provider: "anthropic",
    thinkingSpeed: "fast",
    modelName: "claude-sonnet-4-5",
    assert(result) {
      const opts = (result as any).anthropic;
      if (opts.thinking?.type !== "disabled")
        throw new Error(`Expected type:"disabled", got "${opts.thinking?.type}"`);
      if ("effort" in opts) throw new Error(`effort should not be present for sonnet 4.5`);
    },
  },
  {
    label: 'thinkingSpeed:undefined + sonnet 4.5 → type:"disabled" (not adaptive: 4.5 rejects it)',
    provider: "anthropic",
    thinkingSpeed: undefined,
    modelName: "claude-sonnet-4-5",
    assert(result) {
      const opts = (result as any).anthropic;
      if (opts.thinking?.type !== "disabled")
        throw new Error(`Expected type:"disabled", got "${opts.thinking?.type}"`);
    },
  },
  {
    label: 'thinkingSpeed:"fast" + claude-3-5-sonnet-20241022 → type:"disabled"',
    provider: "anthropic",
    thinkingSpeed: "fast",
    modelName: "claude-3-5-sonnet-20241022",
    assert(result) {
      const opts = (result as any).anthropic;
      if (opts.thinking?.type !== "disabled")
        throw new Error(`Expected type:"disabled", got "${opts.thinking?.type}"`);
    },
  },
  {
    label: 'thinkingSpeed:"fast" + fable → no thinking param, effort:"low" (disabled is a 400)',
    provider: "anthropic",
    thinkingSpeed: "fast",
    modelName: "claude-fable-5-1",
    assert(result) {
      const opts = (result as any).anthropic;
      if ("thinking" in opts) throw new Error(`thinking should be omitted for fable`);
      if (opts.effort !== "low") throw new Error(`Expected effort:"low"`);
    },
  },
  {
    label: 'thinkingSpeed:"fast" + haiku → type:"disabled" (no adaptive mode there)',
    provider: "anthropic",
    thinkingSpeed: "fast",
    modelName: "claude-haiku-4-5",
    assert(result) {
      const opts = (result as any).anthropic;
      if (opts.thinking?.type !== "disabled")
        throw new Error(`Expected type:"disabled", got "${opts.thinking?.type}"`);
      if ("effort" in opts)
        throw new Error(`effort should not be present for a non-adaptive model`);
    },
  },
  {
    label: 'thinkingSpeed:"fast" + opus 5.5 → no thinking param, effort:"low" (disabled is a 400)',
    provider: "anthropic",
    thinkingSpeed: "fast",
    modelName: "claude-opus-5-5",
    assert(result) {
      const opts = (result as any).anthropic;
      if ("thinking" in opts)
        throw new Error(`thinking should be omitted, got ${JSON.stringify(opts.thinking)}`);
      if (opts.effort !== "low")
        throw new Error(`Expected effort:"low", got "${opts.effort}"`);
    },
  },
  {
    label: 'thinkingSpeed:"thinking" + no modelName → type:"enabled" (unknown model, non-adaptive path)',
    provider: "anthropic",
    thinkingSpeed: "thinking",
    modelName: undefined,
    assert(result) {
      const opts = (result as any).anthropic;
      // No modelName → anthropicSupportsAdaptiveThinking returns false
      // → falls to legacy path: explicitThinking=true → type:"enabled"
      if (opts.thinking.type !== "enabled")
        throw new Error(`Expected type:"enabled" for explicit thinking with unknown model, got "${opts.thinking.type}"`);
      if (opts.thinking.budgetTokens !== 24000)
        throw new Error(`Expected budgetTokens:24000`);
    },
  },
];

let passed = 0;
let failed = 0;

for (const tc of tests) {
  try {
    const result = getProviderOptions(tc.provider, tc.thinkingSpeed, tc.modelName);
    tc.assert(result);
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
