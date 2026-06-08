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
    modelName: "claude-sonnet-4-6",
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
    label: 'thinkingSpeed:"fast" + adaptive model → type:"disabled"',
    provider: "anthropic",
    thinkingSpeed: "fast",
    modelName: "claude-opus-4-7",
    assert(result) {
      const opts = (result as any).anthropic;
      if (opts.thinking.type !== "disabled")
        throw new Error(`Expected type:"disabled", got "${opts.thinking.type}"`);
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
