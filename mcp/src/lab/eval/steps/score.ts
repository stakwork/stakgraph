import { z, defineStep } from "vein";

/**
 * LLM-as-judge scorer. Compares an `actual` output against an `expected` gold
 * standard using a `rubric` (the scoring criteria) and returns a structured
 * verdict: { score, reason, insight }.
 *
 * Pure mechanism: the rubric (WHAT to weigh) is NOT baked in here — it's the
 * experiment surface and lives in the calling workflow's `params` block (see
 * `workflows/eval-score.yaml`), passed in via config. The step only enforces
 * the strict SCORE/REASON/INSIGHT output contract so the result parses.
 *
 * Self-contained: like vein's core `llm` step, it reaches the model via a
 * dynamic `import("ai")` — no services needed — so it's portable as a custom
 * step on disk.
 *
 * Output: { score, reason, insight, markdown }.
 */

const FORMAT = `Respond in EXACTLY this format and nothing else:
SCORE: <0 or 0.5 or 1>
REASON: <1-3 sentences on what is right or wrong>
INSIGHT: <1 sentence: what should change to score higher next time?>`;

function parseVerdict(text: string): { score: number; reason: string; insight: string } {
  const scoreMatch = text.match(/SCORE:\s*(0\.5|0|1)/i);
  const reasonMatch = text.match(/REASON:\s*([\s\S]*?)(?=\nINSIGHT:|$)/i);
  const insightMatch = text.match(/INSIGHT:\s*([\s\S]*)/i);
  return {
    score: scoreMatch ? parseFloat(scoreMatch[1]!) : 0,
    reason: reasonMatch ? reasonMatch[1]!.trim().split("\n")[0]! : "Could not parse judge response",
    insight: insightMatch ? insightMatch[1]!.trim().split("\n")[0]! : "",
  };
}

export default defineStep({
  type: "eval/score",
  description:
    "LLM-as-judge: score `actual` against `expected` using `rubric` criteria. Required config: actual, expected, rubric (rubric supplied by the calling workflow's `params` — the experiment surface). Optional: provider, model. Output: { score, reason, insight, markdown }.",
  input: z.object({
    actual: z.string(),
    expected: z.string(),
    rubric: z.string(),
    provider: z.string().optional(),
    model: z.string().optional(),
  }),
  output: z.any(),
  async run(cfg) {
    const provider = cfg.provider ?? process.env["VEIN_LLM_PROVIDER"] ?? "anthropic";
    const modelName = cfg.model ?? process.env["VEIN_LLM_MODEL"];

    const { generateText } = await import("ai");
    let model: Parameters<typeof generateText>[0]["model"];
    switch (provider) {
      case "anthropic": {
        const { anthropic } = await import("@ai-sdk/anthropic");
        model = anthropic(modelName ?? "claude-sonnet-4-20250514");
        break;
      }
      case "openai": {
        const { openai } = await import("@ai-sdk/openai");
        model = openai(modelName ?? "gpt-4o");
        break;
      }
      default:
        throw new Error(`Unknown LLM provider: "${provider}". Supported: anthropic, openai`);
    }

    const prompt = `${cfg.rubric}

# EXPECTED (gold standard)
${cfg.expected}

# ACTUAL (produced)
${cfg.actual}

${FORMAT}`;

    const { text } = await generateText({ model, prompt });
    const verdict = parseVerdict(text);

    const markdown = `**Score: ${verdict.score}**\n\n${verdict.reason}${
      verdict.insight ? `\n\n_Insight: ${verdict.insight}_` : ""
    }`;

    return { ...verdict, markdown };
  },
});
