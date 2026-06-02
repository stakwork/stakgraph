import { z, defineStep } from "vein";

/**
 * Recall-oriented LLM-as-judge for concept sets. The judge MATCHES each
 * expected (gold) capability to a produced concept semantically and reports
 * matched / missing / spurious; the step then computes a continuous score
 * (recall-weighted F-beta) deterministically from those counts — so the number
 * is consistent and the gradient is informative for optimization.
 *
 * Pure mechanism: the matching criteria (WHAT counts as a match, what's
 * "spurious", naming standards) live in the `rubric` param (the experiment
 * surface). The step enforces the structured output contract + the scoring math.
 *
 * Self-contained: reaches the model via dynamic `import("ai")` (no services),
 * using structured output (`generateObject`).
 *
 * Output: { score, recall, precision, matched, missing, spurious, reason,
 *           insight, markdown }.
 */

const VerdictSchema = z.object({
  matched: z
    .array(z.object({ expected: z.string(), produced: z.string() }))
    .describe("Each expected capability that IS present in the produced set, paired with the produced concept that covers it."),
  missing: z
    .array(z.string())
    .describe("Expected capabilities with NO adequate match in the produced set."),
  spurious: z
    .array(z.string())
    .describe("Produced concepts that are noise — not real user-facing capabilities, or over-granular/implementation-named, or not in the expected set."),
  reason: z.string().describe("1-3 sentences on the biggest gaps."),
  insight: z.string().describe("1 sentence: what should change to recover the missing capabilities?"),
});

interface Verdict {
  matched: Array<{ expected: string; produced: string }>;
  missing: string[];
  spurious: string[];
  reason: string;
  insight: string;
}

const BETA = 2; // weight recall this many times more than precision

function fBeta(recall: number, precision: number): number {
  const b2 = BETA * BETA;
  const denom = b2 * precision + recall;
  return denom === 0 ? 0 : ((1 + b2) * precision * recall) / denom;
}

export default defineStep({
  type: "eval/score",
  description:
    "Recall-based LLM-judge for concept sets: matches expected (gold) capabilities to produced concepts, computes a continuous recall-weighted score. Required config: actual, expected, rubric (rubric is the calling workflow's param — the experiment surface). Optional: provider, model. Output: { score, recall, precision, matched, missing, spurious, reason, insight, markdown }.",
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

    const { generateObject } = await import("ai");
    let model: any;
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

# EXPECTED (gold capabilities)
${cfg.expected}

# PRODUCED (what the workflow generated)
${cfg.actual}

Match each EXPECTED capability to a PRODUCED concept SEMANTICALLY (not by exact
wording — a capability can be named differently). Then report:
- matched:  expected capabilities that ARE covered (paired with the produced concept)
- missing:  expected capabilities with no adequate match
- spurious: produced concepts that are noise / not real capabilities / not expected
Every expected capability must appear in exactly one of matched or missing.`;

    const { object } = await generateObject({ model, prompt, schema: VerdictSchema as any });
    const v = object as Verdict;

    const expectedTotal = v.matched.length + v.missing.length;
    const producedTotal = v.matched.length + v.spurious.length;
    const recall = expectedTotal === 0 ? 1 : v.matched.length / expectedTotal;
    const precision = producedTotal === 0 ? 1 : v.matched.length / producedTotal;
    const score = Math.round(fBeta(recall, precision) * 100) / 100;

    const pct = (n: number) => `${Math.round(n * 100)}%`;
    const markdown = [
      `**Score: ${score}**  (recall ${pct(recall)}, precision ${pct(precision)})`,
      `Matched ${v.matched.length}/${expectedTotal} expected capabilities.`,
      v.missing.length ? `\n**Missing:** ${v.missing.join(", ")}` : "",
      v.spurious.length ? `**Spurious:** ${v.spurious.join(", ")}` : "",
      `\n${v.reason}`,
      v.insight ? `\n_Insight: ${v.insight}_` : "",
    ]
      .filter(Boolean)
      .join("\n");

    return {
      score,
      recall,
      precision,
      matched: v.matched,
      missing: v.missing,
      spurious: v.spurious,
      reason: v.reason,
      insight: v.insight,
      markdown,
    };
  },
});
