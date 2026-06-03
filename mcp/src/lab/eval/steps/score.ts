import { z, defineStep } from "vein";

/**
 * GENERIC recall-oriented LLM-as-judge. Domain-agnostic: it matches a PRODUCED
 * set of items against an EXPECTED (gold) set and computes a continuous,
 * recall-weighted score (F-beta) deterministically from the match counts — so
 * the number is consistent and the gradient is informative for optimization.
 *
 * Nothing here is concepts-specific. WHAT counts as a match / what's "spurious"
 * / naming standards all come from the `rubric` input (the experiment surface);
 * the step only enforces the structured-output contract + the scoring math. An
 * experiment supplies its own rubric (e.g. the concepts experiment ships a
 * concept-quality rubric in `concepts-eval-score`).
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
    .describe("Each EXPECTED item that IS present in the produced set, paired with the produced item that covers it."),
  missing: z
    .array(z.string())
    .describe("EXPECTED items with NO adequate match in the produced set."),
  spurious: z
    .array(z.string())
    .describe("PRODUCED items that are noise — not a real item per the rubric, over-granular/implementation-named, or not in the expected set."),
  reason: z.string().describe("1-3 sentences on the biggest gaps in THIS result (specifics are fine here)."),
  insight: z
    .string()
    .describe(
      "1 sentence: a GENERAL, domain-agnostic lesson about the GENERATOR (the prompt/workflow being evaluated) that would help on ANY input. Do NOT name this input's specific items or domain — generalize the failure into a transferable principle (e.g. 'the generator over-weights low-level/implementation items and under-weights top-level user-facing ones').",
    ),
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
    "Generic recall-based LLM-judge: matches a produced set against an expected (gold) set and computes a continuous recall-weighted score. Domain-agnostic — all criteria live in the `rubric`. Required config: actual, expected, rubric. Optional: provider, model. Output: { score, recall, precision, matched, missing, spurious, reason, insight, markdown }.",
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

# EXPECTED (gold)
${cfg.expected}

# PRODUCED (what the generator produced)
${cfg.actual}

Match each EXPECTED item to a PRODUCED item SEMANTICALLY (not by exact wording —
an item can be named differently). Then report:
- matched:  expected items that ARE covered (paired with the produced item)
- missing:  expected items with no adequate match
- spurious: produced items that are noise / not real items / not expected
Every expected item must appear in exactly one of matched or missing.

For "insight", step back from THIS input: give one transferable, domain-agnostic
lesson about the generator (never name this input's specific items or domain). It
must read as advice that applies to any input.`;

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
      `Matched ${v.matched.length}/${expectedTotal} expected items.`,
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
