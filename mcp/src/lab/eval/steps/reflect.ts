import { z, defineStep, usageFromResult, computeCost } from "vein";

/**
 * GENERIC optimizer "propose" step: given the CURRENT candidate prompt and the
 * AGGREGATED scoring results across a dataset, emit an improved prompt. This is
 * the step that "changes a param" — its output is the next value for the prompt
 * being tuned, fed back into the next eval round. Pair it with `eval/score` in
 * a loop:
 *
 *   score (per example) → aggregate → reflect → new prompt → score again → …
 *
 * Domain-agnostic: the only domain framing comes from `task` (what the prompt is
 * supposed to do) + optional `rubric`/`guidance`. The concepts experiment wraps
 * this with a concept-specific task in `concepts-eval-reflect`.
 *
 * CRITICAL — generalization (EVAL_SPEC §4/§7): reflect must see the
 * **aggregate over many examples**, not one, or it overfits (fixing one
 * example's specific miss instead of the underlying weakness). So `results` is
 * an ARRAY; the prompt is told to only make changes that help across ALL of
 * them and never encode a single example's specifics.
 *
 * Self-contained: reaches the model via dynamic `import("ai")` (no services),
 * structured output (`generateObject`). Output: { prompt, rationale, usage, cost }
 * — usage/cost are this reflection's own LLM tokens + dollar cost.
 */

const ResultSchema = z.object({
  label: z.string().optional().describe("which example this result is for (e.g. owner/repo)"),
  score: z.number().optional(),
  missing: z.array(z.string()).default([]).describe("expected items the prompt FAILED to produce for this example"),
  spurious: z.array(z.string()).default([]).describe("noise / over-granular items the prompt produced for this example"),
  insight: z.string().optional().describe("the per-example judge's general lesson (already domain-agnostic)"),
});

const ProposalSchema = z.object({
  rationale: z
    .string()
    .describe("2-4 sentences: the GENERAL patterns seen across ALL examples (recurring missing kinds, recurring spurious kinds) and the specific, transferable changes you made to address them. No single-example specifics."),
  prompt: z
    .string()
    .describe("The full improved prompt, ready to drop in. Preserve any {placeholder} tokens present in the current prompt verbatim. Change wording/criteria/structure only."),
});

interface Proposal {
  rationale: string;
  prompt: string;
}

/** Explicit shape of a `results[]` entry — annotated so the aggregation
 *  callbacks below don't depend on zod inference flowing through
 *  `defineStep` (which widens to `any` when vein is consumed as built
 *  `.d.ts` rather than raw `.ts`, tripping noImplicitAny in the prod build). */
interface Result {
  label?: string;
  score?: number;
  missing: string[];
  spurious: string[];
  insight?: string;
}

export default defineStep({
  type: "eval/reflect",
  description:
    "Generic optimizer 'propose' step: from the current candidate prompt + AGGREGATED per-example scoring results (an array — must be multi-example to avoid overfitting), emit an improved prompt. Required config: prompt, results (array of { score?, missing[], spurious[], insight? }). Optional: task (what the prompt should do), rubric (the target / standards), guidance (meta-rules for revising), provider, model. Output: { prompt, rationale, usage, cost }.",
  input: z.object({
    prompt: z.string().describe("the current candidate prompt being tuned"),
    results: z.array(ResultSchema).describe("per-example scoring results across the dataset (>=1; more = less overfit)"),
    task: z.string().optional().describe("one line: what this prompt is supposed to accomplish (the domain framing)"),
    rubric: z.string().optional().describe("what a good output looks like (matching criteria / standards)"),
    guidance: z.string().optional().describe("meta-rules for how to revise (e.g. 'prefer broad names', 'never name a specific item')"),
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

    const results = cfg.results as Result[];

    // Aggregate the dataset into a compact, labeled digest so the model
    // optimizes for the COMMON failure modes, not one example's quirks.
    const digest = results
      .map((r: Result, i: number) => {
        const tag = r.label ?? `example ${i + 1}`;
        const score = r.score != null ? ` (score ${r.score})` : "";
        const missing = r.missing.length ? `\n  missing:  ${r.missing.join(", ")}` : "";
        const spurious = r.spurious.length ? `\n  spurious: ${r.spurious.join(", ")}` : "";
        const insight = r.insight ? `\n  insight:  ${r.insight}` : "";
        return `- ${tag}${score}${missing}${spurious}${insight}`;
      })
      .join("\n");

    const meanScore =
      results.length && results.every((r: Result) => r.score != null)
        ? results.reduce((s: number, r: Result) => s + (r.score ?? 0), 0) / results.length
        : undefined;

    const prompt = `You are optimizing a PROMPT. ${cfg.task ?? "The prompt produces a set of items that we score against a gold set for recall (did we recover the real items?) and precision (did we avoid noise?)."}

# CURRENT PROMPT (what produced the results below)
"""
${cfg.prompt}
"""
${cfg.rubric ? `\n# WHAT "GOOD" LOOKS LIKE (rubric)\n${cfg.rubric}\n` : ""}${cfg.guidance ? `\n# HOW TO REVISE (guidance)\n${cfg.guidance}\n` : ""}
# RESULTS ACROSS ${cfg.results.length} EXAMPLE(S)${meanScore != null ? ` — mean score ${Math.round(meanScore * 100) / 100}` : ""}
${digest}

Propose a BETTER version of the prompt that raises the mean score across ALL of
these examples. Rules:
- Generalize. Only make changes justified by patterns that recur across MULTIPLE
  examples. Never encode a single example's specific domain or missing item —
  encode the underlying PRINCIPLE instead.
- Recall first: the biggest lever is recovering recurring 'missing' kinds.
- Cut recurring 'spurious' kinds by sharpening the inclusion/exclusion criteria.
- Preserve any {placeholder} tokens present in the current prompt verbatim.
- Return the COMPLETE new prompt (not a diff).`;

    const { object, usage: rawUsage } = await generateObject({ model, prompt, schema: ProposalSchema as any });
    const p = object as Proposal;
    const usage = usageFromResult(rawUsage);
    return { prompt: p.prompt, rationale: p.rationale, usage, cost: computeCost(provider, usage) };
  },
});
