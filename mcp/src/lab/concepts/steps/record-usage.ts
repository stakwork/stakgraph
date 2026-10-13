import { z, defineStep } from "strut";
import type { ConceptServices } from "../services.js";

/**
 * Add what a sweep spent to the repo's running total — the
 * `cumulativeUsage` that `GET /gitree/concepts` reports, which
 * `gitree_process` grew with its bootstrap + process + summarize usage.
 *
 * Each entry of `usage` is one step's usage (the `collect-results` rollup,
 * a `summarize` or `bootstrap-explore` output); a missing or malformed entry
 * counts as zero. The sum carries both the base fields and the legacy
 * `inputTokens` / `outputTokens` / `totalTokens` the store reads.
 * Output: { repo, usage }.
 */

interface Usage {
  input: number;
  cache_read: number;
  cache_write: number;
  output: number;
  total: number;
}

const n = (v: unknown): number => (typeof v === "number" && Number.isFinite(v) ? v : 0);

function add(a: Usage, b: unknown): Usage {
  const u = (b ?? {}) as Record<string, unknown>;
  return {
    input: a.input + n(u.input),
    cache_read: a.cache_read + n(u.cache_read),
    cache_write: a.cache_write + n(u.cache_write),
    output: a.output + n(u.output),
    total: a.total + n(u.total),
  };
}

export default defineStep({
  type: "concepts/record-usage",
  description:
    "Add the given step usages to the repo's cumulative total (what GET /gitree/concepts reports as cumulativeUsage). Output: { repo, usage }.",
  input: z.object({
    owner: z.string(),
    repo: z.string(),
    usage: z.array(z.any()).default([]).describe("step usages to add; missing entries count as zero"),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    const { storage } = ctx.services as ConceptServices;
    const repoId = `${cfg.owner}/${cfg.repo}`;
    let sum: Usage = { input: 0, cache_read: 0, cache_write: 0, output: 0, total: 0 };
    for (const u of cfg.usage) sum = add(sum, u);
    const usage = {
      ...sum,
      inputTokens: sum.input + sum.cache_read + sum.cache_write,
      outputTokens: sum.output,
      totalTokens: sum.total,
    };
    await storage.addToTotalUsage(repoId, usage);
    return { repo: repoId, usage };
  },
});
