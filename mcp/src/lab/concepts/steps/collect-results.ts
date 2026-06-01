import { z, defineStep } from "vein";

/**
 * Merge the per-change results from a foreach into a single rollup: union of
 * modified concept ids + summed token usage. Output:
 * { modifiedConceptIds, usage, count }.
 *
 * Self-contained: the (small) usage accumulation is inlined.
 */

interface Usage {
  input: number;
  cache_read: number;
  cache_write: number;
  output: number;
  total: number;
}

function emptyUsage(): Usage {
  return { input: 0, cache_read: 0, cache_write: 0, output: 0, total: 0 };
}

function addUsage(a: Usage, b: any): Usage {
  const n = (v: unknown) =>
    typeof v === "number" && Number.isFinite(v) ? v : 0;
  return {
    input: a.input + n(b?.input),
    cache_read: a.cache_read + n(b?.cache_read),
    cache_write: a.cache_write + n(b?.cache_write),
    output: a.output + n(b?.output),
    total: a.total + n(b?.total),
  };
}

export default defineStep({
  type: "concepts/collect-results",
  description: "Merge per-change results: union modifiedConceptIds + sum usage. Output: { modifiedConceptIds, usage, count }.",
  input: z.object({
    results: z.array(z.any()),
  }),
  output: z.any(),
  async run(cfg) {
    const ids = new Set<string>();
    let usage = emptyUsage();
    for (const r of cfg.results ?? []) {
      for (const id of r?.modifiedConceptIds ?? []) ids.add(id);
      if (r?.usage) usage = addUsage(usage, r.usage);
    }
    return {
      modifiedConceptIds: [...ids],
      usage,
      count: (cfg.results ?? []).length,
    };
  },
});
