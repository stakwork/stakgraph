import { z, defineStep } from "vein";

/**
 * The drafting fan-out plan (the derive_basename + build_drafter_plan
 * equivalent). Pure computation: given the canonical deliverable filenames
 * and the fan-out widths, emit per-drafter working directories and the
 * per-verifier critique filenames the later phases read by convention.
 *
 * Conventions (relative to the run's artifacts dir):
 *  - drafter k writes its full deliverable set under  ./draft_k/
 *  - verifier <name> writes                            ./critiques/critique-<name>.md
 *  - the aggregator writes the FINAL canonical files   ./output/<deliverable>
 */
export default defineStep({
  type: "harvey/drafter-plan",
  description:
    "Compute the drafting fan-out plan: per-drafter write dirs (draft_k/), per-verifier critique file " +
    "paths (critiques/critique-<name>.md), and the canonical output filenames (output/<name>). Pure — " +
    "no filesystem access. Output: { basename, canonical, drafts: [{ k, dir, files }], critiqueFiles }.",
  input: z.object({
    deliverables: z
      .array(z.string())
      .min(1)
      .describe("Canonical deliverable filenames (exact names the aggregator must produce in ./output/)."),
    drafters: z.number().int().positive().max(8).default(1).describe("How many parallel drafters."),
    verifiers: z
      .array(z.string())
      .default(["completeness", "correctness", "arithmetic", "doctrine"])
      .describe("Verifier names — one critique file per name."),
  }),
  output: z.any(),
  async run(cfg) {
    const basename = (cfg.deliverables[0] ?? "deliverable").replace(/\.[^.]*$/, "");
    const drafts = Array.from({ length: cfg.drafters }, (_, i) => {
      const k = i + 1;
      const dir = `draft_${k}`;
      return { k, dir, files: cfg.deliverables.map((d) => `${dir}/${d}`) };
    });
    return {
      basename,
      canonical: cfg.deliverables,
      outputFiles: cfg.deliverables.map((d) => `output/${d}`),
      drafts,
      critiqueFiles: cfg.verifiers.map((v) => `critiques/critique-${v}.md`),
    };
  },
});
