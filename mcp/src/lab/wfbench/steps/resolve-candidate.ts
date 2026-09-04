import { z, defineStep } from "vein";

/**
 * Resolve WHAT the author actually shipped — never trust its echo (the
 * gaia-evolve-gen lesson: in schema mode a no-tool text turn ends the loop
 * and IS the structured output, so "placeholder" versions happen).
 *
 *   version = vpin (the echoed version, if it resolves) || vactive (the
 *             candidate's active version — the author's own last publish)
 *   published = a version resolved AND it differs from vbefore (the active
 *               version BEFORE the author ran) — at first run vbefore is
 *               { error } and any publish trips it.
 *
 * meta/get-workflow returns { error } instead of throwing, so a bad pin
 * simply has no `version`.
 */
const ok = (w: any) => w && typeof w === "object" && !w.error && typeof w.version === "string" && w.version;

export default defineStep({
  type: "wfbench/resolve-candidate",
  description:
    "Resolve the produced workflow from the author's structured output plus meta/get-workflow reads: version = vpin || vactive, published = version differs from vbefore. Output: { workflow, version, published, yaml, summary, changes, customSteps, missingSecrets, authorCost, authorSteps, authorError }.",
  input: z.object({
    author: z.any().describe("The author agent step's output ({ object, cost, steps } or an onError pack { error })."),
    candidate: z.string().describe("The harness-pinned candidate workflow name."),
    vbefore: z.any().optional().describe("meta/get-workflow of the candidate BEFORE the author ran."),
    vpin: z.any().optional().describe("meta/get-workflow of the candidate at the author's echoed version."),
    vactive: z.any().optional().describe("meta/get-workflow of the candidate's active version AFTER the author ran."),
  }),
  output: z.any(),
  async run(cfg) {
    const a = cfg.author && typeof cfg.author === "object" ? cfg.author : {};
    const obj = a.object && typeof a.object === "object" ? a.object : {};
    const src = ok(cfg.vpin) ? cfg.vpin : ok(cfg.vactive) ? cfg.vactive : null;
    const version: string | null = src ? src.version : null;
    const before: string | null = ok(cfg.vbefore) ? cfg.vbefore.version : null;
    const published = !!version && version !== before;
    const strs = (v: unknown) => (Array.isArray(v) ? v.filter((x) => typeof x === "string") : []);
    return {
      workflow: cfg.candidate,
      version,
      published,
      version_before: before,
      yaml: published && typeof src?.yaml === "string" ? src.yaml : "",
      summary: typeof obj.summary === "string" ? obj.summary : "",
      changes: strs(obj.changes),
      customSteps: strs(obj.customSteps),
      missingSecrets: Array.isArray(obj.missingSecrets) ? obj.missingSecrets : [],
      authorCost: typeof a.cost === "number" ? a.cost : null,
      authorSteps: typeof a.steps === "number" ? a.steps : null,
      authorError: typeof a.error === "string" ? a.error : null,
    };
  },
});
