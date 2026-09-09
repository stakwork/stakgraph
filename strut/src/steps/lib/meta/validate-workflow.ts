import { z } from "zod";
import { defineStep } from "../../../core.js";
import { requireAuthoring } from "./_shared.js";

export default defineStep({
  type: "meta/validate-workflow",
  description:
    "Statically check workflow YAML WITHOUT publishing — call before meta/publish-workflow so a typo never becomes a published version. Errors (would fail or hang at run time): YAML/unquoted-template problems, missing/duplicate step ids, unknown step types, `depends` on unknown ids, dependency cycles, template references to unknown roots, config fields that fail a step's schema (template-valued fields are skipped), subflows naming a workflow/version that doesn't exist. Warnings: unknown config fields, `when` without an `if` gate, references to steps that aren't upstream dependencies. Returns { ok, errors: [{ path, message }], warnings: [...], summary }.",
  input: z.object({
    yaml: z.string().describe("Full workflow YAML to check"),
    name: z
      .string()
      .optional()
      .describe("The name you will publish under (lets a YAML without `name:` pass, as publishing stamps it in)."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    return requireAuthoring(ctx.services).validateWorkflow(cfg.yaml, cfg.name);
  },
});
