import { z } from "zod";
import { defineStep } from "../../../core.js";
import { requireAuthoring } from "./_shared.js";

export default defineStep({
  type: "meta/create-step",
  description:
    "Author a NEW custom step type from TypeScript source. The code is a self-contained vein step: `import { z, defineStep } from \"vein\"` and `export default defineStep({ type, input, output, async run(cfg, ctx) {...} })`. Reach external capabilities through `ctx.services` — `ctx.services.http(url, opts)` for network calls, `ctx.services.secrets.get(name)` for credentials (NOT global fetch / process.env), so the step is recordable/replayable by meta/run-step's cassette. The source is load-verified before returning: a broken step comes back as an error with the import failure, not a silent no-op. Use meta/edit-step to change an existing step. Publishing creates version v1.",
  input: z.object({
    name: z
      .string()
      .describe("Step type name. Slashes nest it (e.g. 'candidates/my-fetcher') and become the registry type."),
    code: z
      .string()
      .describe(
        'Full TypeScript source. Shape: import { z, defineStep } from "vein"; export default defineStep({ type: "<name>", input: z.object({...}), output: z.any(), async run(cfg, ctx) {...} });',
      ),
    description: z.string().optional(),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    return requireAuthoring(ctx.services).createStep(cfg.name, cfg.code, cfg.description);
  },
});
