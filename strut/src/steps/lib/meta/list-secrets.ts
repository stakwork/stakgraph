import { z } from "zod";
import { defineStep } from "../../../core.js";
import { requireAuthoring } from "./_shared.js";

export default defineStep({
  type: "meta/list-secrets",
  description:
    "List the NAMES of credentials in the deployment's secret store (never the values). Use before authoring a step that needs auth: reference an existing name via ctx.services.secrets.get(\"NAME\") in the step source.",
  input: z.object({}),
  output: z.any(),
  async run(_cfg, ctx) {
    return requireAuthoring(ctx.services).listSecrets();
  },
});
