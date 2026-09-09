import { z } from "zod";
import { defineStep } from "../../../core.js";
import { requireAuthoring } from "./_shared.js";

export default defineStep({
  type: "meta/list-workflows",
  description:
    "List all published workflows with each one's active version, versions, description, and publisher stamp. Use to discover what exists before authoring — only workflows stamped publisher 'ai' can be republished, run, or have their run history read through the meta surface.",
  input: z.object({}),
  output: z.any(),
  async run(_cfg, ctx) {
    return requireAuthoring(ctx.services).listWorkflows();
  },
});
