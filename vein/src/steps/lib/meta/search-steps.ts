import { z } from "zod";
import { defineStep } from "../../../core.js";
import { requireAuthoring } from "./_shared.js";

export default defineStep({
  type: "meta/search-steps",
  description:
    "Search step types by keyword — matches type names and descriptions across core, lib, and custom steps. Returns ranked matches. Use to check whether a step already exists before authoring one.",
  input: z.object({
    query: z.string().describe("Search keywords, e.g. 'github pr' or 'http request'"),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    return requireAuthoring(ctx.services).searchSteps(cfg.query);
  },
});
