import { z } from "zod";
import { defineStep } from "../../../core.js";
import { requireAuthoring } from "./_shared.js";

export default defineStep({
  type: "meta/run-step",
  description:
    "Run a SINGLE step in isolation with a given config + input and return its output + events — the inner loop for authoring: meta/create-step → meta/run-step → meta/edit-step → meta/run-step until the output is right. Set cassette:'record' to run live AND capture the step's external service calls to a reusable fixture (secrets scrubbed); then cassette:'replay' to iterate OFFLINE against it — deterministic, no rate limits, no cost, no side effects. Sees steps published earlier in this same run (the registry is re-read fresh). Returns { status, output?, error?, events, recorded? }.",
  input: z.object({
    type: z.string().describe("Step type to run, e.g. 'candidates/my-fetcher' or 'http'."),
    config: z
      .record(z.string(), z.any())
      .optional()
      .describe("The step's config (same shape as in a workflow). Templates like {{ input.* }} / {{ params.* }} are resolved."),
    input: z.any().optional().describe("Workflow input object, referenced in config via {{ input.* }}."),
    params: z.record(z.string(), z.any()).optional().describe("Params knobs, referenced via {{ params.* }}."),
    cassette: z
      .enum(["record", "replay"])
      .optional()
      .describe("record: run live + capture external calls to a fixture. replay: serve them from the fixture (offline). Omit for a plain live run."),
    cassetteName: z
      .string()
      .optional()
      .describe("Fixture name (defaults to the step type). Use distinct names to keep multiple scenarios per step."),
  }),
  output: z.any(),
  async run(cfg, ctx) {
    return requireAuthoring(ctx.services).runStep(cfg.type, {
      config: cfg.config,
      input: cfg.input,
      params: cfg.params,
      cassette: cfg.cassette,
      cassetteName: cfg.cassetteName,
    });
  },
});
