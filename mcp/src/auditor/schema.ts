import { z } from "zod";

export const OutcomeSchema = z.enum(["works", "broken", "unknown"]);

export const ClaimSchema = z.object({
  claim: z.string().describe("The specific thing the task claimed to do."),
  verdict: OutcomeSchema.describe(
    "works only if you captured proof; broken with a reason; unknown if you could not tell.",
  ),
  proof: z
    .array(z.string())
    .describe(
      "Evidence ids returned by the capture tool that back this verdict. Required for a works verdict.",
    ),
  reasoning: z
    .string()
    .describe("Why this verdict, referencing what you actually observed."),
});

export const VerdictSchema = z.object({
  overall: OutcomeSchema.describe("Holistic verdict for the whole task."),
  claims: z
    .array(ClaimSchema)
    .describe("Per-claim verdicts, each backed by captured evidence ids."),
  observations: z
    .array(z.string())
    .describe(
      "Feature-level or incidental notes. Never counted as failures of THIS task.",
    ),
  summary: z.string().describe("A short holistic summary of the audit."),
});

export type VerdictInput = z.infer<typeof VerdictSchema>;
