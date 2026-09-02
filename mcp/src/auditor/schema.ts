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
      "Probe-captured evidence ids that back this verdict — ids returned by http_request, sample, read_logs, browser_extract, browser_screenshot, or browser_current_url. Notes from the capture tool do NOT count. A works verdict with no such id is downgraded to unknown.",
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
