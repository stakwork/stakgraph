import { z } from "zod";

export const sessionConfigSchema = z.object({
  truncateToolResults: z.boolean().optional(),
  maxToolResultLines: z.number().int().positive().optional(),
  maxToolResultChars: z.number().int().positive().optional(),
});

export const stakworkRunSummarySchema = z.object({
  projectId: z.number(),
  type: z.string(),
  status: z.string(),
  feature: z.string().nullable().optional(),
  createdAt: z.string(),
  agentLogs: z.array(z.object({
    agent: z.string(),
    url: z.string(),
  })).optional(),
});

export const logsAgentBodySchema = z.object({
  prompt: z.string().min(1),
  model: z.string().optional(),
  logs: z.boolean().optional(),
  swarmName: z.string().optional(),
  sessionId: z.string().optional(),
  sessionConfig: sessionConfigSchema.optional(),
  stakworkApiKey: z.string().optional(),
  stakworkRuns: z.array(stakworkRunSummarySchema).optional(),
  printAgentProgress: z.boolean().optional(),
  workspaceSlug: z.string().optional(),
});
