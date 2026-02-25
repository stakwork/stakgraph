import { z } from "zod";

export const mcpServerSchema = z.object({
  name: z.string().min(1),
  url: z.string().min(1),
  token: z.string().optional(),
  headers: z.record(z.string(), z.string()).optional(),
  toolFilter: z.array(z.string()).optional(),
});

export const sessionConfigSchema = z.object({
  truncateToolResults: z.boolean().optional(),
  maxToolResultLines: z.number().int().positive().optional(),
  maxToolResultChars: z.number().int().positive().optional(),
});

export const repoAgentBodySchema = z.object({
  repo_url: z.string().min(1),
  username: z.string().optional(),
  pat: z.string().optional(),
  commit: z.string().optional(),
  branch: z.string().optional(),
  prompt: z.union([z.string().min(1), z.array(z.any())]),
  toolsConfig: z
    .record(z.string(), z.union([z.string(), z.boolean(), z.null()]))
    .optional(),
  jsonSchema: z.record(z.string(), z.any()).optional(),
  model: z.string().optional(),
  logs: z.boolean().optional(),
  sessionId: z.string().optional(),
  sessionConfig: sessionConfigSchema.optional(),
  mcpServers: z.array(mcpServerSchema).optional(),
  systemOverride: z.string().optional(),
});

export const getAgentSessionQuerySchema = z
  .object({
    session_id: z.string().optional(),
    sessionId: z.string().optional(),
  })
  .superRefine((value, ctx) => {
    if (!(value.session_id || "").trim() && !(value.sessionId || "").trim()) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Missing session_id",
      });
    }
  });

export const getLeaksQuerySchema = z.object({
  repo_url: z.string().min(1),
  username: z.string().optional(),
  pat: z.string().optional(),
  commit: z.string().optional(),
  ignore: z.string().optional(),
});
