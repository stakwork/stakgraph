import { z } from "zod";
import { booleanLikeSchema } from "../validation.js";

const optionalBooleanLikeSchema = z
  .preprocess(
    (value) => (value === undefined ? undefined : value),
    booleanLikeSchema
  )
  .optional();

export const gitseeBodySchema = z.object({
  owner: z.string().min(1),
  repo: z.string().min(1),
  data: z
    .array(
      z.enum([
        "contributors",
        "icon",
        "repo_info",
        "commits",
        "branches",
        "files",
        "stats",
        "file_content",
        "exploration",
      ])
    )
    .min(1),
  filePath: z.string().optional(),
  explorationMode: z.enum(["features", "first_pass"]).optional(),
  explorationPrompt: z.string().optional(),
  cloneOptions: z.record(z.string(), z.any()).optional(),
  useCache: optionalBooleanLikeSchema,
});

export const gitseeEventsParamsSchema = z.object({
  owner: z.string().min(1),
  repo: z.string().min(1),
});

export const gitseeServicesQuerySchema = z.object({
  owner: z.string().min(1),
  repo: z.string().min(1),
  username: z.string().optional(),
  pat: z.string().optional(),
});

export const gitseeAgentQuerySchema = z.object({
  owner: z.string().min(1),
  repo: z.string().min(1),
  prompt: z.string().min(1),
  system: z.string().optional(),
  final_answer: z.string().optional(),
  username: z.string().optional(),
  pat: z.string().optional(),
});

export const requestIdQuerySchema = z.object({
  request_id: z.string().min(1),
});
