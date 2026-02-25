import { z } from "zod";

const booleanLikeSchema = z
  .union([z.boolean(), z.enum(["true", "false", "1", "0", "True", "False"])])
  .transform((value) => {
    if (typeof value === "boolean") return value;
    return value === "true" || value === "1" || value === "True";
  });

const optionalBooleanLikeSchema = z
  .preprocess((value) => (value === undefined ? undefined : value), booleanLikeSchema)
  .optional();

export const gitreeProcessQuerySchema = z.object({
  owner: z.string().optional(),
  repo: z.string().optional(),
  repo_url: z.string().optional(),
  token: z.string().optional(),
  summarize: optionalBooleanLikeSchema,
  link: optionalBooleanLikeSchema,
  analyze_clues: optionalBooleanLikeSchema,
});

export const gitreeRepoQuerySchema = z.object({
  repo: z.string().optional(),
});

export const gitreeFeatureParamsSchema = z.object({
  id: z.string().min(1),
});

export const gitreeGetFeatureQuerySchema = z.object({
  include: z.string().optional(),
  repo: z.string().optional(),
});

export const gitreeGetPrParamsSchema = z.object({
  number: z.coerce.number().int().positive(),
});

export const gitreeGetCommitParamsSchema = z.object({
  sha: z.string().min(1),
});

export const gitreeFeatureFilesQuerySchema = z.object({
  expand: z.string().optional(),
  output: z.string().optional(),
});

export const gitreeSummarizeFeatureParamsSchema = z.object({
  id: z.string().min(1),
});

export const gitreeLinkFilesQuerySchema = z.object({
  feature_id: z.string().optional(),
});

export const gitreeAllFeaturesGraphQuerySchema = z.object({
  repo: z.string().optional(),
  limit: z.string().optional(),
  node_types: z.string().optional(),
  node_type: z.string().optional(),
  concise: optionalBooleanLikeSchema,
  depth: z.string().optional(),
  per_type_limits: z.string().optional(),
});

export const gitreeRelevantFeaturesBodySchema = z.object({
  prompt: z.string().min(1),
});

export const gitreeCreateFeatureBodySchema = z.object({
  prompt: z.string().min(1),
  name: z.string().min(1),
  owner: z.string().min(1),
  repo: z.string().min(1),
  pat: z.string().optional(),
});

export const gitreeAnalyzeCluesQuerySchema = z.object({
  owner: z.string().min(1),
  repo: z.string().min(1),
  feature_id: z.string().optional(),
  force: optionalBooleanLikeSchema,
  auto_link: optionalBooleanLikeSchema,
  token: z.string().optional(),
});

export const gitreeAnalyzeChangesQuerySchema = z.object({
  owner: z.string().min(1),
  repo: z.string().min(1),
  force: optionalBooleanLikeSchema,
  token: z.string().optional(),
});

export const gitreeListCluesQuerySchema = z.object({
  feature_id: z.string().optional(),
  repo: z.string().optional(),
});

export const gitreeClueParamsSchema = z.object({
  id: z.string().min(1),
});

export const gitreeLinkCluesQuerySchema = z.object({
  owner: z.string().min(1),
  repo: z.string().min(1),
  force: optionalBooleanLikeSchema,
});

export const gitreeSearchCluesBodySchema = z.object({
  query: z.string().min(1),
  featureId: z.string().optional(),
  limit: z.number().int().positive().optional(),
  similarityThreshold: z.number().optional(),
  repo: z.string().optional(),
});

export const gitreeSearchCluesQuerySchema = z.object({
  repo: z.string().optional(),
});

export const gitreeProvenanceBodySchema = z.object({
  conceptIds: z.array(z.string()),
});
