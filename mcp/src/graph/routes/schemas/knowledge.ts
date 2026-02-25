import { z } from "zod";
import { booleanLikeSchema, optionalNumberSchema } from "../validation.js";

const optionalBooleanLikeSchema = z
  .preprocess(
    (value) => (value === undefined ? undefined : value),
    booleanLikeSchema
  )
  .optional();

export const exploreQuerySchema = z.object({
  prompt: z.string().min(1),
});

export const understandQuerySchema = z.object({
  question: z.string().min(1),
  threshold: optionalNumberSchema,
  provider: z.string().optional(),
});

export const seedUnderstandingQuerySchema = z.object({
  budget: optionalNumberSchema,
  provider: z.string().optional(),
});

export const askQuerySchema = z.object({
  question: z.string().min(1),
  threshold: optionalNumberSchema,
  provider: z.string().optional(),
  maxAgeHours: optionalNumberSchema,
  forceRefresh: optionalBooleanLikeSchema,
  forceCache: optionalBooleanLikeSchema,
});

export const getLearningsQuerySchema = z.object({
  question: z.string().optional(),
});

export const createPullRequestBodySchema = z.object({
  name: z.string().min(1),
  docs: z.string().min(1),
  number: z.string().min(1),
});

export const createLearningBodySchema = z
  .object({
    question: z.string().min(1),
    answer: z.string().min(1),
    context: z.string().optional(),
    featureIds: z.array(z.string()).optional(),
    conceptIds: z.array(z.string()).optional(),
  })
  .superRefine((value, ctx) => {
    const ids = value.conceptIds || value.featureIds;
    const hasContext = (value.context || "").trim().length > 0;
    if (ids === undefined && !hasContext) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Either featureIds/conceptIds or context must be provided",
      });
    }
  });

export const seedStoriesQuerySchema = z.object({
  prompt: z.string().optional(),
  budget: optionalNumberSchema,
  provider: z.string().optional(),
});

export const reconnectQuerySchema = z.object({
  provider: z.string().optional(),
});
