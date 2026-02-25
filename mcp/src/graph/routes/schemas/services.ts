import { z } from "zod";
import { booleanLikeSchema, optionalIntSchema } from "../validation.js";

const optionalBooleanLikeSchema = z
  .preprocess(
    (value) => (value === undefined ? undefined : value),
    booleanLikeSchema
  )
  .optional();

export const getServicesQuerySchema = z
  .object({
    clone: optionalBooleanLikeSchema,
    repo_url: z.string().optional(),
    username: z.string().optional(),
    pat: z.string().optional(),
    commit: z.string().optional(),
  })
  .superRefine((value, ctx) => {
    if (value.clone === true && !(value.repo_url || "").trim()) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "repo_url is required when clone=true",
      });
    }
  });

export const mocksInventoryQuerySchema = z.object({
  search: z.string().optional(),
  repo: z.string().optional(),
  limit: optionalIntSchema,
  offset: optionalIntSchema,
});
