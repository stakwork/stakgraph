import { z } from "zod";
import { all_edge_types, all_node_types, EdgeType, NodeType } from "../../types.js";
import {
  booleanLikeSchema,
  csvStringArraySchema,
  optionalIntSchema,
  optionalNumberSchema,
} from "../validation.js";

const nodeTypeSchema = z
  .string()
  .refine(
    (value): value is NodeType => all_node_types().includes(value as NodeType),
    { message: "Invalid node_type" }
  )
  .transform((value) => value as NodeType);

const edgeTypeSchema = z
  .string()
  .refine(
    (value): value is EdgeType => all_edge_types().includes(value as EdgeType),
    { message: "Invalid edge_type" }
  )
  .transform((value) => value as EdgeType);

const outputFormatSchema = z.enum(["snippet", "json"]);
const searchMethodSchema = z.enum(["vector", "fulltext"]);
const directionSchema = z.enum(["up", "down", "both"]);
const limitModeSchema = z.enum(["per_type", "total"]);
const optionalBooleanLikeSchema = z
  .preprocess(
    (value) => (value === undefined ? undefined : value),
    booleanLikeSchema
  )
  .optional();

const nodeTypesSchema = csvStringArraySchema
  .refine(
    (values) =>
      values.every((value) => all_node_types().includes(value as NodeType)),
    { message: "Invalid node_types value" }
  )
  .transform((values) => values as NodeType[]);

export const getNodesQuerySchema = z.object({
  node_type: nodeTypeSchema,
  concise: optionalBooleanLikeSchema,
  ref_ids: z.preprocess(
    (value) => (value === undefined ? undefined : value),
    csvStringArraySchema
  ).optional(),
  output: outputFormatSchema.optional(),
  language: z.string().optional(),
});

export const postNodesBodySchema = z.object({
  node_type: nodeTypeSchema,
  concise: optionalBooleanLikeSchema,
  ref_ids: z.array(z.string()).optional(),
  output: outputFormatSchema.optional(),
  language: z.string().optional(),
});

export const getEdgesQuerySchema = z.object({
  edge_type: edgeTypeSchema.optional(),
  concise: optionalBooleanLikeSchema,
  ref_ids: z.preprocess(
    (value) => (value === undefined ? undefined : value),
    csvStringArraySchema
  ).optional(),
  output: outputFormatSchema.optional(),
  language: z.string().optional(),
});

export const searchQuerySchema = z.object({
  query: z.string().min(1),
  limit: optionalIntSchema,
  concise: optionalBooleanLikeSchema,
  node_types: z
    .preprocess((value) => (value === undefined ? undefined : value), nodeTypesSchema)
    .optional(),
  node_type: nodeTypeSchema.optional(),
  method: searchMethodSchema.optional(),
  output: outputFormatSchema.optional(),
  tests: optionalBooleanLikeSchema,
  max_tokens: optionalIntSchema,
  language: z.string().optional(),
});

export const refIdQuerySchema = z.object({
  ref_id: z.string().min(1),
});

export const workflowQuerySchema = z.object({
  ref_id: z.string().min(1),
  concise: optionalBooleanLikeSchema,
});

export const mapQuerySchema = z
  .object({
    node_type: z.string().optional(),
    name: z.string().optional(),
    file: z.string().optional(),
    ref_id: z.string().optional(),
    tests: optionalBooleanLikeSchema,
    depth: optionalIntSchema,
    direction: directionSchema.optional(),
    trim: z.preprocess(
      (value) => (value === undefined ? undefined : value),
      csvStringArraySchema
    ).optional(),
  })
  .superRefine((value, ctx) => {
    const nodeType = (value.node_type || "").trim();
    const name = (value.name || "").trim();
    const file = (value.file || "").trim();
    const refId = (value.ref_id || "").trim();
    const hasNameAndType = nodeType.length > 0 && name.length > 0;
    const hasFileAndType = nodeType.length > 0 && file.length > 0;
    if (!hasNameAndType && !hasFileAndType && refId.length === 0) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "either node_type+name, node_type+file, or ref_id required",
      });
    }
  });

export const repoMapQuerySchema = z.object({
  name: z.string().optional(),
  ref_id: z.string().optional(),
  node_type: nodeTypeSchema.optional(),
  include_functions_and_classes: optionalBooleanLikeSchema,
});

export const shortestPathQuerySchema = z
  .object({
    start_node_key: z.string().optional(),
    end_node_key: z.string().optional(),
    start_ref_id: z.string().optional(),
    end_ref_id: z.string().optional(),
  })
  .superRefine((value, ctx) => {
    const hasNodePair =
      (value.start_node_key || "").trim().length > 0 &&
      (value.end_node_key || "").trim().length > 0;
    const hasRefPair =
      (value.start_ref_id || "").trim().length > 0 &&
      (value.end_ref_id || "").trim().length > 0;
    if (!hasNodePair && !hasRefPair) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message:
          "Provide start_node_key+end_node_key or start_ref_id+end_ref_id",
      });
    }
  });

export const graphQuerySchema = z.object({
  edge_type: edgeTypeSchema.optional(),
  concise: optionalBooleanLikeSchema,
  edges: optionalBooleanLikeSchema,
  language: z.string().optional(),
  since: optionalNumberSchema,
  limit: optionalIntSchema,
  limit_mode: limitModeSchema.optional(),
  node_types: z
    .preprocess((value) => (value === undefined ? undefined : value), nodeTypesSchema)
    .optional(),
  node_type: nodeTypeSchema.optional(),
  ref_ids: z.preprocess(
    (value) => (value === undefined ? undefined : value),
    csvStringArraySchema
  ).optional(),
});

export type MapQuery = z.infer<typeof mapQuerySchema>;
