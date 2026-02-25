import { Request, Response } from "express";
import { z } from "zod";
import { sendValidationError } from "../../validation.js";

const booleanLikeValues = ["true", "false", "1", "0", "True", "False"];

export const booleanLikeSchema = z
  .union([z.boolean(), z.enum(booleanLikeValues as [string, ...string[]])])
  .transform((value) => {
    if (typeof value === "boolean") return value;
    return value === "true" || value === "1" || value === "True";
  });

export const optionalNumberSchema = z.preprocess((value) => {
  if (value === undefined || value === null || value === "") return undefined;
  return value;
}, z.coerce.number().optional());

export const optionalIntSchema = z.preprocess((value) => {
  if (value === undefined || value === null || value === "") return undefined;
  return value;
}, z.coerce.number().int().optional());

export const csvStringArraySchema = z
  .union([z.string(), z.array(z.string())])
  .transform((value) => {
    if (Array.isArray(value)) {
      return value.map((item) => item.trim()).filter((item) => item.length > 0);
    }
    return value
      .split(",")
      .map((item) => item.trim())
      .filter((item) => item.length > 0);
  });

function validationError(
  res: Response,
  source: "query" | "body" | "params",
  error: z.ZodError
): null {
  sendValidationError(res, source, error);
  return null;
}

function parsePart<T extends z.ZodTypeAny>(
  source: "query" | "body" | "params",
  res: Response,
  schema: T,
  input: unknown
): z.output<T> | null {
  const result = schema.safeParse(input);
  if (!result.success) {
    return validationError(res, source, result.error);
  }
  return result.data;
}

export function parseQuery<T extends z.ZodTypeAny>(
  req: Request,
  res: Response,
  schema: T
): z.output<T> | null {
  return parsePart("query", res, schema, req.query);
}

export function parseBody<T extends z.ZodTypeAny>(
  req: Request,
  res: Response,
  schema: T
): z.output<T> | null {
  return parsePart("body", res, schema, req.body);
}

export function parseParams<T extends z.ZodTypeAny>(
  req: Request,
  res: Response,
  schema: T
): z.output<T> | null {
  return parsePart("params", res, schema, req.params);
}
