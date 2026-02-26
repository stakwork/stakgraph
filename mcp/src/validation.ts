import { Response } from "express";
import { z } from "zod";

export function sendValidationError(
  res: Response,
  source: "query" | "params" | "body",
  error: z.ZodError
): void {
  res.status(400).json({
    error: "ValidationError",
    message: `Invalid request ${source}`,
    details: error.issues.map((issue) => ({
      path: issue.path.join("."),
      message: issue.message,
      code: issue.code,
    })),
  });
}
