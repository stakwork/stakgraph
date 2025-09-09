import { z } from "zod";
import { Json, Tool } from "./types.js";
import { Request, Response, NextFunction } from "express";
import * as stakgraph from "./stakgraph/index.js";
import * as stagehand from "./stagehand/tools.js";

const USE_STAGEHAND: boolean =
  process.env.USE_STAGEHAND === "true" || process.env.USE_STAGEHAND === "1";

export function getMcpTools(): Tool[] {
  const coreTools = stakgraph.ALL_TOOLS;
  if (USE_STAGEHAND) {
    coreTools.push(...stagehand.TOOLS);
  }
  return coreTools;
}

export function use_stagehand(): boolean {
  return USE_STAGEHAND;
}

export function bearerToken(req: Request, res: Response, next: NextFunction) {
  const apiToken = process.env.API_TOKEN;
  if (!apiToken) {
    return next();
  }
  const requestToken =
    req.header("Authorization") || req.header("authorization");
  if (!requestToken || requestToken !== `Bearer ${apiToken}`) {
    res.status(401).json({ error: "Unauthorized: Invalid API token" });
    return;
  }
  next();
}

export function mcpSession(req: Request, _: Response, next: NextFunction) {
  const sessionId =
    req.header("mcp-session-id") || req.header("MCP-Session-Id");
  (req as any).sessionId = sessionId;
  next();
}

export function parseSchema(schema: z.ZodSchema): Json {
  const s = z.toJSONSchema(schema);
  delete s["$schema"];
  return s;
}
