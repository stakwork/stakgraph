import { Request, Response, NextFunction } from "express";
import {
  ask_question,
  QUESTIONS,
  LEARN_HTML,
  ask_prompt,
  learnings,
} from "../../tools/intelligence/index.js";

export function logEndpoint(req: Request, res: Response, next: NextFunction) {
  if (req.headers["x-api-token"]) {
    console.log(`=> ${req.method} ${req.path} [auth]`);
  } else {
    console.log(`=> ${req.method} ${req.path}`);
  }
  next();
}

export function authMiddleware(
  req: Request,
  res: Response,
  next: NextFunction
) {
  const apiToken = process.env.API_TOKEN;
  if (!apiToken) {
    return next();
  }

  // Check for x-api-token header
  const requestToken = req.header("x-api-token");
  if (requestToken && requestToken === apiToken) {
    return next();
  }

  // Check for Basic Auth header
  const authHeader = req.header("Authorization") || req.header("authorization");
  if (authHeader && authHeader.startsWith("Basic ")) {
    try {
      const base64Credentials = authHeader.substring(6);
      const credentials = Buffer.from(base64Credentials, "base64").toString(
        "ascii"
      );
      const [username, token] = credentials.split(":");
      if (token && token === apiToken) {
        return next();
      }
    } catch (error) {
      // Invalid base64 encoding
    }
  }

  res.status(401).json({ error: "Unauthorized: Invalid API token" });
  return;
}

export function learn(req: Request, res: Response) {
  const apiToken = process.env.API_TOKEN;
  if (!apiToken) {
    res.setHeader("Content-Type", "text/html");
    res.send(LEARN_HTML);
    return;
  }

  // Check if user is already authenticated
  const authHeader = req.header("Authorization") || req.header("authorization");
  if (authHeader && authHeader.startsWith("Basic ")) {
    try {
      const base64Credentials = authHeader.substring(6);
      const credentials = Buffer.from(base64Credentials, "base64").toString(
        "ascii"
      );
      const [username, token] = credentials.split(":");

      if (token && token === apiToken) {
        res.setHeader("Content-Type", "text/html");
        res.send(LEARN_HTML);
        return;
      }
    } catch (error) {
      // Invalid base64 encoding, fall through to challenge
    }
  }

  // Send Basic Auth challenge
  res.setHeader("WWW-Authenticate", 'Basic realm="API Access"');
  res.status(401).send("Authentication required");
}
