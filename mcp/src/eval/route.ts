import express, { Express } from "express";
import { evaluate } from "./stagehand.js";
import { promises as dns } from "dns";
import {
  getConsoleLogs,
  getSessionId,
  exportSessionDetails,
} from "../tools/stagehand/utils.js";

export async function evalRoutes(app: Express) {
  app.post("/evaluate", async (req: express.Request, res: express.Response) => {
    try {
      const test_url = req.body.test_url || req.body.base_url;
      const prompt = req.body.prompt || req.body.instruction;

      if (!test_url) {
        res.status(400).json({ error: "Missing test_url" });
        return;
      }

      if (!prompt) {
        res.status(400).json({ error: "Missing prompt or instruction" });
        return;
      }

      const result = await evaluate(test_url, prompt);
      res.json(result);
    } catch (error) {
      console.error("Evaluation failed:", error);
      res.status(500).json({
        error: "Evaluation failed",
        message: error instanceof Error ? error.message : "Unknown error",
      });
    }
  });

  // Simple HTTP endpoint for console logs (Phase 2A: Dual Access Pattern)
  // Future Phase 2B: Add GET /console-logs/stream for real-time SSE streaming
  // - Streams logs immediately as they're captured vs current batch approach
  // - Client: const stream = new EventSource('/console-logs/stream')
  // - Perfect for live monitoring during automation runs
  // - Implementation: Modify addConsoleLog() to broadcast to streaming clients
  app.get(
    "/console-logs",
    async (req: express.Request, res: express.Response) => {
      try {
        const logs = getConsoleLogs();
        const sessionId = getSessionId();

        res.json({
          logs,
          timestamp: new Date().toISOString(),
          count: logs.length,
          metadata: {
            session_active: true,
            access_method: "http_rest",
            session_id: sessionId,
          },
        });
      } catch (error) {
        console.error("Console logs retrieval failed:", error);
        res.status(500).json({
          error: "Console logs retrieval failed",
          message: error instanceof Error ? error.message : "Unknown error",
          logs: [],
          count: 0,
        });
      }
    }
  );

  app.get("/session", async (req: express.Request, res: express.Response) => {
    try {
      const sessionId = getSessionId();

      res.json({
        session_id: sessionId,
        timestamp: new Date().toISOString(),
        active: sessionId !== null,
      });
    } catch (error) {
      console.error("Session ID retrieval failed:", error);
      res.status(500).json({
        error: "Session ID retrieval failed",
        message: error instanceof Error ? error.message : "Unknown error",
      });
    }
  });

  app.get(
    "/export-session",
    async (req: express.Request, res: express.Response) => {
      try {
        const sessionDetails = await exportSessionDetails();

        if (req.query.download === "true") {
          const sessionId = getSessionId();
          const timestamp = new Date().toISOString().replace(/:/g, "-");
          const filename = `stagehand-session-${sessionId}-${timestamp}.json`;

          res.setHeader("Content-Type", "application/json");
          res.setHeader(
            "Content-Disposition",
            `attachment; filename=${filename}`
          );
          res.json(sessionDetails);
        } else {
          res.json(sessionDetails);
        }
      } catch (error) {
        console.error("Session export failed:", error);
        res.status(500).json({
          error: "Session export failed",
          message: error instanceof Error ? error.message : "Unknown error",
        });
      }
    }
  );
}

export async function resolve_browser_url(
  browser_url: string
): Promise<string> {
  let resolvedUrl = browser_url;
  // If using hostname, resolve to IP
  if (browser_url.includes("chrome.sphinx")) {
    try {
      const { address } = await dns.lookup("chrome.sphinx");
      resolvedUrl = browser_url.replace("chrome.sphinx", address);
      console.log(`Resolved ${browser_url} to ${resolvedUrl}`);
    } catch (error) {
      console.error("DNS resolution failed:", error);
    }
  }
  return resolvedUrl;
}
