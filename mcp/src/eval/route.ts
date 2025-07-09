import express, { Express } from "express";
import { evaluate } from "./stagehand.js";
import { promises as dns } from "dns";
import { getConsoleLogs, startAction, getActionLogs } from "../tools/stagehand/utils.js";
import { randomUUID } from "crypto";

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

      // Generate unique action ID for this evaluation
      const actionId = `eval_${randomUUID()}_${Date.now()}`;
      startAction(actionId);

      const result = await evaluate(test_url, prompt);
      
      // Add action_id to response
      res.json({
        ...result,
        action_id: actionId
      });
    } catch (error) {
      console.error("Evaluation failed:", error);
      res.status(500).json({
        error: "Evaluation failed",
        message: error instanceof Error ? error.message : "Unknown error"
      });
    }
  });

  // Simple HTTP endpoint for console logs (Phase 2A: Dual Access Pattern)
  // Future Phase 2B: Add GET /console-logs/stream for real-time SSE streaming
  // - Streams logs immediately as they're captured vs current batch approach
  // - Client: const stream = new EventSource('/console-logs/stream')
  // - Perfect for live monitoring during automation runs
  // - Implementation: Modify addConsoleLog() to broadcast to streaming clients
  app.get("/console-logs", async (req: express.Request, res: express.Response) => {
    try {
      const actionId = req.query.action_id as string;
      let logs: any[];
      let accessMethod: string;
      
      if (actionId) {
        // Get action-specific logs
        logs = getActionLogs(actionId);
        accessMethod = "http_rest_action_specific";
      } else {
        // Fall back to global logs (backward compatibility)
        logs = getConsoleLogs();
        accessMethod = "http_rest_global";
      }

      const response: any = {
        logs,
        timestamp: new Date().toISOString(),
        count: logs.length,
        metadata: {
          session_active: true,
          access_method: accessMethod
        }
      };

      // Include action_id in response if provided
      if (actionId) {
        response.action_id = actionId;
      }

      res.json(response);
    } catch (error) {
      console.error("Console logs retrieval failed:", error);
      res.status(500).json({
        error: "Console logs retrieval failed",
        message: error instanceof Error ? error.message : "Unknown error",
        logs: [],
        count: 0
      });
    }
  });
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
