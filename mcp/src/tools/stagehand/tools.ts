import { z } from "zod";
import { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { Tool } from "../types.js";
import { parseSchema } from "../utils.js";
import {
  getOrCreateStagehand,
  sanitize,
  getConsoleLogs,
  getNetworkEntries,
} from "./core.js";
import { Stagehand, Page } from "@browserbasehq/stagehand";
import { getProvider } from "./providers.js";

// Helper to get the active page from a Stagehand instance
function getActivePage(stagehand: Stagehand): Page {
  const page = stagehand.context.activePage();
  if (!page) {
    throw new Error("No active page available");
  }
  return page;
}

// human logs + session tracking to generate a tracker
// integrate click tracker (puppeteer or playwright)
// saving steps in the graph (fixtures). storing all the agent logs
// how to feed a fixture back into a larger agentic flow
// console logs as mcp tool, and networking
// high level agentic workflow
// session id: connecting backend with frontend
// pm2 logs endpoint GET /logs

// Schemas
export const NavigateSchema = z.object({
  url: z.string().describe("The URL to navigate to"),
});

export const ActSchema = z.object({
  action: z
    .string()
    .describe(
      "The action to perform. Should be atomic and specific, i.e. 'Click the sign in button'. If unsure, use observe before using act."
    ),
  variables: z
    .record(z.string(), z.any())
    .optional()
    .describe(
      "Variables for sensitive data or dynamic content. ONLY use if needed, e.g. passwords."
    ),
});

export const ExtractSchema = z.object({});

export const ObserveSchema = z.object({
  instruction: z
    .string()
    .describe(
      "Specific observation instruction (e.g., 'find the login button')"
    ),
});

export const ScreenshotSchema = z.object({});

export const AgentSchema = z.object({
  instruction: z
    .string()
    .describe("The high-level instruction for the agent to execute"),
  provider: z
    .enum(["openai", "anthropic"])
    .optional()
    .default("anthropic")
    .describe("The provider to use for agent functionality."),
  include_screenshot: z
    .boolean()
    .optional()
    .default(false)
    .describe(
      "Whether to include a final screenshot in the agent's response (as a base64 string)."
    ),
});

export const LogsSchema = z.object({
  verbose: z.boolean().optional().default(false),
});

export const NetworkActivitySchema = z.object({
  resource_type_filter: z
    .enum(["all", "xhr", "fetch"])
    .optional()
    .default("all"),
  status_filter: z.enum(["all", "success", "failed"]).optional().default("all"),
  verbose: z.boolean().optional().default(false),
});

// Tools
export const NavigateTool: Tool = {
  name: "stagehand_navigate",
  description:
    "Navigate to a URL in the browser. Only use this tool with URLs you're confident will work and stay up to date. Otherwise use https://google.com as the starting point",
  inputSchema: parseSchema(NavigateSchema),
};

export const ActTool: Tool = {
  name: "stagehand_act",
  description:
    "Performs an action on a web page element. Act actions should be as atomic and specific as possible, i.e. 'Click the sign in button' or 'Type hello into the search input'. AVOID actions that are more than one step.",
  inputSchema: parseSchema(ActSchema),
};

export const ExtractTool: Tool = {
  name: "stagehand_extract",
  description: "Extracts all text from the current page.",
  inputSchema: parseSchema(ExtractSchema),
};

export const ObserveTool: Tool = {
  name: "stagehand_observe",
  description:
    "Observes elements on the web page. Use this to find actionable elements rather than extracting text. More often use extract instead when scraping structured text.",
  inputSchema: parseSchema(ObserveSchema),
};

export const ScreenshotTool: Tool = {
  name: "stagehand_screenshot",
  description:
    "Takes a screenshot of the current page. Use when other tools are insufficient.",
  inputSchema: parseSchema(ScreenshotSchema),
};

export const AgentTool: Tool = {
  name: "stagehand_agent",
  description:
    "Execute complex multi-step actions using AI agent. Use for larger workflows that require multiple steps.",
  inputSchema: parseSchema(AgentSchema),
};

export const LogsTool: Tool = {
  name: "stagehand_logs",
  description:
    "Read the console logs from the browser and return them. Captures all console.log, console.warn, console.error, and other console messages with timestamps and source locations.",
  inputSchema: parseSchema(LogsSchema),
};

export const NetworkActivityTool: Tool = {
  name: "stagehand_network_activity",
  description:
    "Monitor API calls during browser automation. Captures meaningful network activity with timing and metadata, filtering out resource noise. Supports filtering by request type (XHR and fetch requests) and HTTP status (success/failed).",
  inputSchema: parseSchema(NetworkActivitySchema),
};

export const TOOLS: Tool[] = [
  NavigateTool,
  ActTool,
  ExtractTool,
  ObserveTool,
  ScreenshotTool,
  AgentTool,
  LogsTool,
  NetworkActivityTool,
];

type TextResult = {
  [x: string]: unknown;
  type: "text";
  text: string;
};

type ImageResult = {
  [x: string]: unknown;
  type: "image";
  data: string;
  mimeType: string;
};

type SimpleResult = TextResult | ImageResult;

export async function call(
  name: string,
  args: Record<string, any>,
  sessionId?: string
): Promise<CallToolResult> {
  const stagehand = await getOrCreateStagehand(sessionId);

  const error = (msg: string): CallToolResult => ({
    content: [{ type: "text" as const, text: msg }],
    isError: true,
  });

  const success = (text: string, extra?: SimpleResult): CallToolResult => {
    const content: SimpleResult[] = [{ type: "text" as const, text }];
    if (extra) {
      content.push(extra as SimpleResult);
    }
    return { content, isError: false };
  };

  try {
    // Get the active page for page-level operations
    const page = getActivePage(stagehand);

    switch (name) {
      case NavigateTool.name: {
        const parsedArgs = NavigateSchema.parse(args);
        await page.goto(parsedArgs.url);
        return success(`Navigated to: ${parsedArgs.url}`);
      }

      case ActTool.name: {
        const parsedArgs = ActSchema.parse(args);
        // act() is now a method on Stagehand directly
        await stagehand.act(parsedArgs.action, {
          variables: parsedArgs.variables,
        });
        return success(`Action performed: ${parsedArgs.action}`);
      }

      case ExtractTool.name: {
        ExtractSchema.parse(args); // Validate even though no args expected
        // Use page.evaluate for raw text extraction
        const bodyText = await page.evaluate(
          () => document.body.innerText
        );
        const content = sanitize(bodyText);
        return success(`Extracted content:\n${content.join("\n")}`);
      }

      case ObserveTool.name: {
        const parsedArgs = ObserveSchema.parse(args);
        // observe() is now a method on Stagehand directly
        const observations = await stagehand.observe(parsedArgs.instruction);
        return success(`${JSON.stringify(observations)}`);
      }

      case ScreenshotTool.name: {
        ScreenshotSchema.parse(args); // Validate even though no args expected
        const buffer = await page.screenshot({ fullPage: false });
        const base64 = buffer.toString("base64");
        const screenshotName = `screenshot-${new Date()
          .toISOString()
          .replace(/:/g, "-")}`;
        return success(`Screenshot taken: ${screenshotName}`, {
          type: "image" as const,
          data: base64,
          mimeType: "image/png",
        });
      }

      case AgentTool.name: {
        const parsedArgs = AgentSchema.parse(args);
        let provider = getProvider(parsedArgs.provider);
        console.log("agent provider:", provider.computer_use_model);
        // New agent API uses 'model' field in AgentConfig, and 'mode' for CUA
        const agent = stagehand.agent({
          model: provider.computer_use_model,
          mode: "cua", // Use Computer Use Agent mode
        });
        const rez = (await agent.execute({
          instruction: parsedArgs.instruction,
          maxSteps: 25,
        })) as any;
        if (parsedArgs.include_screenshot) {
          console.log("=====> agent taking screenshot");
          const buffer = await page.screenshot({ fullPage: false });
          const base64 = buffer.toString("base64");
          rez.screenshot = {
            type: "image" as const,
            data: base64,
            mimeType: "image/png",
          };
        }
        return success(`${JSON.stringify(rez)}`);
      }

      case LogsTool.name: {
        const parsedArgs = LogsSchema.parse(args);
        const logs = getConsoleLogs(sessionId || "default-session-id");
        if (parsedArgs.verbose) {
          return success(JSON.stringify(logs, null, 2));
        } else {
          let log_str = "";
          for (const log of logs) {
            log_str += `${log.text}\n`;
          }
          return success(log_str);
        }
      }

      case NetworkActivityTool.name: {
        const parsedArgs = NetworkActivitySchema.parse(args);
        const networkEntries = getNetworkEntries(
          sessionId || "default-session-id"
        );

        // Apply resource type filtering - default to showing only xhr and fetch (API calls)
        let filteredEntries = networkEntries;
        if (parsedArgs.resource_type_filter === "all") {
          // Show both xhr and fetch requests (API calls only)
          filteredEntries = networkEntries.filter(
            (entry) =>
              entry.resourceType === "xhr" || entry.resourceType === "fetch"
          );
        } else {
          // Show specific type
          filteredEntries = networkEntries.filter(
            (entry) => entry.resourceType === parsedArgs.resource_type_filter
          );
        }

        // Apply status filtering
        if (parsedArgs.status_filter === "success") {
          filteredEntries = filteredEntries.filter(
            (entry) => entry.status && entry.status >= 200 && entry.status < 400
          );
        } else if (parsedArgs.status_filter === "failed") {
          filteredEntries = filteredEntries.filter(
            (entry) => entry.status && entry.status >= 400
          );
        }
        // 'all' status_filter requires no additional filtering

        if (parsedArgs.verbose) {
          return success(JSON.stringify(filteredEntries, null, 2));
        } else {
          // Generate comprehensive summary with all API data regardless of filters
          const allEntries = getNetworkEntries(
            sessionId || "default-session-id"
          );
          const allApiEntries = allEntries.filter(
            (entry) =>
              entry.resourceType === "xhr" || entry.resourceType === "fetch"
          );

          // Simple mode: comprehensive summary first, then filtered entries
          const response = {
            summary: {
              total_entries: allApiEntries.length,
              requests: allApiEntries.filter((e) => e.type === "request")
                .length,
              responses: allApiEntries.filter((e) => e.type === "response")
                .length,
              successful: allApiEntries.filter(
                (e) => e.status && e.status >= 200 && e.status < 400
              ).length,
              failed: allApiEntries.filter((e) => e.status && e.status >= 400)
                .length,
              xhr_requests: allApiEntries.filter(
                (e) => e.resourceType === "xhr"
              ).length,
              fetch_requests: allApiEntries.filter(
                (e) => e.resourceType === "fetch"
              ).length,
              filtered_count: filteredEntries.length,
            },
            entries: filteredEntries.map((entry) => ({
              method: entry.method,
              url: entry.url,
              type: entry.type,
              status: entry.status,
              duration: entry.duration,
              resourceType: entry.resourceType,
            })),
          };
          return success(JSON.stringify(response, null, 2));
        }
      }

      default:
        return error(`Unknown tool: ${name}`);
    }
  } catch (e) {
    return error(`Failed: ${e instanceof Error ? e.message : String(e)}`);
  }
}
