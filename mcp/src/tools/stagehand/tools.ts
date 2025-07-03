import { z } from "zod";
import { CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import { Tool } from "../index.js";
import { parseSchema } from "../utils.js";
import { getOrCreateStagehand, sanitize, getConsoleLogs } from "./utils.js";
import { AgentProviderType } from "@browserbasehq/stagehand";
import { getProvider } from "./providers.js";

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
    .record(z.any())
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
    .default("openai")
    .describe("The provider to use for agent functionality."),
  include_screenshot: z
    .boolean()
    .optional()
    .default(false)
    .describe(
      "Whether to include a final screenshot in the agent's response (as a base64 string)."
    ),
});

export const LogsSchema = z.object({});

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

export const TOOLS: Tool[] = [
  NavigateTool,
  ActTool,
  ExtractTool,
  ObserveTool,
  ScreenshotTool,
  AgentTool,
  LogsTool,
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
  args: Record<string, any>
): Promise<CallToolResult> {
  const stagehand = await getOrCreateStagehand();

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
    switch (name) {
      case NavigateTool.name: {
        const parsedArgs = NavigateSchema.parse(args);
        await stagehand.page.goto(parsedArgs.url);
        return success(`Navigated to: ${parsedArgs.url}`);
      }

      case ActTool.name: {
        const parsedArgs = ActSchema.parse(args);
        await stagehand.page.act({
          action: parsedArgs.action,
          variables: parsedArgs.variables,
        });
        return success(`Action performed: ${parsedArgs.action}`);
      }

      case ExtractTool.name: {
        ExtractSchema.parse(args); // Validate even though no args expected
        const bodyText = await stagehand.page.evaluate(
          () => document.body.innerText
        );
        const content = sanitize(bodyText);
        return success(`Extracted content:\n${content.join("\n")}`);
      }

      case ObserveTool.name: {
        const parsedArgs = ObserveSchema.parse(args);
        const observations = await stagehand.page.observe({
          instruction: parsedArgs.instruction,
          returnAction: false,
        });
        return success(`${JSON.stringify(observations)}`);
      }

      case ScreenshotTool.name: {
        ScreenshotSchema.parse(args); // Validate even though no args expected
        const buffer = await stagehand.page.screenshot({ fullPage: false });
        const base64 = buffer.toString("base64");
        const name = `screenshot-${new Date()
          .toISOString()
          .replace(/:/g, "-")}`;
        return success(`Screenshot taken: ${name}`, {
          type: "image" as const,
          data: base64,
          mimeType: "image/png",
        });
      }

      case AgentTool.name: {
        const parsedArgs = AgentSchema.parse(args);
        let provider = getProvider(parsedArgs.provider);
        console.log("agent provider:", provider.computer_use_model);
        const agent = stagehand.agent({
          provider: provider.name as AgentProviderType,
          model: provider.computer_use_model,
        });
        const rez = (await agent.execute({
          instruction: parsedArgs.instruction,
          maxSteps: 25,
        })) as any;
        if (parsedArgs.include_screenshot) {
          console.log("=====> agent taking screenshot");
          const buffer = await stagehand.page.screenshot({ fullPage: false });
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
        LogsSchema.parse(args); // Validate even though no args expected
        const logs = getConsoleLogs();
        return success(JSON.stringify(logs, null, 2));
      }

      default:
        return error(`Unknown tool: ${name}`);
    }
  } catch (e) {
    return error(`Failed: ${e instanceof Error ? e.message : String(e)}`);
  }
}
