import { createMCPClient } from "@ai-sdk/mcp";
import type { Tool } from "ai";
import type { LanguageModelV2ToolResultOutput } from "@ai-sdk/provider";

export interface McpServer {
  name: string;
  url: string;
  token?: string; // Shorthand for Authorization: Bearer <token>
  headers?: Record<string, string>; // Full headers for custom auth
  toolFilter?: string[]; // Only include these tools (if empty/undefined, include all)
}

// Safe wrapper for toModelOutput that handles undefined output
// This fixes a bug in @ai-sdk/mcp where mcpToModelOutput crashes on undefined
function createSafeToModelOutput(
  originalToModelOutput?: (params: {
    toolCallId: string;
    input: unknown;
    output: unknown;
  }) => LanguageModelV2ToolResultOutput
) {
  return (params: {
    toolCallId: string;
    input: unknown;
    output: unknown;
  }): LanguageModelV2ToolResultOutput => {
    const { output } = params;

    // Handle undefined/null output before passing to original
    if (output === undefined || output === null) {
      return {
        type: "text",
        value: "Error: Tool returned undefined result",
      };
    }

    // If no original toModelOutput, handle the MCP format ourselves
    if (!originalToModelOutput) {
      const result = output as { content?: Array<{ type: string; text?: string }>; isError?: boolean };
      if (result.content && Array.isArray(result.content)) {
        const textParts = result.content
          .filter((c) => c.type === "text" && c.text)
          .map((c) => c.text);
        return { type: "text", value: textParts.join("\n") };
      }
      return { type: "json", value: output as any };
    }

    return originalToModelOutput(params);
  };
}

export async function getMcpTools(
  mcpServers: McpServer[]
): Promise<Record<string, Tool<any, any>>> {
  const allTools: Record<string, Tool<any, any>> = {};

  for (const server of mcpServers) {
    try {
      // Build headers: token takes precedence, then headers
      const headers: Record<string, string> = server.headers || {};
      if (server.token) {
        headers.Authorization = `Bearer ${server.token}`;
      }

      const mcpClient = await createMCPClient({
        transport: {
          type: "http",
          url: server.url,
          headers,
        },
      });

      const tools = await mcpClient.tools();

      // Filter tools if toolFilter is specified and non-empty
      for (const [toolName, tool] of Object.entries(tools)) {
        const shouldInclude =
          !server.toolFilter ||
          server.toolFilter.length === 0 ||
          server.toolFilter.includes(toolName);

        if (shouldInclude) {
          // Prefix with server name to avoid collisions
          const prefixedName = `${server.name}_${toolName}`;

          // Wrap the tool with safe toModelOutput handling
          const wrappedTool = {
            ...tool,
            toModelOutput: createSafeToModelOutput(
              (tool as any).toModelOutput
            ),
          };

          allTools[prefixedName] = wrappedTool as Tool<any, any>;
        }
      }

      console.log(
        `[MCP] Loaded ${Object.keys(tools).length} tools from ${server.name}, included ${
          server.toolFilter?.length || Object.keys(tools).length
        }`
      );
    } catch (error) {
      console.error(`[MCP] Failed to connect to ${server.name}:`, error);
    }
  }

  return allTools;
}