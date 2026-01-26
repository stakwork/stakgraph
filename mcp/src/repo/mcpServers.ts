import { createMCPClient } from "@ai-sdk/mcp";
import { Tool } from "ai";

export interface McpServer {
  name: string;
  url: string;
  token?: string; // Shorthand for Authorization: Bearer <token>
  headers?: Record<string, string>; // Full headers for custom auth
  toolFilter?: string[]; // Only include these tools (if empty/undefined, include all)
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
          allTools[prefixedName] = tool as Tool<any, any>;
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