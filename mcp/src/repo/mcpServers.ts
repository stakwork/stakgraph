import { createMCPClient } from "@ai-sdk/mcp";
import { tool, Tool } from "ai";

export interface McpServer {
  name: string;
  url: string;
  token?: string; // Shorthand for Authorization: Bearer <token>
  headers?: Record<string, string>; // Full headers for custom auth
  toolFilter?: string[]; // Only include these tools (if empty/undefined, include all)
}

/**
 * Wrap an MCP tool to ensure it always returns a valid result.
 * The @ai-sdk/mcp adapter crashes with "Cannot use 'in' operator to search for 'content' in undefined"
 * if an MCP tool returns undefined or an invalid response.
 * 
 * By wrapping with tool(), we bypass the MCP adapter's toModelOutput conversion
 * and handle the execution ourselves as a regular AI SDK tool.
 */
function wrapMcpTool(
  mcpTool: Tool<any, any>,
  toolName: string
): Tool<any, any> {
  // MCP tools from @ai-sdk/mcp have parameters as a JSON schema object
  const schema = (mcpTool as any).inputSchema || (mcpTool as any).parameters;
  
  console.log(`[MCP] Wrapping tool ${toolName}, schema:`, JSON.stringify(schema, null, 2));
  
  return tool({
    description: mcpTool.description || `MCP tool: ${toolName}`,
    inputSchema: schema,
    execute: async (args: any) => {
      console.log(`[MCP] Executing ${toolName} with args:`, JSON.stringify(args));
      try {
        const result = await (mcpTool as any).execute(args);
        console.log(`[MCP] Tool ${toolName} result:`, typeof result, result === undefined ? 'undefined' : 'has value');
        // Ensure we always return something valid
        if (result === undefined || result === null) {
          console.warn(`[MCP] Tool ${toolName} returned undefined/null`);
          return { error: `Tool ${toolName} returned no result` };
        }
        return result;
      } catch (error: any) {
        console.error(`[MCP] Tool ${toolName} execution failed:`, error);
        return { error: error.message || "Tool execution failed" };
      }
    },
  });
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
      for (const [toolName, mcpTool] of Object.entries(tools)) {
        const shouldInclude =
          !server.toolFilter ||
          server.toolFilter.length === 0 ||
          server.toolFilter.includes(toolName);

        if (shouldInclude) {
          // Prefix with server name to avoid collisions
          const prefixedName = `${server.name}_${toolName}`;
          // Wrap the MCP tool to handle undefined results
          allTools[prefixedName] = wrapMcpTool(
            mcpTool as Tool<any, any>,
            prefixedName
          );
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