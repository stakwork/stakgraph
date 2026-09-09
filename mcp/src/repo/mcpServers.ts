import { createMCPClient } from "@ai-sdk/mcp";
import { Experimental_StdioMCPTransport } from "@ai-sdk/mcp/mcp-stdio";
import type { Tool } from "ai";
import type { LanguageModelV2ToolResultOutput } from "@ai-sdk/provider";

// Discriminated union: HTTP server vs. stdio (local process) server
export type McpServer =
  | {
      name: string;
      url: string;
      token?: string; // Shorthand for Authorization: Bearer <token>
      headers?: Record<string, string>; // Full headers for custom auth
      toolFilter?: string[]; // Only include these tools (if empty/undefined, include all)
    }
  | {
      name: string;
      command: string;
      args?: string[];
      env?: Record<string, string>; // Secrets go here — stripped before persisting to sidecar
      toolFilter?: string[]; // Only include these tools (if empty/undefined, include all)
    };

// Built-in default tool filters keyed on server.name.
// Returning undefined means "no filter = include all" (unchanged legacy behavior).
// NOTE: convert_to_pdf is deliberately excluded — it requires LibreOffice/soffice
// which is NOT installed in the MCP Docker image to keep the image lean.
const DEFAULT_TOOL_FILTERS: Record<string, string[]> = {
  docx: [
    "open_document",
    "create_document",
    "save_document",
    "close_document",
    "get_headings",
    "search_text",
    "insert_text",
    "delete_text",
    "replace_text",
    "add_comment",
    "add_footnote",
    "audit_document",
    "generate_change_summary",
    "diff_to_text",
  ],
};

function defaultFilterFor(server: McpServer): string[] | undefined {
  return DEFAULT_TOOL_FILTERS[server.name];
}

function maskCredential(cred?: string): string {
  if (!cred) return "(no credential)";
  if (cred.length <= 4) return "*".repeat(cred.length);
  return `${"*".repeat(cred.length - 4)}${cred.slice(-4)}`;
}

// Safe wrapper for toModelOutput that handles undefined output
// This fixes a bug in @ai-sdk/mcp where mcpToModelOutput crashes on undefined
function createSafeToModelOutput(
  serverName: string,
  toolName: string,
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
    const { toolCallId, input, output } = params;

    // Debug logging to see what the agent receives
    console.log(`[MCP Debug] ========================================`);
    console.log(`[MCP Debug] Server: ${serverName}, Tool: ${toolName}`);
    console.log(`[MCP Debug] toolCallId: ${toolCallId}`);
    console.log(`[MCP Debug] input:`, JSON.stringify(input, null, 2));
    console.log(`[MCP Debug] output type: ${typeof output}`);
    console.log(`[MCP Debug] output:`, JSON.stringify(output, null, 2));
    console.log(`[MCP Debug] ========================================`);

    // Handle undefined/null output before passing to original
    if (output === undefined || output === null) {
      console.error(
        `[MCP Debug] ERROR: Tool ${serverName}_${toolName} returned undefined/null`
      );
      console.error(`[MCP Debug] Input was:`, JSON.stringify(input, null, 2));
      return {
        type: "text",
        value: `Error: Tool ${serverName}_${toolName} returned undefined result. Input was: ${JSON.stringify(input)}`,
      };
    }

    // If no original toModelOutput, handle the MCP format ourselves
    if (!originalToModelOutput) {
      const result = output as {
        content?: Array<{ type: string; text?: string }>;
        isError?: boolean;
      };
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

// Timeout (ms) for stdio server connect + initial tools() handshake.
// Prevents a process that spawns but never completes the MCP handshake
// from blocking the agent run indefinitely.
const STDIO_CONNECT_TIMEOUT_MS = 15_000;

export interface McpToolsResult {
  tools: Record<string, Tool<any, any>>;
  clients: Array<{ close: () => Promise<void> }>;
}

export async function getMcpTools(
  mcpServers: McpServer[],
  repoPath?: string
): Promise<McpToolsResult> {
  const allTools: Record<string, Tool<any, any>> = {};
  const allClients: Array<{ close: () => Promise<void> }> = [];

  for (const server of mcpServers) {
    // Defensive guard: reject ambiguous entries that specify both url and command
    if ("url" in server && "command" in server) {
      console.error(
        `[MCP] Skipping server "${server.name}": entry specifies both "url" and "command" — use one or the other`
      );
      continue;
    }

    // Compute effective tool filter once: caller-supplied takes precedence over
    // built-in default; undefined means "no filter = all" (unchanged behavior).
    const effectiveFilter = server.toolFilter ?? defaultFilterFor(server);

    try {
      let mcpClient: Awaited<ReturnType<typeof createMCPClient>>;

      if ("command" in server) {
        // ── Stdio branch (new) ──────────────────────────────────────────────
        // Log command only — never args/env (they may carry secrets/paths).
        console.log(
          `[MCP] Connecting to ${server.name} via command ${server.command}`
        );

        const transport = new Experimental_StdioMCPTransport({
          command: server.command,
          args: server.args,
          env: server.env,
          // Set cwd so relative file paths in tool calls resolve to the repo dir.
          // StdioConfig supports cwd natively in @ai-sdk/mcp@1.0.18.
          ...(repoPath ? { cwd: repoPath } : {}),
        });

        // Wrap connect + initial tools() in a timeout to guard against a process
        // that spawns but never completes the MCP handshake.
        const connectWithTimeout = async () => {
          const timeoutPromise = new Promise<never>((_, reject) =>
            setTimeout(
              () => reject(new Error(`Stdio connect timeout after ${STDIO_CONNECT_TIMEOUT_MS}ms`)),
              STDIO_CONNECT_TIMEOUT_MS
            )
          );
          const connectPromise = (async () => {
            const client = await createMCPClient({ transport });
            const tools = await client.tools();
            return { client, tools };
          })();
          return Promise.race([connectPromise, timeoutPromise]);
        };

        const { client, tools } = await connectWithTimeout();
        mcpClient = client;
        allClients.push(mcpClient);

        for (const [toolName, tool] of Object.entries(tools)) {
          const shouldInclude =
            !effectiveFilter ||
            effectiveFilter.length === 0 ||
            effectiveFilter.includes(toolName);

          if (shouldInclude) {
            const prefixedName = `${server.name}_${toolName}`;
            const wrappedTool = {
              ...tool,
              toModelOutput: createSafeToModelOutput(
                server.name,
                toolName,
                (tool as any).toModelOutput
              ),
            };
            allTools[prefixedName] = wrappedTool as Tool<any, any>;
          }
        }

        const includedCount = effectiveFilter
          ? Object.keys(tools).filter((n) => effectiveFilter.includes(n)).length
          : Object.keys(tools).length;
        console.log(
          `[MCP] Loaded ${Object.keys(tools).length} tools from ${server.name}, included ${includedCount}`
        );
      } else {
        // ── HTTP branch (existing behavior) ─────────────────────────────────
        const headers: Record<string, string> = { ...(server.headers || {}) };
        if (server.token) {
          headers.Authorization = `Bearer ${server.token}`;
        }

        const credential =
          server.token ||
          (headers.Authorization || headers["authorization"] || "").replace(
            /^Bearer\s+/i,
            ""
          ) ||
          undefined;
        console.log(
          `[MCP] Connecting to ${server.name} at ${server.url} (credential: ${maskCredential(credential)})`
        );

        mcpClient = await createMCPClient({
          transport: {
            type: "http",
            url: server.url,
            headers,
          },
        });
        allClients.push(mcpClient);

        const tools = await mcpClient.tools();

        for (const [toolName, tool] of Object.entries(tools)) {
          const shouldInclude =
            !effectiveFilter ||
            effectiveFilter.length === 0 ||
            effectiveFilter.includes(toolName);

          if (shouldInclude) {
            const prefixedName = `${server.name}_${toolName}`;
            const wrappedTool = {
              ...tool,
              toModelOutput: createSafeToModelOutput(
                server.name,
                toolName,
                (tool as any).toModelOutput
              ),
            };
            allTools[prefixedName] = wrappedTool as Tool<any, any>;
          }
        }

        const includedCount = effectiveFilter
          ? Object.keys(tools).filter((n) => effectiveFilter.includes(n)).length
          : Object.keys(tools).length;
        console.log(
          `[MCP] Loaded ${Object.keys(tools).length} tools from ${server.name}, included ${includedCount}`
        );
      }
    } catch (error) {
      console.error(`[MCP] Failed to connect to ${server.name}:`, error);
    }
  }

  return { tools: allTools, clients: allClients };
}
