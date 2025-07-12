import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import {
  InitializeRequestSchema,
  JSONRPCError,
} from "@modelcontextprotocol/sdk/types.js";
import { Request, Response } from "express";
import { randomUUID } from "crypto";

export class MCPServer {
  server: Server;

  // to support multiple simultaneous connections
  transports: { [sessionId: string]: StreamableHTTPServerTransport } = {};

  initializedSessions: Set<string> = new Set(); // Track which sessions are initialized

  constructor(server: Server) {
    this.server = server;
  }

  get_server(): Server {
    return this.server;
  }

  async handleGetRequest(req: Request, res: Response) {
    const mcpSessionId = req.headers["mcp-session-id"] as string | undefined;

    if (!mcpSessionId) {
      res
        .status(400)
        .json(
          this.createErrorResponse(
            "Bad Request: missing mcp-session-id header."
          )
        );
      return;
    }

    const transport = this.transports[mcpSessionId];
    if (!transport) {
      res
        .status(400)
        .json(
          this.createErrorResponse("Bad Request: no active session found.")
        );
      return;
    }

    await transport.handleRequest(req, res);
  }

  async handlePostRequest(req: Request, res: Response) {
    const mcpSessionId = req.headers["mcp-session-id"] as string | undefined;

    try {
      if (mcpSessionId) {
        // If this is an initialize request and we already have this session, clean it up first
        if (
          this.isInitializeRequest(req.body) &&
          this.transports[mcpSessionId]
        ) {
          console.log(
            `Cleaning up existing session ${mcpSessionId} for reconnection`
          );
          this.cleanupSession(mcpSessionId);
        }

        // Reuse existing transport (only if not cleaned up above)
        if (this.transports[mcpSessionId]) {
          const transport = this.transports[mcpSessionId];
          await transport.handleRequest(req, res, req.body);
          return;
        }

        // Only create new transport if this is an initialize request
        if (!this.isInitializeRequest(req.body)) {
          res
            .status(400)
            .json(
              this.createErrorResponse(
                "Bad Request: Session not found. Please initialize first."
              )
            );
          return;
        }

        // Create new transport for this session
        const transport = new StreamableHTTPServerTransport({
          sessionIdGenerator: () => mcpSessionId,
        });

        await this.server.connect(transport);
        this.transports[mcpSessionId] = transport;
        await transport.handleRequest(req, res, req.body);
        return;
      }

      // Handle no session ID case
      if (this.isInitializeRequest(req.body)) {
        const newSessionId = randomUUID();
        const transport = new StreamableHTTPServerTransport({
          sessionIdGenerator: () => newSessionId,
        });

        await this.server.connect(transport);
        await transport.handleRequest(req, res, req.body);

        this.transports[newSessionId] = transport;
        this.initializedSessions.add(newSessionId);
        return;
      }

      res
        .status(400)
        .json(
          this.createErrorResponse(
            "Bad Request: missing mcp-session-id header or invalid request."
          )
        );
    } catch (error) {
      console.error("Error handling MCP request:", error);

      if (mcpSessionId) {
        this.cleanupSession(mcpSessionId);
      }

      res.status(500).json(this.createErrorResponse("Internal server error."));
    }
  }

  cleanupSession(mcpSessionId: string) {
    if (this.transports[mcpSessionId]) {
      const transport = this.transports[mcpSessionId];
      transport.close();
      delete this.transports[mcpSessionId];
    }
    this.initializedSessions.delete(mcpSessionId);
  }

  private createErrorResponse(message: string): JSONRPCError {
    return {
      jsonrpc: "2.0",
      error: {
        code: -32000,
        message: message,
      },
      id: randomUUID(),
    };
  }

  private isInitializeRequest(body: any): boolean {
    const isInitial = (data: any) => {
      const result = InitializeRequestSchema.safeParse(data);
      return result.success;
    };
    if (Array.isArray(body)) {
      return body.some((request) => isInitial(request));
    }
    return isInitial(body);
  }
}
