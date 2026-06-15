import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import {
  InitializeRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { Request, Response } from "express";
import { randomUUID } from "crypto";

interface SessionEntry {
  transport: StreamableHTTPServerTransport;
  lastActivity: Date;
}

// streamable http MCP server
// mcp-session-id can be reused, even after reconnect later
export class MCPServer {
  server: Server;
  sessions: { [sessionId: string]: SessionEntry } = {};
  private cleanupInterval: ReturnType<typeof setInterval>;

  constructor(server: Server) {
    this.server = server;
    this.cleanupInterval = setInterval(() => {
      this.cleanupStaleSessions();
    }, 60 * 1000);
  }

  private cleanupStaleSessions() {
    const now = Date.now();
    const staleThreshold = 5 * 60 * 1000;
    for (const sessionId of Object.keys(this.sessions)) {
      const age = now - this.sessions[sessionId].lastActivity.getTime();
      if (age > staleThreshold) {
        console.log(`Cleaning up stale session ${sessionId} (idle: ${age}ms)`);
        this.cleanupSession(sessionId);
      }
    }
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

    const session = this.sessions[mcpSessionId];
    if (!session) {
      res
        .status(400)
        .json(
          this.createErrorResponse("Bad Request: no active session found.")
        );
      return;
    }

    session.lastActivity = new Date();
    await session.transport.handleRequest(req, res);
  }

  async handlePostRequest(req: Request, res: Response) {
    const mcpSessionId = req.headers["mcp-session-id"] as string | undefined;

    try {
      if (mcpSessionId) {
        // If this is an initialize request and we already have this session, clean it up first
        if (
          this.isInitializeRequest(req.body) &&
          this.sessions[mcpSessionId]
        ) {
          console.log(
            `Cleaning up existing session ${mcpSessionId} for reconnection`
          );
          this.cleanupSession(mcpSessionId);
        }

        // Reuse existing session (only if not cleaned up above)
        if (this.sessions[mcpSessionId]) {
          const session = this.sessions[mcpSessionId];
          session.lastActivity = new Date();
          await session.transport.handleRequest(req, res, req.body);
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
        this.sessions[mcpSessionId] = { transport, lastActivity: new Date() };
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

        this.sessions[newSessionId] = { transport, lastActivity: new Date() };
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
    const session = this.sessions[mcpSessionId];
    if (session) {
      try {
        session.transport.close();
      } catch (_) {
        // Ignore close errors
      }
      delete this.sessions[mcpSessionId];
    }
  }

  private createErrorResponse(message: string) {
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
