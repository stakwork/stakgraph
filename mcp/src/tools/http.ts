import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { InitializeRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { Request, Response } from "express";
import { randomUUID } from "crypto";

export interface MCPServerLike {
  connect(transport: StreamableHTTPServerTransport): Promise<void>;
}

interface SessionEntry {
  server: MCPServerLike;
  transport: StreamableHTTPServerTransport;
  lastActivity: Date;
}

export class MCPHttpServer {
  private serverFactory: () => MCPServerLike;
  private sessions: Map<string, SessionEntry> = new Map();
  private cleanupInterval: ReturnType<typeof setInterval>;

  constructor(serverFactory: () => MCPServerLike) {
    this.serverFactory = serverFactory;
    this.cleanupInterval = setInterval(() => {
      this.cleanupStaleSessions();
    }, 60 * 1000);
  }

  private cleanupStaleSessions() {
    const now = Date.now();
    const staleThreshold = 5 * 60 * 1000;
    for (const [sessionId, session] of this.sessions) {
      const idle = now - session.lastActivity.getTime();
      if (idle > staleThreshold) {
        console.log(`Cleaning up stale session ${sessionId} (idle: ${idle}ms)`);
        this.cleanupSession(sessionId);
      }
    }
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

    const session = this.sessions.get(mcpSessionId);
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
        if (
          this.isInitializeRequest(req.body) &&
          this.sessions.has(mcpSessionId)
        ) {
          console.log(
            `Cleaning up existing session ${mcpSessionId} for reconnection`
          );
          this.cleanupSession(mcpSessionId);
        }

        const existing = this.sessions.get(mcpSessionId);
        if (existing) {
          existing.lastActivity = new Date();
          await existing.transport.handleRequest(req, res, req.body);
          return;
        }

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

        await this.createSession(mcpSessionId, req, res);
        return;
      }

      if (this.isInitializeRequest(req.body)) {
        await this.createSession(randomUUID(), req, res);
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

  private async createSession(sessionId: string, req: Request, res: Response) {
    const server = this.serverFactory();
    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: () => sessionId,
    });

    await server.connect(transport);
    this.sessions.set(sessionId, {
      server,
      transport,
      lastActivity: new Date(),
    });
    await transport.handleRequest(req, res, req.body);
  }

  cleanupSession(sessionId: string) {
    const session = this.sessions.get(sessionId);
    if (session) {
      try {
        session.transport.close();
      } catch (_) {
        // Ignore close errors
      }
      this.sessions.delete(sessionId);
    }
  }

  private createErrorResponse(message: string) {
    return {
      jsonrpc: "2.0",
      error: {
        code: -32000,
        message,
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
