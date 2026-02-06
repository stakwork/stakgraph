import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { InitializeRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { Request, Response } from "express";
import { randomUUID } from "crypto";

interface Session {
  server: McpServer;
  transport: StreamableHTTPServerTransport;
  createdAt: Date;
}

// Streamable HTTP MCP server
// Creates a new McpServer per session (SDK requires one transport per server)
export class MCPHttpServer {
  private serverFactory: () => McpServer;
  private sessions: Map<string, Session> = new Map();
  private cleanupInterval: NodeJS.Timeout;

  constructor(serverFactory: () => McpServer) {
    this.serverFactory = serverFactory;
    
    // Clean up stale sessions every 60 seconds
    this.cleanupInterval = setInterval(() => {
      this.cleanupStaleSessions();
    }, 60 * 1000);
  }

  private cleanupStaleSessions() {
    const now = Date.now();
    const staleThreshold = 5 * 60 * 1000; // 5 minutes

    for (const [sessionId, session] of this.sessions) {
      const age = now - session.createdAt.getTime();
      if (age > staleThreshold) {
        console.log(`Cleaning up stale session ${sessionId} (age: ${age}ms)`);
        this.cleanupSession(sessionId);
      }
    }
  }

  async handleGetRequest(req: Request, res: Response) {
    const sessionId = req.headers["mcp-session-id"] as string | undefined;

    if (!sessionId) {
      res.status(400).json(
        this.createErrorResponse("Bad Request: missing mcp-session-id header.")
      );
      return;
    }

    const session = this.sessions.get(sessionId);
    if (!session) {
      res.status(400).json(
        this.createErrorResponse("Bad Request: no active session found.")
      );
      return;
    }

    await session.transport.handleRequest(req, res);
  }

  async handlePostRequest(req: Request, res: Response) {
    const sessionId = req.headers["mcp-session-id"] as string | undefined;

    try {
      if (sessionId) {
        // If this is an initialize request and we already have this session, clean it up first
        if (this.isInitializeRequest(req.body) && this.sessions.has(sessionId)) {
          console.log(`Cleaning up existing session ${sessionId} for reconnection`);
          this.cleanupSession(sessionId);
        }

        // Reuse existing session
        const existingSession = this.sessions.get(sessionId);
        if (existingSession) {
          // Update timestamp on activity
          existingSession.createdAt = new Date();
          await existingSession.transport.handleRequest(req, res, req.body);
          return;
        }

        // Only create new session if this is an initialize request
        if (!this.isInitializeRequest(req.body)) {
          res.status(400).json(
            this.createErrorResponse("Bad Request: Session not found. Please initialize first.")
          );
          return;
        }

        // Create new session with provided ID
        await this.createSession(sessionId, req, res);
        return;
      }

      // Handle no session ID case - create new session with generated ID
      if (this.isInitializeRequest(req.body)) {
        const newSessionId = randomUUID();
        await this.createSession(newSessionId, req, res);
        return;
      }

      res.status(400).json(
        this.createErrorResponse("Bad Request: missing mcp-session-id header or invalid request.")
      );
    } catch (error) {
      console.error("Error handling MCP request:", error);
      if (sessionId) {
        this.cleanupSession(sessionId);
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
      createdAt: new Date() 
    });
    await transport.handleRequest(req, res, req.body);
  }

  cleanupSession(sessionId: string) {
    const session = this.sessions.get(sessionId);
    if (session) {
      try {
        session.transport.close();
      } catch (e) {
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
