import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import { bearerToken, mcpSession } from "./utils.js";
import { Express } from "express";
import { Tool, Json } from "./types.js";
import { createGraphServer } from "./server.js";
import { getMcpTools } from "./utils.js";

export function graph_sse_routes(app: Express) {
  // Keyed by the SSEServerTransport's own generated sessionId. A single
  // shared transport/server (as this used to be) breaks as soon as there
  // is more than one concurrent client: every /sse connection would (a)
  // throw/corrupt the previous session on the shared Server instance (see
  // createGraphServer()'s doc comment), and (b) /messages would always be
  // routed to whichever client connected *last*, silently misdirecting
  // every other client's messages. Track one transport per session instead.
  const sessions = new Map<string, SSEServerTransport>();

  app.get("/sse", bearerToken, mcpSession, async (req, res) => {
    try {
      const transport = new SSEServerTransport("/messages", res);
      const graphServer = createGraphServer();
      await graphServer.connect(transport);

      sessions.set(transport.sessionId, transport);

      const cleanup = () => {
        sessions.delete(transport.sessionId);
      };
      res.on("close", cleanup);
      res.on("error", cleanup);
    } catch (error) {
      if (!res.headersSent) {
        res.status(500).send("Connection failed");
      }
    }
  });

  // Raw route without any body parsing middleware
  app.post("/messages", bearerToken, mcpSession, async (req, res) => {
    try {
      const sessionId = req.query.sessionId as string | undefined;
      const transport = sessionId ? sessions.get(sessionId) : undefined;
      if (transport) {
        await transport.handlePostMessage(req, res);
      } else {
        res.status(400).json({ error: "No active transport" });
      }
    } catch (error) {
      if (!res.headersSent) {
        res
          .status(500)
          .json({ error: "Message handling failed", details: error });
      }
    }
  });

  app.get("/tools", bearerToken, (_, res) => {
    const obj: { tools: HttpTool[]; headers?: { [k: string]: any } } = {
      tools: getMcpTools().map(fmtToolForHttp),
    };
    if (process.env.API_TOKEN) {
      obj.headers = {
        Authorization: "******",
      };
    }
    res.send(obj);
  });
}

// not for mcp, for http tool schema
export interface HttpTool {
  name: string;
  description: string;
  input_schema: Json;
}

function fmtToolForHttp(tool: Tool): HttpTool {
  return {
    name: tool.name,
    description: tool.description,
    input_schema: tool.inputSchema,
  };
}
