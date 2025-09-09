import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import { bearerToken, mcpSession } from "./utils.js";
import { Express } from "express";
import { Tool, Json } from "./types.js";
import { server } from "./server.js";
import { getMcpTools } from "./utils.js";

export function sse_routes(app: Express) {
  let currentTransport: SSEServerTransport | null = null;

  app.get("/sse", bearerToken, mcpSession, async (req, res) => {
    try {
      currentTransport = new SSEServerTransport("/messages", res);
      await server.get_server().connect(currentTransport);
      res.on("close", () => {
        currentTransport = null;
      });
      res.on("error", (error) => {
        currentTransport = null;
      });
    } catch (error) {
      currentTransport = null;
      if (!res.headersSent) {
        res.status(500).send("Connection failed");
      }
    }
  });

  // Raw route without any body parsing middleware
  app.post("/messages", bearerToken, mcpSession, async (req, res) => {
    console.log("===> messages - handling POST");
    try {
      if (currentTransport) {
        console.log("===> messages - sessionId", (req as any).sessionId);
        await currentTransport.handlePostMessage(req, res);
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
        Authorization: "Bearer YOUR_TOKEN",
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
