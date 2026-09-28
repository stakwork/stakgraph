import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { MCPHttpServer } from "./http.js";
import { bearerToken } from "./utils.js";
import { Express } from "express";
import * as stakgraph from "./stakgraph/index.js";
import * as stagehand from "./stagehand/tools.js";
import { getMcpTools } from "./utils.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

// The MCP SDK's Server/Protocol only supports a single active transport at
// a time -- calling connect() a second time on the same instance either
// throws "Already connected to a transport" or silently corrupts the
// previous session (see modelcontextprotocol/typescript-sdk#1405). Every
// caller here (the /graph_mcp streamable-HTTP routes via MCPHttpServer,
// and the legacy /sse route) must get its own fresh Server per session --
// never reuse/share a single instance across connections.
export function createGraphServer(): Server {
  const graphServer = new Server(
    {
      name: "Stakgraph",
      version: "0.1.0",
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  graphServer.setRequestHandler(ListToolsRequestSchema, async () => {
    return { tools: getMcpTools() };
  });

  graphServer.setRequestHandler(
    CallToolRequestSchema,
    async (request, extra) => {
      const { name, arguments: args } = request.params;
      switch (name) {
        case stakgraph.SearchTool.name: {
          const fa = stakgraph.SearchSchema.parse(args);
          return await stakgraph.search(fa);
        }
        case stakgraph.GetMapTool.name: {
          const fa = stakgraph.GetMapSchema.parse(args);
          return await stakgraph.getMap(fa);
        }
        case stakgraph.GetCodeTool.name: {
          const fa = stakgraph.GetCodeSchema.parse(args);
          return await stakgraph.getCode(fa);
        }
        case stakgraph.ShortestPathTool.name: {
          const fa = stakgraph.ShortestPathSchema.parse(args);
          return await stakgraph.shortestPath(fa);
        }
        case stakgraph.GetRulesFilesTool.name: {
          return await stakgraph.getRulesFiles();
        }
        default:
          if (name.startsWith("stagehand_")) {
            return await stagehand.call(name, args || {}, extra.sessionId);
          }
          throw new Error(`Unknown tool: ${name}`);
      }
    }
  );

  graphServer.onerror = (error) => console.error("[MCP Error]", error);
  graphServer.onclose = () => console.log("[MCP] Server connection closed");

  return graphServer;
}

// streamable http server -- a fresh Server instance is created per session
// (see createGraphServer() above)
export const server = new MCPHttpServer(createGraphServer);

// streamable http routes
export function graph_mcp_routes(app: Express) {
  app.get("/graph_mcp", bearerToken, async (req, res) => {
    await server.handleGetRequest(req, res);
  });
  app.post("/graph_mcp", bearerToken, async (req, res) => {
    await server.handlePostRequest(req, res);
  });
}

