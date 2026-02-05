import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { MCPServer } from "./http.js";
import { bearerToken } from "./utils.js";
import { Express } from "express";
import * as stakgraph from "./stakgraph/index.js";
import * as stagehand from "./stagehand/tools.js";
import { getMcpTools } from "./utils.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

// streamable http server
export const server = new MCPServer(
  new Server(
    {
      name: "Stakgraph",
      version: "0.1.0",
    },
    {
      capabilities: {
        tools: {},
      },
    }
  )
);

// streamable http routes
export function graph_mcp_routes(app: Express) {
  app.get("/graph_mcp", bearerToken, async (req, res) => {
    await server.handleGetRequest(req, res);
  });
  app.post("/graph_mcp", bearerToken, async (req, res) => {
    await server.handlePostRequest(req, res);
  });
}

server.get_server().setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools: getMcpTools() };
});

server
  .get_server()
  .setRequestHandler(CallToolRequestSchema, async (request, extra) => {
    const { name, arguments: args } = request.params;
    // console.log("===> call tool", JSON.stringify(extra, null, 2));
    switch (name) {
      case stakgraph.SearchTool.name: {
        const fa = stakgraph.SearchSchema.parse(args);
        return await stakgraph.search(fa);
      }
      case stakgraph.GetNodesTool.name: {
        const fa = stakgraph.GetNodesSchema.parse(args);
        return await stakgraph.getNodes(fa);
      }
      case stakgraph.GetEdgesTool.name: {
        const fa = stakgraph.GetEdgesSchema.parse(args);
        return await stakgraph.getEdges(fa);
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
      case stakgraph.RepoMapTool.name: {
        const fa = stakgraph.RepoMapSchema.parse(args);
        return await stakgraph.repoMap(fa);
      }
      case stakgraph.GetRulesFilesTool.name: {
        return await stakgraph.getRulesFiles();
      }
      case stakgraph.ExploreTool.name: {
        const fa = stakgraph.ExploreSchema.parse(args);
        return await stakgraph.explore(fa);
      }
      default:
        if (name.startsWith("stagehand_")) {
          return await stagehand.call(name, args || {}, extra.sessionId);
        }
        throw new Error(`Unknown tool: ${name}`);
    }
  });

// Handle server lifecycle
server.get_server().onerror = (error) => console.error("[MCP Error]", error);
server.get_server().onclose = () =>
  console.log("[MCP] Server connection closed");
