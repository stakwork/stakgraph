import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { MCPHttpServer } from "./http.js";
import { bearerToken } from "./utils.js";
import { Express } from "express";
import * as stakgraph from "./stakgraph/index.js";
import * as stagehand from "./stagehand/tools.js";
import * as verify from "./verify/index.js";
import { getMcpTools, use_stagehand } from "./utils.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

export const graphServer = new Server(
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

// streamable http server (shared server instance across sessions)
export const server = new MCPHttpServer(() => graphServer);

// streamable http routes
export function graph_mcp_routes(app: Express) {
  app.get("/graph_mcp", bearerToken, async (req, res) => {
    await server.handleGetRequest(req, res);
  });
  app.post("/graph_mcp", bearerToken, async (req, res) => {
    await server.handlePostRequest(req, res);
  });
}

graphServer.setRequestHandler(ListToolsRequestSchema, async () => {
  const tools = getMcpTools();
  return { tools: use_stagehand() ? [...tools, ...verify.VERIFY_TOOLS] : tools };
});

graphServer.setRequestHandler(CallToolRequestSchema, async (request, extra) => {
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
    case verify.HttpRequestTool.name: {
      const sid = extra.sessionId || "default-session-id";
      return await verify.httpRequest(sid, args || {});
    }
    case verify.SampleTool.name: {
      const sid = extra.sessionId || "default-session-id";
      return await verify.sampleUrl(sid, args || {});
    }
    case verify.DbQueryTool.name: {
      const sid = extra.sessionId || "default-session-id";
      return await verify.dbQuery(sid, args || {});
    }
    case verify.SubmitVerdictTool.name: {
      const sid = extra.sessionId || "default-session-id";
      return await verify.submitVerdict(sid, args || {});
    }
    default:
      if (name.startsWith("stagehand_")) {
        const sid = extra.sessionId || "default-session-id";
        const result = await stagehand.call(name, args || {}, extra.sessionId);
        return verify.tagEvidence(sid, name, result);
      }
      throw new Error(`Unknown tool: ${name}`);
  }
});

// Handle server lifecycle
graphServer.onerror = (error) => console.error("[MCP Error]", error);
graphServer.onclose = () => console.log("[MCP] Server connection closed");
