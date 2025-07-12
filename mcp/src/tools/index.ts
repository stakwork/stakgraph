import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import * as search from "./search.js";
import * as get_nodes from "./get_nodes.js";
import * as get_map from "./get_map.js";
import * as get_code from "./get_code.js";
import * as get_rules_files from "./get_rules_files.js";
import * as repo_map from "./repo_map.js";
import * as shortest_path from "./shortest_path.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { Request, Response, NextFunction, Express } from "express";
import * as stagehand from "./stagehand/tools.js";
import { MCPServer } from "./server.js";

const USE_STAGEHAND: boolean =
  process.env.USE_STAGEHAND === "true" || process.env.USE_STAGEHAND === "1";

const server = new MCPServer(
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

export type Json = Record<string, unknown> | undefined;

export interface Tool {
  name: string;
  description: string;
  inputSchema: Json;
}

function getMcpTools(): Tool[] {
  const coreTools = [
    search.SearchTool,
    get_nodes.GetNodesTool,
    get_map.GetMapTool,
    repo_map.RepoMapTool,
    get_code.GetCodeTool,
    shortest_path.ShortestPathTool,
    get_rules_files.GetRulesFilesTool,
  ];
  if (USE_STAGEHAND) {
    coreTools.push(...stagehand.TOOLS);
  }
  return coreTools;
}

server.get_server().setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools: getMcpTools() };
});

server
  .get_server()
  .setRequestHandler(CallToolRequestSchema, async (request, extra) => {
    const { name, arguments: args } = request.params;
    switch (name) {
      case search.SearchTool.name: {
        const fa = search.SearchSchema.parse(args);
        return await search.search(fa);
      }
      case get_nodes.GetNodesTool.name: {
        const fa = get_nodes.GetNodesSchema.parse(args);
        return await get_nodes.getNodes(fa);
      }
      case get_map.GetMapTool.name: {
        const fa = get_map.GetMapSchema.parse(args);
        return await get_map.getMap(fa);
      }
      case get_code.GetCodeTool.name: {
        const fa = get_code.GetCodeSchema.parse(args);
        return await get_code.getCode(fa);
      }
      case shortest_path.ShortestPathTool.name: {
        const fa = shortest_path.ShortestPathSchema.parse(args);
        return await shortest_path.shortestPath(fa);
      }
      case repo_map.RepoMapTool.name: {
        const fa = repo_map.RepoMapSchema.parse(args);
        return await repo_map.repoMap(fa);
      }
      case get_rules_files.GetRulesFilesTool.name: {
        return await get_rules_files.getRulesFiles();
      }
      default:
        if (USE_STAGEHAND && name.startsWith("stagehand_")) {
          return await stagehand.call(name, args || {}, extra.sessionId);
        }
        throw new Error(`Unknown tool: ${name}`);
    }
  });

// Handle server lifecycle
server.get_server().onerror = (error) => console.error("[MCP Error]", error);
server.get_server().onclose = () =>
  console.log("[MCP] Server connection closed");

export function bearerToken(req: Request, res: Response, next: NextFunction) {
  const apiToken = process.env.API_TOKEN;
  if (!apiToken) {
    return next();
  }
  const requestToken =
    req.header("Authorization") || req.header("authorization");
  if (!requestToken || requestToken !== `Bearer ${apiToken}`) {
    res.status(401).json({ error: "Unauthorized: Invalid API token" });
    return;
  }
  next();
}

export function mcp_routes(app: Express) {
  app.get("/mcp", bearerToken, async (req, res) => {
    await server.handleGetRequest(req, res);
  });

  app.post("/mcp", bearerToken, async (req, res) => {
    await server.handlePostRequest(req, res);
  });
}

export function sse_routes(app: Express) {
  let currentTransport: SSEServerTransport | null = null;

  app.get("/sse", bearerToken, async (_, res) => {
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
  app.post("/messages", bearerToken, async (req, res) => {
    console.log("===> messages - handling POST");
    try {
      if (currentTransport) {
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
