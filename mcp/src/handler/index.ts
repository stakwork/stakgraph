import { z } from 'zod';
import { Express } from 'express';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { MCPHttpServer } from './server.js';
import { listConcepts, learnConcept, searchLogsHandler } from './tools.js';

function createServer(): McpServer {
  const server = new McpServer({
    name: "Stakgraph",
    version: "0.1.0",
  });

  server.registerTool(
    "list_concepts",
    {
      description: "Lists all concepts (features) in the knowledge base. Optionally filter by repository.",
      inputSchema: {
        repo: z.string().optional().describe("Optional repository filter in 'owner/repo' format"),
      },
    },
    async ({ repo }) => listConcepts({ repo })
  );

  server.registerTool(
    "learn_concept",
    {
      description: "Gets detailed information and documentation about a specific concept (feature) by its ID.",
      inputSchema: {
        id: z.string().describe("The concept/feature ID to retrieve"),
        repo: z.string().optional().describe("Optional repository filter in 'owner/repo' format"),
      },
    },
    async ({ id, repo }) => learnConcept({ id, repo })
  );

  server.registerTool(
    "search_logs",
    {
      description: "Search application logs using Quickwit. Supports Lucene query syntax.",
      inputSchema: {
        query: z.string().describe("The search query (Lucene syntax supported). Use '*' to match all logs."),
        max_hits: z.number().optional().describe("Maximum number of results to return (default: 100)"),
        start_timestamp: z.number().optional().describe("Start timestamp filter (Unix epoch in seconds)"),
        end_timestamp: z.number().optional().describe("End timestamp filter (Unix epoch in seconds)"),
      },
    },
    async ({ query, max_hits, start_timestamp, end_timestamp }) => 
      searchLogsHandler({ query, max_hits, start_timestamp, end_timestamp })
  );

  return server;
}

const httpServer = new MCPHttpServer(createServer);

export function mcp_routes(app: Express) {
  app.get('/mcp', async (req, res) => {
    await httpServer.handleGetRequest(req, res);
  });

  app.post('/mcp', async (req, res) => {
    await httpServer.handlePostRequest(req, res);
  });
}
