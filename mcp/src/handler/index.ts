import { z } from 'zod';
import { Express } from 'express';
import { createMcpHandler } from 'mcp-handler';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { createExpressAdapter } from './utils.js';
import { listConcepts, learnConcept, searchLogsHandler } from './tools.js';

function registerTools(server: McpServer) {
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
}

export function mcp_routes(app: Express) {
  const handler = createMcpHandler(registerTools, {}, {
    basePath: "/mcp",
    verboseLogs: true
  });

  app.all('/mcp*', createExpressAdapter(handler));
}
