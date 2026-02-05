import { z } from 'zod';
import { Express } from 'express';
import { createMcpHandler } from 'mcp-handler';
import { createExpressAdapter } from './utils.js';
import { listConcepts, learnConcept, searchLogsHandler } from './tools.js';

type McpServer = {
  tool: (
    name: string,
    description: string,
    schema: Record<string, z.ZodTypeAny>,
    handler: (params: Record<string, unknown>) => Promise<{
      content: { type: "text"; text: string }[];
      isError?: boolean;
    }>
  ) => void;
};

function registerTools(server: McpServer) {
  server.tool(
    "list_concepts",
    "Lists all concepts (features) in the knowledge base. Optionally filter by repository.",
    {
      repo: z.string().optional().describe("Optional repository filter in 'owner/repo' format"),
    },
    async (params) => listConcepts({ repo: params.repo as string | undefined })
  );

  server.tool(
    "learn_concept",
    "Gets detailed information and documentation about a specific concept (feature) by its ID.",
    {
      id: z.string().describe("The concept/feature ID to retrieve"),
      repo: z.string().optional().describe("Optional repository filter in 'owner/repo' format"),
    },
    async (params) => learnConcept({
      id: params.id as string,
      repo: params.repo as string | undefined,
    })
  );

  server.tool(
    "search_logs",
    "Search application logs using Quickwit. Supports Lucene query syntax.",
    {
      query: z.string().describe("The search query (Lucene syntax supported). Use '*' to match all logs."),
      max_hits: z.number().optional().describe("Maximum number of results to return (default: 100)"),
      start_timestamp: z.number().optional().describe("Start timestamp filter (Unix epoch in seconds)"),
      end_timestamp: z.number().optional().describe("End timestamp filter (Unix epoch in seconds)"),
    },
    async (params) => searchLogsHandler({
      query: params.query as string,
      max_hits: params.max_hits as number | undefined,
      start_timestamp: params.start_timestamp as number | undefined,
      end_timestamp: params.end_timestamp as number | undefined,
    })
  );
}

export function mcp_routes(app: Express) {
  const handler = createMcpHandler(registerTools, {}, {
    basePath: "/mcp",
    verboseLogs: true
  });

  app.all('/mcp*', createExpressAdapter(handler));
}
