import { z } from 'zod';
import { GraphStorage } from '../gitree/store/index.js';
import { formatFeatureWithDetails } from '../gitree/utils.js';
import { searchLogs } from './logs.js';

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

export function registerTools(server: McpServer) {
  // List all concepts (features)
  server.tool(
    "list_concepts",
    "Lists all concepts (features) in the knowledge base. Optionally filter by repository.",
    {
      repo: z.string().optional().describe("Optional repository filter in 'owner/repo' format"),
    },
    async ({ repo }) => {
      const storage = new GraphStorage();
      await storage.initialize();

      const features = await storage.getAllFeatures(repo as string | undefined);

      const conceptList = features.map((f) => ({
        id: f.id,
        repo: f.repo,
        ref_id: f.ref_id,
        name: f.name,
        description: f.description,
        prCount: f.prNumbers.length,
        commitCount: (f.commitShas || []).length,
        lastUpdated: f.lastUpdated.toISOString(),
        hasDocumentation: !!f.documentation,
      }));

      return {
        content: [{
          type: "text" as const,
          text: JSON.stringify({
            concepts: conceptList,
            total: conceptList.length,
            repo: repo || "all",
          }, null, 2)
        }],
      };
    }
  );

  // Learn about a specific concept
  server.tool(
    "learn_concept",
    "Gets detailed information and documentation about a specific concept (feature) by its ID.",
    {
      id: z.string().describe("The concept/feature ID to retrieve"),
      repo: z.string().optional().describe("Optional repository filter in 'owner/repo' format"),
    },
    async ({ id, repo }) => {
      const storage = new GraphStorage();
      await storage.initialize();

      const feature = await storage.getFeature(id as string, repo as string | undefined);

      if (!feature) {
        return {
          content: [{ type: "text" as const, text: `Concept not found: ${id}` }],
          isError: true,
        };
      }

      const response = await formatFeatureWithDetails(feature, storage);

      // Return the documentation as the primary content
      const documentation = feature.documentation || "No documentation available for this concept.";

      return {
        content: [{
          type: "text" as const,
          text: `# ${feature.name}\n\n${feature.description || ""}\n\n## Documentation\n\n${documentation}\n\n## Metadata\n\n${JSON.stringify({
            id: response.feature.id,
            ref_id: response.feature.ref_id,
            prCount: response.feature.prNumbers.length,
            commitCount: response.feature.commitShas.length,
            createdAt: response.feature.createdAt,
            lastUpdated: response.feature.lastUpdated,
          }, null, 2)}`
        }],
      };
    }
  );

  // Search logs via Quickwit
  server.tool(
    "search_logs",
    "Search application logs using Quickwit. Supports Lucene query syntax.",
    {
      query: z.string().describe("The search query (Lucene syntax supported). Use '*' to match all logs."),
      max_hits: z.number().optional().describe("Maximum number of results to return (default: 100)"),
      start_timestamp: z.number().optional().describe("Start timestamp filter (Unix epoch in seconds)"),
      end_timestamp: z.number().optional().describe("End timestamp filter (Unix epoch in seconds)"),
    },
    async ({ query, max_hits, start_timestamp, end_timestamp }) => {
      try {
        const result = await searchLogs({
          query: query as string,
          max_hits: max_hits as number | undefined,
          start_timestamp: start_timestamp as number | undefined,
          end_timestamp: end_timestamp as number | undefined,
        });

        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({
              num_hits: result.num_hits,
              elapsed_time_micros: result.elapsed_time_micros,
              hits: result.hits,
            }, null, 2)
          }],
        };
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        return {
          content: [{ type: "text" as const, text: `Error searching logs: ${errorMessage}` }],
          isError: true,
        };
      }
    }
  );
}
