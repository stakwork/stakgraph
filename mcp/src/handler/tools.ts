import { GraphStorage } from '../gitree/store/index.js';
import { formatFeatureWithDetails } from '../gitree/utils.js';
import { searchLogs } from './logs.js';

type ToolResult = {
  content: { type: "text"; text: string }[];
  isError?: boolean;
};

export async function listConcepts(params: { repo?: string }): Promise<ToolResult> {
  const storage = new GraphStorage();
  await storage.initialize();

  const features = await storage.getAllFeatures(params.repo);

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
        repo: params.repo || "all",
      }, null, 2)
    }],
  };
}

export async function learnConcept(params: { id: string; repo?: string }): Promise<ToolResult> {
  const storage = new GraphStorage();
  await storage.initialize();

  const feature = await storage.getFeature(params.id, params.repo);

  if (!feature) {
    return {
      content: [{ type: "text" as const, text: `Concept not found: ${params.id}` }],
      isError: true,
    };
  }

  const response = await formatFeatureWithDetails(feature, storage);

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

export async function searchLogsHandler(params: {
  query: string;
  max_hits?: number;
  start_timestamp?: number;
  end_timestamp?: number;
}): Promise<ToolResult> {
  try {
    const result = await searchLogs({
      query: params.query,
      max_hits: params.max_hits,
      start_timestamp: params.start_timestamp,
      end_timestamp: params.end_timestamp,
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
