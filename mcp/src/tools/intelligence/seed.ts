import { db } from "../../graph/neo4j.js";
import { getApiKeyForProvider, Provider } from "../../aieo/src/provider.js";
import { HintExtraction, Neo4jNode } from "../../graph/types.js";
import { z } from "zod";
import { callGenerateObject } from "../../aieo/src/index.js";

async function findNodesFromExtraction(
  extracted: HintExtraction
): Promise<{ node: Neo4jNode; relevancy: number }[]> {
  const foundNodes: { node: Neo4jNode; relevancy: number }[] = [];
  const typeMapping = {
    function_names: "Function",
    file_names: "File",
    datamodel_names: "Datamodel",
    endpoint_names: "Endpoint",
    page_names: "Page",
  };

  for (const [key, nodeType] of Object.entries(typeMapping)) {
    const weightedNodes = extracted[key as keyof HintExtraction] || [];
    for (const weightedNode of weightedNodes) {
      if (weightedNode.name && weightedNode.name.trim()) {
        const nodes = await db.findNodesByName(
          weightedNode.name.trim(),
          nodeType
        );
        for (const node of nodes) {
          foundNodes.push({ node, relevancy: weightedNode.relevancy });
        }
      }
    }
  }

  return foundNodes;
}

export async function create_hint_edges_llm(
  hint_ref_id: string,
  answer: string,
  llm_provider?: Provider | string
): Promise<{
  edges_added: number;
  linked_ref_ids: string[];
  usage: { inputTokens: number; outputTokens: number; totalTokens: number };
}> {
  if (!answer)
    return {
      edges_added: 0,
      linked_ref_ids: [],
      usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
    };
  const provider = llm_provider ? llm_provider : "anthropic";
  const apiKey = getApiKeyForProvider(provider);
  if (!apiKey)
    return {
      edges_added: 0,
      linked_ref_ids: [],
      usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
    };

  const result = await extractHintReferences(
    answer,
    provider as Provider,
    apiKey
  );

  const foundNodes = await findNodesFromExtraction(result.extraction);
  const weightedRefIds = foundNodes
    .map((item) => ({
      ref_id: item.node.ref_id || item.node.properties.ref_id,
      relevancy: item.relevancy,
    }))
    .filter((item) => item.ref_id);

  if (weightedRefIds.length === 0)
    return {
      edges_added: 0,
      linked_ref_ids: [],
      usage: result.usage,
    };

  const edgeResult = await db.createEdgesDirectly(hint_ref_id, weightedRefIds);
  return {
    ...edgeResult,
    usage: result.usage,
  };
}

export async function extractHintReferences(
  answer: string,
  provider: Provider,
  apiKey: string
): Promise<{
  extraction: HintExtraction;
  usage: { inputTokens: number; outputTokens: number; totalTokens: number };
}> {
  const truncated = answer.slice(0, 8000);
  const item = z.object({
    name: z.string(),
    relevancy: z.number().min(0).max(1),
  });
  const schema = z.object({
    function_names: z
      .array(item)
      .describe(
        "functions or react components with relevancy scores (0.0-1.0). e.g [{name: 'getUser', relevancy: 0.9}, {name: 'handleClick', relevancy: 0.6}]"
      ),
    file_names: z
      .array(item)
      .describe(
        "complete file paths with relevancy scores (0.0-1.0). e.g [{name: 'src/app/page.tsx', relevancy: 0.8}]"
      ),
    datamodel_names: z
      .array(item)
      .describe(
        "database models, schemas, or data structures with relevancy scores (0.0-1.0). e.g [{name: 'User', relevancy: 0.9}]"
      ),
    endpoint_names: z
      .array(item)
      .describe(
        "API endpoints with relevancy scores (0.0-1.0). e.g [{name: '/api/person', relevancy: 0.7}]"
      ),
    page_names: z
      .array(item)
      .describe(
        "web pages, components, or views with relevancy scores (0.0-1.0). e.g [{name: 'HomePage', relevancy: 0.8}]"
      ),
  });
  try {
    const result = await callGenerateObject({
      provider,
      apiKey,
      prompt: `Extract exact code nodes referenced with relevancy scores (0.0-1.0). Higher scores for more central/important nodes. Return JSON only. Use empty arrays if none.\n\n${truncated}`,
      schema,
    });
    return {
      extraction: result.object,
      usage: result.usage,
    };
  } catch (_) {
    return {
      extraction: {
        function_names: [],
        file_names: [],
        datamodel_names: [],
        endpoint_names: [],
        page_names: [],
      },
      usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
    };
  }
}
