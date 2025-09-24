
import { cleanSegments } from "./intelligence/summary/preprocess.js";
import { llmSummary } from "./intelligence/summary/llm.js";
import { generate_conversation_summary } from "../graph/neo4j.js";
import { callGenerateObject } from "../aieo/src/stream.js";

interface OrchestrateSummaryOptions {
  conversation_history?: any[];
  tool_outputs?: any[];
  meta?: Record<string, any>;
  embeddings?: any[];
  llmConfig?: any;
}

  export async function orchestrateConversationSummary(options: Partial<OrchestrateSummaryOptions> = {}) {
    const conversation_history = options.conversation_history || [];
    const tool_outputs = options.tool_outputs || [];
    const meta = options.meta || {};
    const embeddings = options.embeddings || [];
    const llmConfig = options.llmConfig;

    const cleanedHistory = cleanSegments(conversation_history);
    const cleanedTools = cleanSegments(tool_outputs);
    const body = [...cleanedHistory, ...cleanedTools].join("\n");

    let summary = "";
    if (llmConfig && llmConfig.provider && llmConfig.apiKey && llmConfig.schema) {
      try {
        const result = await callGenerateObject({
          prompt: body,
          provider: llmConfig.provider,
          apiKey: llmConfig.apiKey,
          schema: llmConfig.schema,
          ...(llmConfig.extra || {})
        });
        if (typeof result === "object" && result.summary) summary = result.summary;
        else if (typeof result === "string") summary = result;
      } catch {}
    }
    if (!summary) {
      summary = await llmSummary(body);
    }

    return generate_conversation_summary(
      cleanedHistory,
      cleanedTools,
      meta,
      embeddings
    );
  }
