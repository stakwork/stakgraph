import { Request, Response } from "express";
import { db } from "../graph/neo4j.js";
import { generateText } from "ai";
import { resolveLLMConfig, getTokenPricing } from "../aieo/src/index.js";
import { vectorizeBatch } from "../vector/index.js";
import * as asyncReqs from "../graph/reqs.js";
import { startTracking, endTracking } from "../busy.js";
import PQueueModule from "p-queue";
const PQueue = (PQueueModule as any).default ?? PQueueModule;
import { shouldSkipDescription } from "./builtin-skips.js";

const BUILTIN_DESCRIPTION_MARKER = "__builtin_or_external_symbol__";

function pickNodeType(labels: string[]): string {
  return (
    labels.find((l) =>
      ["Class", "Endpoint", "Request", "Function", "Datamodel", "Page", "Trait", "Var"].includes(l),
    ) || "Node"
  );
}

function embeddingTextFromNode(input: {
  labels?: string[];
  name?: string;
  docs?: string;
  body?: string;
  description?: string;
}) {
  const description = (input.description || "").trim();
  if (description && description !== BUILTIN_DESCRIPTION_MARKER) {
    return description.slice(0, 3000);
  }
  const nodeType = pickNodeType(input.labels || []);
  const name = (input.name || "Unknown").trim();
  const docs = (input.docs || "").trim();
  const body = (input.body || "").trim();
  const bodySlice = body.slice(0, 2500);
  return `${nodeType}: ${name}\n${docs ? `Docs: ${docs}\n` : ""}${bodySlice}`;
}


function extractRepoPaths(repo_url?: string): string[] | null {
  if (!repo_url || repo_url.trim() === "") return null;

  const urls = repo_url
    .split(",")
    .map((u) => u.trim())
    .filter((u) => u.length > 0);
  const paths: string[] = [];

  for (const url of urls) {
    const match = url.match(/github\.com\/([^\/]+\/[^\/]+?)(\.git)?$/);
    if (match) {
      paths.push(match[1]);
    } else {
      throw new Error(`Invalid repo URL format: ${url}`);
    }
  }

  return paths.length > 0 ? paths : null;
}

export const describe_nodes_agent = async (req: Request, res: Response) => {
  const request_id = asyncReqs.startReq();

  const cost_limit = parseFloat(req.body.cost_limit || "0.5");
  const batch_size = parseInt(req.body.batch_size || "50");
  const concurrency = parseInt(req.body.concurrency || "5");
  const repo_url = req.body.repo_url as string | undefined;
  const file_paths = (req.body.file_paths || []) as string[];
  const do_embed = req.body.embed !== false && req.body.embed !== "false";
  const reqModel = req.body.model as string | undefined;
  const reqApiKey = req.body.apiKey as string | undefined;

  if (isNaN(cost_limit) || cost_limit <= 0) {
    res
      .status(400)
      .json({ error: "Invalid cost_limit. Must be a positive number." });
    return;
  }
  if (isNaN(batch_size) || batch_size <= 0) {
    res
      .status(400)
      .json({ error: "Invalid batch_size. Must be a positive integer." });
    return;
  }

  let repo_paths: string[] | null = null;
  try {
    repo_paths = extractRepoPaths(repo_url);
  } catch (error: any) {
    res.status(400).json({ error: error.message });
    return;
  }

  if (repo_paths) {
    const testNodes = await db.get_nodes_without_description(
      1,
      repo_paths,
      file_paths,
    );
    if (testNodes.length === 0) {
      res.status(400).json({
        error: `No nodes found for repository: ${repo_url}. Repositories available may not match.`,
        repo_paths,
      });
      return;
    }
  }

  const llm = resolveLLMConfig({ model: reqModel, apiKey: reqApiKey, light: true });

  console.log(
    `[describe_nodes] Starting job. Provider: ${llm.provider}, Model: ${llm.modelName || "(default)"}, Cost limit: $${cost_limit}, Batch size: ${batch_size}, Concurrency: ${concurrency}${repo_paths ? `, Repos: ${repo_paths.join(", ")}` : ""}${file_paths.length > 0 ? `, Files: ${file_paths.length}` : ""}`,
  );
  res.json({
    request_id,
    status: "pending",
    message: "Started description generation job",
  });

  const opId = startTracking("describe_nodes");

  try {
    let totalCost = 0;
    let totalProcessed = 0;
    let describedNodes = 0;
    let skippedBuiltinDescriptions = 0;
    let embeddedBuiltinNodes = 0;
    let totalTokens = { input: 0, output: 0 };
    const pricing = getTokenPricing(llm.provider);
    const model = llm.model;

    while (true) {
      if (totalCost >= cost_limit) {
        console.log(
          `[describe_nodes] Cost limit reached ($${totalCost.toFixed(4)} >= $${cost_limit}). Stopping.`,
        );
        break;
      }

      const nodes = await db.get_nodes_without_description(
        batch_size,
        repo_paths,
        file_paths,
      );
      if (nodes.length === 0) {
        console.log(`[describe_nodes] No more nodes to process.`);
        break;
      }

      const candidates = nodes.filter((node) => !!node.ref_id);
      const builtinNodes = candidates.filter((node) =>
        shouldSkipDescription(node.properties.name || "", node.properties.file || "", node.labels || []),
      );
      const llmNodes = candidates.filter(
        (node) => !shouldSkipDescription(node.properties.name || "", node.properties.file || "", node.labels || []),
      );

      asyncReqs.updateReq(request_id, {
        processed: totalProcessed,
        total_cost: totalCost,
        total_tokens: totalTokens,
        current_batch_size: nodes.length,
        llm_batch_size: llmNodes.length,
        builtin_batch_size: builtinNodes.length,
      });

      const queue = new PQueue({ concurrency });
      type NodeResult = {
        ref_id: string;
        text: string;
        inputTokens: number;
        outputTokens: number;
        cost: number;
      };
      const results: NodeResult[] = [];

      await queue.addAll(
        llmNodes
          .map((node) => async () => {
            if (totalCost >= cost_limit) return;
            const nodeType = pickNodeType(node.labels || []);
            const content = node.properties.body || "";
            const existingDocs = node.properties.docs || "";
            const name = node.properties.name || "Unknown";
            const prompt = `Please write a short, concise description (1-3 sentences) for this ${nodeType}.
Do not include the code itself in the description, just describe what it does.

Name: ${name}
Docs: ${existingDocs}

Code:
${content.slice(0, 2000)}`;
            try {
              const { text, usage } = await generateText({ model, prompt });
              const inputCost =
                ((usage.inputTokens || 0) / 1000000) * pricing.inputTokenPrice;
              const outputCost =
                ((usage.outputTokens || 0) / 1000000) *
                pricing.outputTokenPrice;
              const cost = inputCost + outputCost;
              results.push({
                ref_id: node.ref_id!,
                text,
                inputTokens: usage.inputTokens || 0,
                outputTokens: usage.outputTokens || 0,
                cost,
              });
            } catch (e) {
              console.error(`[describe_nodes] Error on node ${name}:`, e);
            }
          }),
      );

      for (const r of results) {
        totalCost += r.cost;
        totalTokens.input += r.inputTokens;
        totalTokens.output += r.outputTokens;
      }
      describedNodes += results.length;

      if (builtinNodes.length > 0) {
        if (do_embed) {
          const builtinTexts = builtinNodes.map((n) =>
            embeddingTextFromNode({
              labels: n.labels,
              name: n.properties.name,
              docs: n.properties.docs,
              body: n.properties.body,
            }),
          );
          const builtinEmbeddings = await vectorizeBatch(builtinTexts);
          await db.bulk_update_descriptions_and_embeddings(
            builtinNodes.map((n, i) => ({
              ref_id: n.ref_id!,
              description: BUILTIN_DESCRIPTION_MARKER,
              embeddings: builtinEmbeddings[i],
            })),
          );
          embeddedBuiltinNodes += builtinNodes.length;
        } else {
          await db.bulk_update_descriptions(
            builtinNodes.map((n) => ({
              ref_id: n.ref_id!,
              description: BUILTIN_DESCRIPTION_MARKER,
            })),
          );
        }
        skippedBuiltinDescriptions += builtinNodes.length;
      }

      if (results.length > 0) {
        if (do_embed) {
          const texts = results.map((r) => r.text);
          const embeddings = await vectorizeBatch(texts);
          await db.bulk_update_descriptions_and_embeddings(
            results.map((r, i) => ({
              ref_id: r.ref_id,
              description: r.text,
              embeddings: embeddings[i],
            })),
          );
        } else {
          await db.bulk_update_descriptions(
            results.map((r) => ({ ref_id: r.ref_id, description: r.text })),
          );
        }
      }
      totalProcessed += results.length + builtinNodes.length;

      console.log(
        `[describe_nodes] batch: fetched=${nodes.length} llm=${llmNodes.length} builtin=${builtinNodes.length} processed_total=${totalProcessed} cost_total=$${totalCost.toFixed(6)} budget_remaining=$${Math.max(0, cost_limit - totalCost).toFixed(6)}`,
      );
    }

    const result = {
      success: true,
      processed: totalProcessed,
      described_nodes: describedNodes,
      skipped_builtin_descriptions: skippedBuiltinDescriptions,
      embedded_builtin_nodes: embeddedBuiltinNodes,
      total_cost: totalCost,
      total_tokens: totalTokens,
      usage: {
        inputTokens: totalTokens.input,
        outputTokens: totalTokens.output,
        totalTokens: totalTokens.input + totalTokens.output,
      },
    };

    console.log(`[describe_nodes] Finished. ${JSON.stringify(result)}`);
    asyncReqs.finishReq(request_id, result);
  } catch (error: any) {
    console.error("[describe_nodes] Fatal error:", error);
    asyncReqs.failReq(request_id, error.message || error.toString());
  } finally {
    endTracking(opId);
  }
};
export const embed_nodes_agent = async (req: Request, res: Response) => {
  const request_id = asyncReqs.startReq();

  const batch_size = parseInt(req.body.batch_size || "100");
  const repo_url = req.body.repo_url as string | undefined;
  const file_paths = (req.body.file_paths || []) as string[];

  if (isNaN(batch_size) || batch_size <= 0) {
    res
      .status(400)
      .json({ error: "Invalid batch_size. Must be a positive integer." });
    return;
  }

  let repo_paths: string[] | null = null;
  try {
    repo_paths = extractRepoPaths(repo_url);
  } catch (error: any) {
    res.status(400).json({ error: error.message });
    return;
  }

  console.log(
    `[embed_nodes] Starting job. Batch size: ${batch_size}${repo_paths ? `, Repos: ${repo_paths.join(", ")}` : ""}`,
  );
  res.json({
    request_id,
    status: "pending",
    message: "Started embedding job",
  });

  const opId = startTracking("embed_nodes");

  try {
    let totalProcessed = 0;
    let embeddedFromDescriptions = 0;
    let embeddedFromCode = 0;

    while (true) {
      const nodes = await db.get_nodes_without_embeddings(
        batch_size,
        repo_paths,
        file_paths,
      );
      if (nodes.length === 0) {
        console.log(`[embed_nodes] No more nodes to embed.`);
        break;
      }

      console.log(`[embed_nodes] Embedding batch of ${nodes.length} nodes...`);

      asyncReqs.updateReq(request_id, {
        processed: totalProcessed,
        current_batch_size: nodes.length,
      });

      const texts = nodes.map((n) => {
        const useDescription =
          !!n.description &&
          n.description.trim().length > 0 &&
          n.description !== BUILTIN_DESCRIPTION_MARKER;
        if (useDescription) {
          embeddedFromDescriptions += 1;
        } else {
          embeddedFromCode += 1;
        }
        return embeddingTextFromNode({
          labels: n.labels,
          name: n.name,
          body: n.body,
          description: n.description,
        });
      });
      const embeddings = await vectorizeBatch(texts);

      const batch = nodes.map((n, i) => ({
        ref_id: n.ref_id,
        embeddings: embeddings[i],
      }));

      await db.bulk_update_embeddings(batch);
      totalProcessed += nodes.length;

      console.log(
        `[embed_nodes] batch: size=${nodes.length} from_description=${embeddedFromDescriptions} from_code=${embeddedFromCode} processed_total=${totalProcessed}`,
      );
    }

    const result = {
      success: true,
      processed: totalProcessed,
      embedded_from_descriptions: embeddedFromDescriptions,
      embedded_from_code: embeddedFromCode,
    };
    console.log(`[embed_nodes] Finished. ${JSON.stringify(result)}`);
    asyncReqs.finishReq(request_id, result);
  } catch (error: any) {
    console.error("[embed_nodes] Fatal error:", error);
    asyncReqs.failReq(request_id, error.message || error.toString());
  } finally {
    endTracking(opId);
  }
};