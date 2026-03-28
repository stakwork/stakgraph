import { Request, Response } from "express";
import { db } from "../graph/neo4j.js";
import { generateText } from "ai";
import { resolveLLMConfig, getTokenPricing } from "../aieo/src/index.js";
import { vectorizeBatch } from "../vector/index.js";
import * as asyncReqs from "../graph/reqs.js";
import { startTracking, endTracking } from "../busy.js";
import PQueueModule from "p-queue";
const PQueue = (PQueueModule as any).default ?? PQueueModule;



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
    let totalTokens = { input: 0, output: 0 };
    const pricing = getTokenPricing(llm.provider);
    const model = llm.model;

    // Loop until cost limit reached or no more nodes
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

      console.log(
        `[describe_nodes] Processing batch of ${nodes.length} nodes...`,
      );

      asyncReqs.updateReq(request_id, {
        processed: totalProcessed,
        total_cost: totalCost,
        total_tokens: totalTokens,
        current_batch_size: nodes.length,
      });

      // Fan out LLM calls in parallel with concurrency cap
      const queue = new PQueue({ concurrency });
      type NodeResult = {
        ref_id: string;
        name: string;
        text: string;
        inputTokens: number;
        outputTokens: number;
        cost: number;
      };
      const results: NodeResult[] = [];

      await queue.addAll(
        nodes
          .filter((node) => !!node.ref_id)
          .map((node) => async () => {
            if (totalCost >= cost_limit) return;
            const nodeType =
              node.labels.find((l) =>
                [
                  "Class",
                  "Endpoint",
                  "Request",
                  "Function",
                  "Datamodel",
                  "Page",
                  "Trait",
                  "Var",
                ].includes(l),
              ) || "Node";
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
              console.log(
                `[describe_nodes] LLM done: ${name} ($${cost.toFixed(6)})`,
              );
              results.push({
                ref_id: node.ref_id!,
                name,
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

      // Accumulate costs
      for (const r of results) {
        totalCost += r.cost;
        totalTokens.input += r.inputTokens;
        totalTokens.output += r.outputTokens;
      }

      // Bulk write to Neo4j
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
        totalProcessed += results.length;
      }

      console.log(
        `[describe_nodes] Batch done. total_processed=${totalProcessed} total_cost=$${totalCost.toFixed(6)} budget_remaining=$${(cost_limit - totalCost).toFixed(6)}`,
      );
    }

    const result = {
      success: true,
      processed: totalProcessed,
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

    while (true) {
      const nodes = await db.get_nodes_with_description_without_embeddings(
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

      const texts = nodes.map((n) => n.description);
      const embeddings = await vectorizeBatch(texts);

      const batch = nodes.map((n, i) => ({
        ref_id: n.ref_id,
        embeddings: embeddings[i],
      }));

      await db.bulk_update_embeddings(batch);
      totalProcessed += nodes.length;

      console.log(`[embed_nodes] Embedded ${totalProcessed} nodes so far.`);
    }

    const result = { success: true, processed: totalProcessed };
    console.log(`[embed_nodes] Finished. ${JSON.stringify(result)}`);
    asyncReqs.finishReq(request_id, result);
  } catch (error: any) {
    console.error("[embed_nodes] Fatal error:", error);
    asyncReqs.failReq(request_id, error.message || error.toString());
  } finally {
    endTracking(opId);
  }
};