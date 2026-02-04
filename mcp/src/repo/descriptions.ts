import { Request, Response } from "express";
import { db } from "../graph/neo4j.js";
import { generateText } from "ai";
import {
  getModel,
  getApiKeyForProvider,
  Provider,
  getTokenPricing,
} from "../aieo/src/index.js";
import { vectorizeCodeDocument } from "../vector/index.js";
import * as asyncReqs from "../graph/reqs.js";
import { setBusy } from "../busy.js";

// Hardcoded provider for now as we want to use 'haiku' specifically from Anthropic
const PROVIDER: Provider = "anthropic";
const MODEL_NAME = "haiku";

function extractRepoPaths(repo_url?: string): string[] | null {
  if (!repo_url || repo_url.trim() === "") return null;
  
  const urls = repo_url.split(",").map(u => u.trim()).filter(u => u.length > 0);
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
  const batch_size = parseInt(req.body.batch_size || "25");
  const repo_url = req.body.repo_url as string | undefined;

  if (isNaN(cost_limit) || cost_limit <= 0) {
    res.status(400).json({ error: "Invalid cost_limit. Must be a positive number." });
    return;
  }
  if (isNaN(batch_size) || batch_size <= 0) {
    res.status(400).json({ error: "Invalid batch_size. Must be a positive integer." });
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
    const testNodes = await db.get_nodes_without_description(1, repo_paths);
    if (testNodes.length === 0) {
      res.status(400).json({ 
        error: `No nodes found for repository: ${repo_url}. Repositories available may not match.`,
        repo_paths 
      });
      return;
    }
  }

  console.log(
    `[describe_nodes] Starting job. Provider: ${PROVIDER}, Model: ${MODEL_NAME}, Cost limit: $${cost_limit}, Batch size: ${batch_size}${repo_paths ? `, Repos: ${repo_paths.join(", ")}` : ""}`,
  );
  res.json({
    request_id,
    status: "pending",
    message: "Started description generation job",
  });

  setBusy(true);

  try {
    let totalCost = 0;
    let totalProcessed = 0;
    let totalTokens = { input: 0, output: 0 };
    const pricing = getTokenPricing(PROVIDER);

    // Loop until cost limit reached or no more nodes
    while (true) {
      if (totalCost >= cost_limit) {
        console.log(
          `[describe_nodes] Cost limit reached ($${totalCost.toFixed(4)} >= $${cost_limit}). Stopping.`,
        );
        break;
      }

      const nodes = await db.get_nodes_without_description(batch_size, repo_paths);
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

      // Process batch sequentially
      for (const node of nodes) {
        // double check cost before each node to be safe
        if (totalCost >= cost_limit) break;

        try {
          const nodeType =
            node.labels.find((l) =>
              [
                "Class",
                "Endpoint",
                "Request",
                "Function",
                "Datamodel",
              ].includes(l),
            ) || "Node";
          const content = node.properties.body || "";
          const existingDocs = node.properties.docs || "";
          const name = node.properties.name || "Unknown";

          const prompt = `
                          Please write a short, concise description (1-3 sentences) for this ${nodeType}.
                          Do not include the code itself in the description, just describe what it does.

                          Name: ${name}
                          Docs: ${existingDocs}

                          Code:
                          ${content.slice(0, 2000)} // Truncate to avoid context limit issues
                          `;

          const model = getModel(PROVIDER, { modelName: MODEL_NAME });

          if (totalProcessed === 0) {
            console.log(`[describe_nodes] Invoking ${PROVIDER}/${MODEL_NAME} for first node: ${name}`);
          }

          const { text, usage } = await generateText({
            model,
            prompt,
          });

          // Calculate cost
          // Input cost: (inputTokens / 1,000,000) * inputTokenPrice
          // Output cost: (outputTokens / 1,000,000) * outputTokenPrice
          const inputCost =
            ((usage.inputTokens || 0) / 1000000) * pricing.inputTokenPrice;
          const outputCost =
            ((usage.outputTokens || 0) / 1000000) * pricing.outputTokenPrice;
          const nodeCost = inputCost + outputCost;

          totalCost += nodeCost;
          totalTokens.input += usage.inputTokens || 0;
          totalTokens.output += usage.outputTokens || 0;

          // Vectorize the NEW description
          const embeddings = await vectorizeCodeDocument(text);

          // Update DB
          if (!node.ref_id) {
            console.warn(`[describe_nodes] Node ${name} missing ref_id, skipping`);
            continue;
          }
          await db.update_node_description_and_embeddings(
            node.ref_id,
            text,
            embeddings,
          );
          totalProcessed++;
          console.log(
            `[describe_nodes] Processed ${name} ($${nodeCost.toFixed(6)})`,
          );
        } catch (e) {
          console.error(
            `[describe_nodes] Error processing node ${node.properties.name}:`,
            e,
          );
        }
      }
    }

    const result = {
      success: true,
      processed: totalProcessed,
      total_cost: totalCost,
      total_tokens: totalTokens,
    };

    console.log(`[describe_nodes] Finished. ${JSON.stringify(result)}`);
    asyncReqs.finishReq(request_id, result);
  } catch (error: any) {
    console.error("[describe_nodes] Fatal error:", error);
    asyncReqs.failReq(request_id, error.message || error.toString());
  } finally {
    setBusy(false);
  }
};
