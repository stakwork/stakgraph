import { Request, Response } from "express";
import * as asyncReqs from "../graph/reqs.js";
import { GraphStorage } from "./store/index.js";
import { createGitreeSessionTracker, LLMClient } from "./llm.js";
import { StreamingConceptBuilder } from "./builder.js";
import { Summarizer } from "./summarizer.js";
import { FileLinker } from "./fileLinker.js";
import { ClueAnalyzer } from "./clueAnalyzer.js";
import { Octokit } from "@octokit/rest";
import {
  getApiKeyForProvider,
  getModel,
  getModelDetails,
  getProviderOptions,
  Provider,
} from "../aieo/src/provider.js";
import { generateObject, jsonSchema } from "ai";
import { formatConceptWithDetails } from "./utils.js";
import { listConcepts } from "./service.js";
import {
  toReturnNode,
  toReturnNodeNoBody,
  parseNodeTypes,
  parseLimit,
  buildGraphMeta,
  isTrue,
  IS_TEST,
} from "../graph/utils.js";
import { NodeType, Neo4jNode } from "../graph/types.js";
import { get_context } from "../repo/agent.js";
import { cloneOrUpdateRepo } from "../repo/clone.js";
import { Concept, Usage } from "./types.js";
import { addUsage, normalizeUsage } from "../aieo/src/usage.js";
import { startTracking, endTracking } from "../busy.js";
import { generateSlug, makeRepoId } from "./store/utils.js";
import { bootstrapConcepts } from "./bootstrap.js";
import { randomUUID } from "crypto";
import { createSession, appendSessionEnd, loadStepMeta } from "../repo/session.js";

// In-memory flag to track if processing is currently running
let isProcessing = false;

function getDirectGitreeUsage(sessionId: string): Usage {
  const directUsage = loadStepMeta(sessionId)
    .filter((step) => step.label?.startsWith("gitree "))
    .map((step) => step.usage);
  return normalizeUsage(addUsage(...directUsage));
}

/**
 * Parse Git repository URL to extract owner and repo
 * Supports formats:
 * - https://github.com/owner/repo
 * - https://gitlab.com/owner/repo
 * - git@github.com:owner/repo.git
 * - https://bitbucket.org/owner/repo
 * - owner/repo
 * - Any git hosting service with owner/repo pattern
 */
function parseGitRepoUrl(url: string): { owner: string; repo: string } | null {
  try {
    let cleanUrl = url.trim().replace(/\.git$/, "");

    const ownerRepoMatch = cleanUrl.match(/^([^\/\s]+)\/([^\/\s]+)$/);
    if (ownerRepoMatch) {
      return { owner: ownerRepoMatch[1], repo: ownerRepoMatch[2] };
    }

    const sshMatch = cleanUrl.match(/git@[^:]+:(.+)/);
    if (sshMatch) {
      cleanUrl = sshMatch[1];
    } else {
      cleanUrl = cleanUrl.replace(/^https?:\/\//, "");
      cleanUrl = cleanUrl.replace(/^[^\/]+\//, "");
    }

    const pathParts = cleanUrl.split("/").filter((p) => p.length > 0);
    if (pathParts.length >= 2) {
      const owner = pathParts[0];
      const repo = pathParts[1];
      return { owner, repo };
    }

    return null;
  } catch (error) {
    return null;
  }
}

/**
 * Parse and normalize repo parameter from query string
 * Accepts: owner/repo, https://github.com/owner/repo, etc.
 * Returns normalized "owner/repo" format or undefined
 */
function parseRepoParam(req: Request): string | undefined {
  const repoParam = req.query.repo as string | undefined;
  if (!repoParam) return undefined;
  
  // Use existing parseGitRepoUrl function
  const parsed = parseGitRepoUrl(repoParam);
  if (parsed) {
    return `${parsed.owner}/${parsed.repo}`;
  }
  
  // If it looks like owner/repo already, use as-is
  if (repoParam.match(/^[^\/]+\/[^\/]+$/)) {
    return repoParam;
  }
  
  return undefined;
}

/**
 * Process a Git repository to extract concepts (PRs and commits)
 * POST /gitree/process?owner=stakwork&repo=sphinx-tribes&token=...
 * POST /gitree/process?repo_url=https://github.com/stakwork/sphinx-tribes&token=...
 * POST /gitree/process?repo_url=https://gitlab.com/owner/repo&token=...
 * POST /gitree/process?repo_url=git@github.com:owner/repo.git&token=...
 * POST /gitree/process?repo_url=...&token=...&summarize=true&link=true
 * GET /progress?request_id=xxx
 */

// curl -X POST "http://localhost:3355/gitree/process?owner=stakwork&repo=hive&summarize=true&link=true&analyze_clues=true"
export async function gitree_process(req: Request, res: Response) {
  console.log("===> gitree_process", req.path, req.method);
  const request_id = asyncReqs.startReq();

  let owner = req.query.owner as string;
  let repo = req.query.repo as string;
  const repoUrl = req.query.repo_url as string;
  const githubTokenQuery = req.query.token as string;
  const shouldSummarize = req.query.summarize === "true";
  const shouldLink = req.query.link === "true";
  const shouldAnalyzeClues = req.query.analyze_clues === "true";
  const githubToken = githubTokenQuery || process.env.GITHUB_TOKEN;


  if (repoUrl && (!owner || !repo)) {
    const parsed = parseGitRepoUrl(repoUrl);
    if (parsed) {
      owner = parsed.owner;
      repo = parsed.repo;
    }
  }

  if (!owner || !repo) {
    asyncReqs.failReq(
      request_id,
      new Error("Missing owner/repo or repo_url")
    );
    res.status(400).json({ error: "Missing owner/repo or repo_url" });
    return;
  }

  if (!githubToken) {
    asyncReqs.failReq(request_id, new Error("Missing GitHub token"));
    res.status(400).json({ error: "Missing GitHub token" });
    return;
  }

  const opId = startTracking("gitree/processRepo");

  try {
    // Process repository in background
    isProcessing = true;

    (async () => {
      const sessionId = randomUUID();
      const startTime = Date.now();
      const { modelId, provider } = getModelDetails();
      try {
        createSession(sessionId, undefined, "gitree_process", `${owner}/${repo}`);
        const anthropicKey = getApiKeyForProvider("anthropic");
        const storage = new GraphStorage();
        await storage.initialize();

        // Check if this is a brand-new repo (no concepts yet)
        const repoId = `${owner}/${repo}`;
        const existingConcepts = await storage.getAllConcepts(repoId);
        const isNewRepo = existingConcepts.length === 0;

        // Clone repo if needed for bootstrap or clue analysis
        let repoPath: string | undefined;
        if (isNewRepo || shouldAnalyzeClues) {
          repoPath = await cloneOrUpdateRepo(
            `https://github.com/${owner}/${repo}`,
            undefined,
            githubToken
          );
        }

        // Bootstrap: seed initial concepts by exploring the codebase
        let bootstrapUsage: Usage = normalizeUsage();
        if (isNewRepo && repoPath) {
          const bootstrapResult = await bootstrapConcepts(owner, repo, repoPath, storage, sessionId);
          bootstrapUsage = bootstrapResult.usage;
        }

        const sessionTracker = createGitreeSessionTracker(sessionId);

        const octokit = new Octokit({ auth: githubToken });
        const llm = new LLMClient("anthropic", anthropicKey, sessionTracker);
        const builder = new StreamingConceptBuilder(
          storage,
          llm,
          octokit,
          repoPath,
          shouldAnalyzeClues,
          sessionTracker,
        );

        const { usage: processUsage, modifiedConceptIds } = await builder.processRepo(owner, repo, sessionId);

        let summarizeUsage = null;
        let linkResult = null;

        // If summarize flag is set, run summarization after processing (only for modified concepts)
        if (shouldSummarize && modifiedConceptIds.size > 0) {
          console.log(`===> Starting concept summarization for ${modifiedConceptIds.size} modified concept(s)...`);
          const summarizer = new Summarizer(
            storage,
            "anthropic",
            anthropicKey,
            sessionTracker,
          );
          summarizeUsage = await summarizer.summarizeModifiedConcepts(Array.from(modifiedConceptIds), sessionId);
        }

        // If link flag is set, link files to concepts
        if (shouldLink) {
          console.log("===> Starting concept-file linking...");
          const linker = new FileLinker(storage);
          linkResult = await linker.linkAllConcepts();
        }

        // Note: Clue analysis is now done during PR/commit processing
        // within builder.processRepo() if shouldAnalyzeClues is true

        // Build response message and usage
        const messageParts = [`Processed ${owner}/${repo}`];
        if (shouldSummarize) messageParts.push("summarized");
        if (shouldLink) messageParts.push("linked files");
        if (shouldAnalyzeClues) messageParts.push("analyzed clues");

        const totalUsage = normalizeUsage(addUsage(bootstrapUsage, processUsage, summarizeUsage));
        await storage.addToTotalUsage(repoId, totalUsage);
        
        // Get the new cumulative total
        const cumulativeUsage = await storage.getTotalUsage(repoId);

        const result: any = {
          status: "success",
          message: messageParts.join(", "),
          usage: totalUsage,
          cumulativeUsage: cumulativeUsage,
          sessionId,
        };

        if (linkResult) {
          result.linkResult = linkResult;
        }

        asyncReqs.finishReq(request_id, result);
        await appendSessionEnd(sessionId, {
          end_time: new Date().toISOString(),
          model: modelId,
          provider,
          duration_ms: Date.now() - startTime,
          status: "success",
          token_usage: getDirectGitreeUsage(sessionId),
        });
        isProcessing = false;
        endTracking(opId);
      } catch (error) {
        console.error("===> gitree_process background task failed:", error);
        asyncReqs.failReq(request_id, error);
        await appendSessionEnd(sessionId, {
          end_time: new Date().toISOString(),
          model: modelId,
          provider,
          duration_ms: Date.now() - startTime,
          status: "error",
          error_message: error instanceof Error ? error.message : String(error),
        });
        isProcessing = false;
        endTracking(opId);
      }
    })();

    res.json({ request_id, status: "pending" });
  } catch (error) {
    console.log("===> error");
    asyncReqs.failReq(request_id, error);
    isProcessing = false;
    endTracking(opId);
    res.status(500).json({ error: "Failed to process repository" });
  }
}

/**
 * List all concepts
 * GET /gitree/concepts?repo=owner/repo (optional)
 */
export async function gitree_list_concepts(req: Request, res: Response) {
  try {
    const repo = parseRepoParam(req);
    const result = await listConcepts(repo);

    // Get checkpoint and usage - aggregated if no repo specified
    const storage = new GraphStorage();
    await storage.initialize();
    const { lastProcessedTimestamp, cumulativeUsage } = repo
      ? {
          lastProcessedTimestamp: (await storage.getChronologicalCheckpoint(repo))?.lastProcessedTimestamp || null,
          cumulativeUsage: await storage.getTotalUsage(repo),
        }
      : await storage.getAggregatedMetadata();

    res.json({
      ...result,
      repo: repo || "all",
      lastProcessedTimestamp,
      cumulativeUsage,
      processing: isProcessing,
    });
  } catch (error: any) {
    console.error("Error listing concepts:", error);
    res.status(500).json({ error: error.message || "Failed to list concepts" });
  }
}

/**
 * Get a specific concept
 * GET /gitree/concepts/:id?include=files&repo=owner/repo (repo optional)
 */
export async function gitree_get_concept(req: Request, res: Response) {
  try {
    const conceptId = req.params.id as string;
    const include = req.query.include as string | undefined;
    const repo = parseRepoParam(req);
    const storage = new GraphStorage();
    await storage.initialize();

    const concept = await storage.getConcept(conceptId, repo);

    if (!concept) {
      res.status(404).json({ error: "Concept not found" });
      return;
    }

    const response: any = await formatConceptWithDetails(concept, storage);

    // Include files if requested
    if (include === "files") {
      const files = await storage.getFilesForConcept(conceptId);
      response.files = files;
    }

    res.json(response);
  } catch (error: any) {
    console.error("Error getting concept:", error);
    res.status(500).json({ error: error.message || "Failed to get concept" });
  }
}

/**
 * Update documentation for a specific concept
 * PUT /gitree/concepts/:id/documentation
 * Body: { documentation: string }
 */
export async function gitree_update_concept_documentation(req: Request, res: Response) {
  try {
    const conceptId = req.params.id as string;
    const { documentation } = req.body;

    if (typeof documentation !== "string") {
      res.status(400).json({ error: "documentation field is required and must be a string" });
      return;
    }

    const storage = new GraphStorage();
    await storage.initialize();

    const concept = await storage.getConcept(conceptId);
    if (!concept) {
      res.status(404).json({ error: "Concept not found" });
      return;
    }

    await storage.saveDocumentation(conceptId, documentation);

    res.json({ success: true, conceptId, documentation });
  } catch (error: any) {
    console.error("Error updating concept documentation:", error);
    res.status(500).json({ error: error.message || "Failed to update concept documentation" });
  }
}

/**
 * Delete a specific concept
 * DELETE /gitree/concepts/:id?repo=owner/repo (repo optional)
 */
export async function gitree_delete_concept(req: Request, res: Response) {
  try {
    const conceptId = req.params.id as string;
    const repo = parseRepoParam(req);
    const storage = new GraphStorage();
    await storage.initialize();

    const concept = await storage.getConcept(conceptId, repo);

    if (!concept) {
      res.status(404).json({ error: "Concept not found" });
      return;
    }

    await storage.deleteConcept(conceptId, repo);

    res.json({
      status: "success",
      message: `Concept ${conceptId} deleted`,
    });
  } catch (error: any) {
    console.error("Error deleting concept:", error);
    res
      .status(500)
      .json({ error: error.message || "Failed to delete concept" });
  }
}

/**
 * Get a specific PR
 * GET /gitree/prs/:number?repo=owner/repo (repo optional)
 */
export async function gitree_get_pr(req: Request, res: Response) {
  try {
    const prNumber = parseInt(req.params.number as string);
    const repo = parseRepoParam(req);
    const storage = new GraphStorage();
    await storage.initialize();

    const pr = await storage.getPR(prNumber, repo);

    if (!pr) {
      res.status(404).json({ error: "PR not found" });
      return;
    }

    const concepts = await storage.getConceptsForPR(prNumber, pr.repo);

    res.json({
      pr: {
        ref_id: pr.ref_id,
        number: pr.number,
        repo: pr.repo,
        title: pr.title,
        summary: pr.summary,
        mergedAt: pr.mergedAt.toISOString(),
        url: pr.url,
        files: pr.files,
        newDeclarations: pr.newDeclarations,
      },
      concepts: concepts.map((f) => ({
        id: f.id,
        name: f.name,
        ref_id: f.ref_id,
        description: f.description,
      })),
    });
  } catch (error: any) {
    console.error("Error getting PR:", error);
    res.status(500).json({ error: error.message || "Failed to get PR" });
  }
}

/**
 * Get a specific commit
 * GET /gitree/commits/:sha?repo=owner/repo (repo optional)
 */
export async function gitree_get_commit(req: Request, res: Response) {
  try {
    const sha = req.params.sha as string;
    const repo = parseRepoParam(req);
    const storage = new GraphStorage();
    await storage.initialize();

    const commit = await storage.getCommit(sha, repo);

    if (!commit) {
      res.status(404).json({ error: "Commit not found" });
      return;
    }

    const concepts = await storage.getConceptsForCommit(sha, commit.repo);

    res.json({
      commit: {
        sha: commit.sha,
        repo: commit.repo,
        message: commit.message,
        summary: commit.summary,
        author: commit.author,
        committedAt: commit.committedAt.toISOString(),
        url: commit.url,
        files: commit.files,
        newDeclarations: commit.newDeclarations,
      },
      concepts: concepts.map((f) => ({
        id: f.id,
        name: f.name,
        ref_id: f.ref_id,
        description: f.description,
      })),
    });
  } catch (error: any) {
    console.error("Error getting commit:", error);
    res.status(500).json({ error: error.message || "Failed to get commit" });
  }
}

/**
 * Get files for a specific concept
 * GET /gitree/concepts/:id/files?expand=contains,calls&output=text
 */
export async function gitree_get_concept_files(req: Request, res: Response) {
  try {
    const conceptId = req.params.id as string;
    const expandParam = req.query.expand as string | undefined;
    const outputFormat = req.query.output as string | undefined;
    const storage = new GraphStorage();
    await storage.initialize();

    // Parse expand parameter (comma-separated for future expansion)
    const expand = expandParam ? expandParam.split(",") : [];

    const files = await storage.getFilesForConcept(conceptId, expand);

    // Return text format if requested
    if (outputFormat === "text") {
      const textOutput = formatFilesAsText(files);
      res.setHeader("Content-Type", "text/plain");
      res.send(textOutput);
      return;
    }

    // Default JSON output
    res.json({ files });
  } catch (error: any) {
    console.error("Error getting concept files:", error);
    res
      .status(500)
      .json({ error: error.message || "Failed to get concept files" });
  }
}

/**
 * Format files array as text with nested children
 */
function formatFilesAsText(files: any[]): string {
  const lines: string[] = [];

  for (const file of files) {
    // File line
    lines.push(`**${file.file || file.name}**`);

    // Contained nodes (indented)
    if (file.contains && file.contains.length > 0) {
      for (const contained of file.contains) {
        const nodeType = contained.node_type ? ` (${contained.node_type})` : "";
        lines.push(`  - ${contained.name}${nodeType}`);
      }
    }

    // Called nodes (indented)
    if (file.calls && file.calls.length > 0) {
      for (const called of file.calls) {
        const nodeType = called.node_type ? ` (${called.node_type})` : "";
        lines.push(`  → ${called.name}${nodeType}`);
      }
    }

    // Empty line between files
    lines.push("");
  }

  return lines.join("\n");
}

/**
 * Get knowledge base statistics
 * GET /gitree/stats?repo=owner/repo (repo optional)
 */
export async function gitree_stats(req: Request, res: Response) {
  try {
    const repo = parseRepoParam(req);
    const storage = new GraphStorage();
    await storage.initialize();

    const concepts = await storage.getAllConcepts(repo);
    const prs = await storage.getAllPRs(repo);
    // Only get lastProcessed if repo is specified
    const lastProcessed = repo ? await storage.getLastProcessedPR(repo) : 0;

    const avgPRsPerConcept =
      concepts.length > 0
        ? concepts.reduce((sum, f) => sum + f.prNumbers.length, 0) /
          concepts.length
        : 0;

    const mostActive =
      concepts.length > 0
        ? concepts.reduce((max, f) =>
            f.prNumbers.length > max.prNumbers.length ? f : max
          )
        : null;

    res.json({
      repo: repo || "all",
      totalConcepts: concepts.length,
      totalPRs: prs.length,
      lastProcessedPR: lastProcessed,
      avgPRsPerConcept: parseFloat(avgPRsPerConcept.toFixed(1)),
      mostActiveConcept: mostActive
        ? {
            id: mostActive.id,
            name: mostActive.name,
            ref_id: mostActive.ref_id,
            prCount: mostActive.prNumbers.length,
          }
        : null,
    });
  } catch (error: any) {
    console.error("Error getting stats:", error);
    res.status(500).json({ error: error.message || "Failed to get stats" });
  }
}

/**
 * Summarize a specific concept
 * POST /gitree/summarize/:id
 * GET /progress?request_id=xxx
 */
export async function gitree_summarize_concept(req: Request, res: Response) {
  console.log("===> gitree_summarize_concept", req.path, req.method);
  const request_id = asyncReqs.startReq();
  try {
    const conceptId = req.params.id as string;

    // Summarize in background
    (async () => {
      try {
        const anthropicKey = getApiKeyForProvider("anthropic");
        const storage = new GraphStorage();
        await storage.initialize();

        const summarizer = new Summarizer(storage, "anthropic", anthropicKey);
        const usage = await summarizer.summarizeConcept(conceptId);

        asyncReqs.finishReq(request_id, {
          status: "success",
          message: `Summarized concept ${conceptId}`,
          usage,
        });
      } catch (error) {
        asyncReqs.failReq(request_id, error);
      }
    })();

    res.json({ request_id, status: "pending" });
  } catch (error) {
    console.log("===> error");
    asyncReqs.failReq(request_id, error);
    res.status(500).json({ error: "Failed to summarize concept" });
  }
}

/**
 * Summarize all concepts
 * POST /gitree/summarize-all
 * GET /progress?request_id=xxx
 */
export async function gitree_summarize_all(req: Request, res: Response) {
  console.log("===> gitree_summarize_all", req.path, req.method);
  const request_id = asyncReqs.startReq();
  try {
    // Summarize in background
    (async () => {
      try {
        const anthropicKey = getApiKeyForProvider("anthropic");
        const storage = new GraphStorage();
        await storage.initialize();

        const summarizer = new Summarizer(storage, "anthropic", anthropicKey);
        const usage = await summarizer.summarizeAllConcepts();

        asyncReqs.finishReq(request_id, {
          status: "success",
          message: "Summarized all concepts",
          usage,
        });
      } catch (error) {
        asyncReqs.failReq(request_id, error);
      }
    })();

    res.json({ request_id, status: "pending" });
  } catch (error) {
    console.log("===> error");
    asyncReqs.failReq(request_id, error);
    res.status(500).json({ error: "Failed to summarize all concepts" });
  }
}

/**
 * Link concepts to file nodes in the graph
 * POST /gitree/link-files?concept_id=xxx (optional concept_id)
 * GET /progress?request_id=xxx
 */
export async function gitree_link_files(req: Request, res: Response) {
  console.log("===> gitree_link_files", req.path, req.method);
  const request_id = asyncReqs.startReq();
  try {
    const conceptId = req.query.concept_id as string | undefined;

    // Link in background
    (async () => {
      try {
        const storage = new GraphStorage();
        await storage.initialize();

        const linker = new FileLinker(storage);

        let result;
        if (conceptId) {
          result = await linker.linkConcept(conceptId);
        } else {
          result = await linker.linkAllConcepts();
        }

        asyncReqs.finishReq(request_id, {
          status: "success",
          message: conceptId
            ? `Linked files for concept ${conceptId}`
            : "Linked files for all concepts",
          result,
        });
      } catch (error) {
        asyncReqs.failReq(request_id, error);
      }
    })();

    res.json({ request_id, status: "pending" });
  } catch (error) {
    console.log("===> error");
    asyncReqs.failReq(request_id, error);
    res.status(500).json({ error: "Failed to link files" });
  }
}

/**
 * Convert node to concise format with only name, file, ref_id, and node_type
 */
function toConciseNode(node: Neo4jNode): {
  name: string;
  file: string;
  ref_id: string;
  node_type: string;
} {
  const ref_id = IS_TEST ? "test_ref_id" : node.properties.ref_id || "";
  const node_type = node.labels.find((l: string) => l !== "Data_Bank") || "";
  return {
    name: node.properties.name || "",
    file: node.properties.file || "",
    ref_id,
    node_type,
  };
}

/**
 * Get all concepts with files and contained nodes as a flat graph structure
 * GET /gitree/all-concepts-graph?limit=100&node_types=Function,Class&concise=true&depth=2&per_type_limits=Function:50,Class:25&repo=owner/repo
 */
export async function gitree_all_concepts_graph(req: Request, res: Response) {
  try {
    const repo = parseRepoParam(req);
    const storage = new GraphStorage();
    await storage.initialize();

    // Get all data from GraphStorage (optionally filtered by repo)
    const data = await storage.getAllConceptsWithFilesAndContains(repo);

    // Parse query parameters
    const limit = parseLimit(req.query);
    const requestedNodeTypes = parseNodeTypes(req.query);
    const concise = isTrue(req.query.concise as string);
    const no_body = isTrue(req.query.no_body as string);
    const depth = parseInt(req.query.depth as string) || undefined;
    const perTypeLimitsParam = req.query.per_type_limits as string;

    // Parse per-type limits (e.g., "Function:50,Class:25")
    const perTypeLimits: { [nodeType: string]: number } = {};
    if (perTypeLimitsParam) {
      const pairs = perTypeLimitsParam.split(",");
      for (const pair of pairs) {
        const [nodeType, limitStr] = pair.trim().split(":");
        if (nodeType && limitStr) {
          const typeLimit = parseInt(limitStr);
          if (!isNaN(typeLimit)) {
            perTypeLimits[nodeType] = typeLimit;
          }
        }
      }
    }

    // Combine all nodes
    let allNodes = [...data.concepts, ...data.files, ...data.containedNodes];

    // Filter by node types if specified
    if (requestedNodeTypes.length > 0) {
      allNodes = allNodes.filter((node) => {
        const nodeType = node.labels.find((l: string) => l !== "Data_Bank");
        return (
          nodeType === "Concept" ||
          nodeType === "File" ||
          requestedNodeTypes.includes(nodeType)
        );
      });
    }

    // Apply per-type limits if specified
    if (Object.keys(perTypeLimits).length > 0) {
      const nodesByType: { [nodeType: string]: any[] } = {};

      // Group nodes by type
      allNodes.forEach((node) => {
        const nodeType =
          node.labels.find((l: string) => l !== "Data_Bank") || "Unknown";
        if (!nodesByType[nodeType]) {
          nodesByType[nodeType] = [];
        }
        nodesByType[nodeType].push(node);
      });

      // Apply limits per type
      allNodes = [];
      for (const [nodeType, nodes] of Object.entries(nodesByType)) {
        const typeLimit = perTypeLimits[nodeType];
        if (typeLimit && nodes.length > typeLimit) {
          allNodes.push(...nodes.slice(0, typeLimit));
        } else {
          allNodes.push(...nodes);
        }
      }
    }

    // Depth filtering: 1=concepts only, 2=+files+MODIFIES, 3(default)=+containedNodes+CONTAINS
    if (depth === 1) {
      data.files = [];
      data.containedNodes = [];
      data.modifiesEdges = [];
      data.containsEdges = [];
      allNodes = [...data.concepts];
    } else if (depth === 2) {
      data.containedNodes = [];
      data.containsEdges = [];
      allNodes = [...data.concepts, ...data.files];
    }

    // Apply global limit if specified (after per-type limits)
    if (limit) {
      allNodes = allNodes.slice(0, limit);
    }

    // Transform nodes to return format
    const mapNode = concise ? toConciseNode : no_body ? toReturnNodeNoBody : toReturnNode;
    const returnNodes = allNodes.map((node) => mapNode(node));

    // Combine all edges
    let allEdges = [...data.modifiesEdges, ...data.containsEdges];

    // Transform edges to concise format if requested
    if (concise) {
      allEdges = allEdges.map((e) => ({
        edge_type: e.edge_type,
        source: e.source,
        target: e.target,
      }));
    }

    // Build meta information
    const nodeTypes = Array.from(
      new Set(
        allNodes
          .map((n) => {
            const label = n.labels.find((l: string) => l !== "Data_Bank");
            return label;
          })
          .filter(Boolean)
      )
    ) as NodeType[];

    const meta = buildGraphMeta(nodeTypes, allNodes, limit, "total", undefined);

    res.json({
      nodes: returnNodes,
      edges: allEdges,
      status: "Success",
      meta,
    });
  } catch (error: any) {
    console.error("Error getting all concepts graph:", error);
    res.status(500).json({
      error: error.message || "Failed to get all concepts graph",
    });
  }
}

/**
 * Get relevant concepts for a given prompt using AI
 * POST /gitree/relevant-concepts
 * Body: { prompt: "user's question or description" }
 curl -X POST http://localhost:3355/gitree/relevant-concepts \
    -H "Content-Type: application/json" \
    -d '{
      "prompt": "How does authentication work in this repo?"
    }'
 */
export async function gitree_relevant_concepts(req: Request, res: Response) {
  try {
    const { prompt } = req.body;
    const repo = parseRepoParam(req);

    if (!prompt) {
      res.status(400).json({ error: "Missing prompt in request body" });
      return;
    }

    const storage = new GraphStorage();
    await storage.initialize();

    // Get all concepts without documentation for the initial AI call
    const allConcepts = await storage.getAllConcepts(repo);
    const conceptsWithoutDocs = allConcepts.map((f) => ({
      id: f.id,
      name: f.name,
      description: f.description,
      prCount: f.prNumbers.length,
      commitCount: (f.commitShas || []).length,
    }));

    if (conceptsWithoutDocs.length === 0) {
      res.status(400).json({ error: "No concepts found" });
      return;
    }

    // Use AI to determine relevant concepts
    const provider = process.env.LLM_PROVIDER || "anthropic";
    const apiKey = getApiKeyForProvider(provider);
    const typedProvider = provider as Provider;
    const model = await getModel(typedProvider, apiKey as string);

    const aiPrompt = `<prompt>${prompt}</prompt>
<concepts>${JSON.stringify(conceptsWithoutDocs, null, 2)}</concepts>

Please analyze the user's prompt and the list of available concepts. Return an array of concept IDs that are relevant to the user's prompt. Only include concepts that are directly related to what the user is asking about. Usually 1 concept id is enough, but you can include up to 3.`;

    const schema = {
      type: "object" as const,
      properties: {
        relevantConceptIds: {
          type: "array" as const,
          items: { type: "string" as const },
          description:
            "Array of concept IDs that are relevant to the user's prompt",
        },
      },
      required: ["relevantConceptIds"],
      additionalProperties: false,
    };
    const result = await generateObject({
      model,
      prompt: aiPrompt,
      schema: jsonSchema(schema),
      providerOptions: getProviderOptions(typedProvider, "fast") as any,
    });

    const relevantConceptIds =
      (result.object as any).relevantConceptIds?.slice(0, 3) || [];

    // Fetch full concepts WITH documentation
    const relevantConcepts = await Promise.all(
      relevantConceptIds.map(async (conceptId: string) => {
        const concept = await storage.getConcept(conceptId, repo);
        if (!concept) return null;
        return formatConceptWithDetails(concept, storage);
      })
    );

    // Filter out any null values (concepts that weren't found)
    const validConcepts = relevantConcepts.filter((f) => f !== null);

    // Concatenate all documentation fields
    const concatenatedDocumentation = validConcepts
      .map((f) => f.concept.documentation)
      .filter((doc) => doc) // Filter out undefined/empty docs
      .join("\n\n---\n\n");

    res.json({
      concepts: validConcepts,
      total: validConcepts.length,
      prompt,
      documentation: concatenatedDocumentation,
      conceptIds: relevantConceptIds,
      refIds: relevantConcepts.map((f) => f.concept.ref_id),
    });
  } catch (error: any) {
    console.error("Error getting relevant concepts:", error);
    res.status(500).json({
      error: error.message || "Failed to get relevant concepts",
    });
  }
}

/**
 * Create a new concept with documentation from get_context
 * POST /gitree/create-concept
 * Body: { prompt: string, name: string, owner: string, repo: string, pat?: string }
 * GET /progress?request_id=xxx
 curl -X POST http://localhost:3355/gitree/create-concept \
    -H "Content-Type: application/json" \
    -d '{
      "prompt": "How does authentication work in this repo?",
      "name": "Authentication System",
      "owner": "stakwork",
      "repo": "hive"
    }'
 */
export async function gitree_create_concept(req: Request, res: Response) {
  console.log("===> gitree_create_concept", req.path, req.method);
  const request_id = asyncReqs.startReq();
  try {
    const { prompt, name, owner, repo, pat } = req.body;

    if (!prompt || !name || !owner || !repo) {
      asyncReqs.failReq(
        request_id,
        new Error("Missing required fields: prompt, name, owner, repo")
      );
      res.status(400).json({
        error: "Missing required fields: prompt, name, owner, repo",
      });
      return;
    }

    // Process in background
    (async () => {
      try {
        // Clone or update the repository
        const repoDir = await cloneOrUpdateRepo(
          `https://github.com/${owner}/${repo}`,
          undefined,
          pat
        );
        console.log(`===> Repository cloned/updated at ${repoDir}`);

        // Call get_context to generate documentation
        const sessionId = randomUUID();
        const contextResult = await get_context(prompt, repoDir, {
          pat,
          sessionId,
          source: "gitree_create_concept",
        });
        const documentation = contextResult.content;

        // Generate concept ID from name with repo prefix
        const slug = generateSlug(name);
        const repoId = `${owner}/${repo}`;
        const conceptId = makeRepoId(repoId, slug);

        // Create the concept object
        const concept: Concept = {
          id: conceptId,
          repo: repoId,
          name: name,
          description: prompt,
          prNumbers: [],
          commitShas: [],
          createdAt: new Date(),
          lastUpdated: new Date(),
          documentation: documentation,
        };

        // Save to storage
        const storage = new GraphStorage();
        await storage.initialize();
        await storage.saveConcept(concept);
        await storage.saveDocumentation(conceptId, documentation);

        console.log(`✅ Concept created: ${conceptId}`);

        asyncReqs.finishReq(request_id, {
          status: "success",
          message: `Created concept ${conceptId}`,
          concept: {
            id: concept.id,
            name: concept.name,
            description: concept.description,
            documentation: concept.documentation,
          },
          usage: contextResult.usage,
        });
      } catch (error) {
        asyncReqs.failReq(request_id, error);
      }
    })();

    res.json({ request_id, status: "pending" });
  } catch (error) {
    console.log("===> error");
    asyncReqs.failReq(request_id, error);
    res.status(500).json({ error: "Failed to create concept" });
  }
}

/**
 * POST /gitree/analyze-clues
 * Analyze a specific concept or all concepts for clues
 */
export async function gitree_analyze_clues(req: Request, res: Response) {
  const request_id = asyncReqs.startReq();

  try {
    const owner = req.query.owner as string;
    const repo = req.query.repo as string;
    const conceptId = req.query.concept_id as string | undefined;
    const force = req.query.force === "true";
    const autoLink = req.query.auto_link !== "false"; // Default true
    const pat = req.query.token as string | undefined;

    if (!owner || !repo) {
      asyncReqs.failReq(request_id, new Error("owner and repo are required"));
      return res.status(400).json({ error: "owner and repo are required" });
    }

    (async () => {
      try {
        // Clone or update repository
        const repoPath = await cloneOrUpdateRepo(
          `https://github.com/${owner}/${repo}`,
          undefined,
          pat
        );

        const storage = new GraphStorage();
        await storage.initialize();

        const analyzer = new ClueAnalyzer(storage, repoPath);

        let result;
        if (conceptId) {
          result = await analyzer.analyzeConcept(conceptId);

          // Auto-link after single concept analysis
          if (autoLink && result.clues.length > 0) {
            console.log(`\n🔗 Auto-linking new clues to relevant concepts...\n`);
            try {
              const { ClueLinker } = await import("./clueLinker.js");
              const linker = new ClueLinker(storage);
              const newClueIds = result.clues.map((c: any) => c.id);
              const linkUsage = await linker.linkClues(newClueIds);

              result.usage = normalizeUsage(addUsage(result.usage, linkUsage));
            } catch (error) {
              console.error(`\n⚠️  Auto-linking failed:`, error instanceof Error ? error.message : error);
            }
          }
        } else {
          const usage = await analyzer.analyzeAllConcepts(force, autoLink);
          result = { usage, message: "Analyzed all concepts" + (autoLink ? " and linked clues" : "") };
        }

        asyncReqs.finishReq(request_id, { ...result });
      } catch (error) {
        asyncReqs.failReq(request_id, error);
      }
    })();

    res.json({
      request_id,
      status: "pending",
      message: conceptId
        ? `Analyzing concept ${conceptId} for clues...`
        : "Analyzing all concepts for clues...",
    });
  } catch (error) {
    console.error("Error in gitree_analyze_clues:", error);
    asyncReqs.failReq(request_id, error);
    res.status(500).json({
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
}

/**
 * POST /gitree/analyze-changes
 * Retroactively analyze historical PRs/commits for clues
 * curl -X POST "http://localhost:3355/gitree/analyze-changes?owner=stakwork&repo=hive&force=false"
 */
export async function gitree_analyze_changes(req: Request, res: Response) {
  const request_id = asyncReqs.startReq();

  try {
    const owner = req.query.owner as string;
    const repo = req.query.repo as string;
    const force = req.query.force === "true";
    const pat = req.query.token as string | undefined;

    if (!owner || !repo) {
      asyncReqs.failReq(request_id, new Error("owner and repo are required"));
      return res.status(400).json({ error: "owner and repo are required" });
    }

    (async () => {
      try {
        // Clone or update repository
        const repoPath = await cloneOrUpdateRepo(
          `https://github.com/${owner}/${repo}`,
          undefined,
          pat
        );

        const storage = new GraphStorage();
        await storage.initialize();

        const analyzer = new ClueAnalyzer(storage, repoPath);

        // Get checkpoint (unless force)
        const repoId = `${owner}/${repo}`;
        const checkpoint = force ? null : await storage.getClueAnalysisCheckpoint(repoId);

        console.log(
          checkpoint
            ? `📌 Resuming from checkpoint: ${checkpoint.lastProcessedTimestamp}`
            : force
            ? "🔄 Force mode: analyzing all changes"
            : "🆕 No checkpoint found, starting from beginning"
        );

        // Fetch all PRs and commits
        const allPRs = await storage.getAllPRs();
        const allCommits = await storage.getAllCommits();

        // Combine and sort chronologically
        const changes: Array<{
          type: "pr" | "commit";
          date: Date;
          id: string;
          data: any;
        }> = [];

        for (const pr of allPRs) {
          changes.push({
            type: "pr",
            date: pr.mergedAt,
            id: pr.number.toString(),
            data: pr,
          });
        }

        for (const commit of allCommits) {
          changes.push({
            type: "commit",
            date: commit.committedAt,
            id: commit.sha,
            data: commit,
          });
        }

        // Sort chronologically (oldest first)
        changes.sort((a, b) => a.date.getTime() - b.date.getTime());

        // Filter by checkpoint
        let changesToProcess = changes;
        if (checkpoint) {
          changesToProcess = changes.filter((change) => {
            const changeTime = change.date.toISOString();
            if (changeTime > checkpoint.lastProcessedTimestamp) {
              return true;
            }
            if (changeTime === checkpoint.lastProcessedTimestamp) {
              // Skip if already processed at this exact timestamp
              return !checkpoint.processedAtTimestamp.includes(change.id);
            }
            return false;
          });
        }
        
        if (changesToProcess.length === 0) {
          console.log("✅ No new changes to analyze!");
          asyncReqs.finishReq(request_id, {
            status: "success",
            message: "No new changes to analyze",
            totalClues: 0,
            totalChanges: 0,
            usage: normalizeUsage(),
          });
          return;
        }

        console.log(`📊 Analyzing ${changesToProcess.length} change(s) for clues...`);

        let totalClues = 0;
        let totalUsage: Usage = normalizeUsage();

        for (let i = 0; i < changesToProcess.length; i++) {
          const change = changesToProcess[i];
          const progress = `[${i + 1}/${changesToProcess.length}]`;

          if (change.type === "pr") {
            const pr = change.data;
            console.log(`${progress} PR #${pr.number}: ${pr.title}`);

            try {
              const changeContext = {
                type: "pr" as const,
                identifier: `#${pr.number}`,
                title: pr.title,
                summary: pr.summary,
                files: pr.files,
              };

              const result = await analyzer.analyzeChange(changeContext);
              totalClues += result.clues.length;
              totalUsage = normalizeUsage(addUsage(totalUsage, result.usage));

              // Link clues if any were created
              if (result.clues.length > 0) {
                const { ClueLinker } = await import("./clueLinker.js");
                const linker = new ClueLinker(storage);
                await linker.linkClues(result.clues.map((c) => c.id));
              }

              // Update checkpoint
              await storage.setClueAnalysisCheckpoint(repoId, {
                lastProcessedTimestamp: pr.mergedAt.toISOString(),
                processedAtTimestamp: [pr.number.toString()],
              });
            } catch (error) {
              console.error(
                `   ❌ Error:`,
                error instanceof Error ? error.message : error
              );
              console.log(`   ⏭️  Skipping...`);
            }
          } else {
            const commit = change.data;
            console.log(
              `${progress} Commit ${commit.sha.substring(0, 7)}: ${
                commit.message.split("\n")[0]
              }`
            );

            try {
              const changeContext = {
                type: "commit" as const,
                identifier: commit.sha.substring(0, 7),
                title: commit.message.split("\n")[0],
                summary: commit.summary,
                files: commit.files,
              };

              const result = await analyzer.analyzeChange(changeContext);
              totalClues += result.clues.length;
              totalUsage = normalizeUsage(addUsage(totalUsage, result.usage));

              // Link clues if any were created
              if (result.clues.length > 0) {
                const { ClueLinker } = await import("./clueLinker.js");
                const linker = new ClueLinker(storage);
                await linker.linkClues(result.clues.map((c) => c.id));
              }

              // Update checkpoint
              await storage.setClueAnalysisCheckpoint(repoId, {
                lastProcessedTimestamp: commit.committedAt.toISOString(),
                processedAtTimestamp: [commit.sha],
              });
            } catch (error) {
              console.error(
                `   ❌ Error:`,
                error instanceof Error ? error.message : error
              );
              console.log(`   ⏭️  Skipping...`);
            }
          }
        }

        console.log(`\n Analysis complete!`);
        console.log(`   Total clues created: ${totalClues}`);
        console.log(`   Total token usage: ${totalUsage.totalTokens.toLocaleString()}`);

        asyncReqs.finishReq(request_id, {
          status: "success",
          message: `Analyzed ${changesToProcess.length} change(s) and created ${totalClues} clue(s)`,
          totalClues,
          totalChanges: changesToProcess.length,
          usage: totalUsage,
        });
      } catch (error) {
        console.error("Error in gitree_analyze_changes:", error);
        asyncReqs.failReq(request_id, error);
      }
    })();

    res.json({ request_id, status: "pending" });
  } catch (error) {
    console.error("Error in gitree_analyze_changes:", error);
    asyncReqs.failReq(request_id, error);
    res.status(500).json({
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
}

/**
 * GET /gitree/clues?concept_id=xxx&repo=owner/repo (both optional)
 * List all clues or clues for a specific concept
 */
export async function gitree_list_clues(req: Request, res: Response) {
  try {
    const conceptId = req.query.concept_id as string | undefined;
    const repo = parseRepoParam(req);

    const storage = new GraphStorage();
    await storage.initialize();

    const clues = conceptId
      ? await storage.getCluesForConcept(conceptId, undefined, repo)
      : await storage.getAllClues(repo);

    res.json({
      clues,
      count: clues.length,
      repo: repo || "all",
    });
  } catch (error) {
    console.error("Error in gitree_list_clues:", error);
    res.status(500).json({
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
}

/**
 * GET /gitree/clues/:id?repo=owner/repo (repo optional)
 * Get a specific clue by ID
 */
export async function gitree_get_clue(req: Request, res: Response) {
  try {
    const clueId = req.params.id as string;
    const repo = parseRepoParam(req);

    const storage = new GraphStorage();
    await storage.initialize();

    const clue = await storage.getClue(clueId, repo);

    if (!clue) {
      return res.status(404).json({ error: "Clue not found" });
    }

    res.json({ clue });
  } catch (error) {
    console.error("Error in gitree_get_clue:", error);
    res.status(500).json({
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
}

/**
 * DELETE /gitree/clues/:id?repo=owner/repo (repo optional)
 * Delete a specific clue
 */
export async function gitree_delete_clue(req: Request, res: Response) {
  try {
    const clueId = req.params.id as string;
    const repo = parseRepoParam(req);

    const storage = new GraphStorage();
    await storage.initialize();

    await storage.deleteClue(clueId, repo);

    res.json({ message: "Clue deleted successfully" });
  } catch (error) {
    console.error("Error in gitree_delete_clue:", error);
    res.status(500).json({
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
}

/**
 * POST /gitree/link-clues
 * Link clues to relevant concepts (Step 2 of clue analysis)
 */
export async function gitree_link_clues(req: Request, res: Response) {
  try {
    const owner = req.query.owner as string;
    const repo = req.query.repo as string;
    const force = req.query.force === "true";

    if (!owner || !repo) {
      return res.status(400).json({ error: "owner and repo are required" });
    }

    const storage = new GraphStorage();
    await storage.initialize();

    const { ClueLinker } = await import("./clueLinker.js");
    const linker = new ClueLinker(storage);

    // Start async processing
    const request_id = asyncReqs.startReq();

    (async () => {
      try {
        const usage = await linker.linkAllClues(force);
        asyncReqs.finishReq(request_id, { usage, message: "Linked all clues" });
      } catch (error) {
        asyncReqs.failReq(request_id, error);
      }
    })();

    res.json({
      request_id,
      status: "pending",
      message: "Linking clues to relevant concepts...",
    });
  } catch (error) {
    console.error("Error in gitree_link_clues:", error);
    res.status(500).json({
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
}

/**
 * POST /gitree/search-clues
 * Search clues by relevance using embeddings, keywords, and centrality
 * Body: { query: string, conceptId?: string, limit?: number, similarityThreshold?: number, repo?: string }
 curl -X POST http://localhost:3355/gitree/search-clues \
    -H "Content-Type: application/json" \
    -d '{"query": "workspace access"}'
 */
export async function gitree_search_clues(req: Request, res: Response) {
  try {
    const { query, conceptId, limit = 10, similarityThreshold = 0.5, repo: bodyRepo } = req.body;
    // Support repo from both query param and body
    const repo = parseRepoParam(req) || bodyRepo;

    if (!query || typeof query !== "string") {
      return res.status(400).json({ error: "query is required and must be a string" });
    }

    const storage = new GraphStorage();
    await storage.initialize();

    // Generate embeddings for the query
    const { vectorizeQuery } = await import("../vector/index.js");
    const embeddings = await vectorizeQuery(query);

    // Search clues
    const results = await storage.searchClues(
      query,
      embeddings,
      conceptId,
      limit,
      similarityThreshold,
      repo
    );

    res.json({
      query,
      conceptId: conceptId || "all",
      repo: repo || "all",
      count: results.length,
      results: results.map((r) => ({
        clue: {
          id: r.id,
          repo: r.repo,
          conceptId: r.conceptId,
          type: r.type,
          title: r.title,
          content: r.content,
          entities: r.entities,
          files: r.files,
          keywords: r.keywords,
          centrality: r.centrality,
        },
        score: r.score,
        relevanceBreakdown: r.relevanceBreakdown,
      })),
    });
  } catch (error) {
    console.error("Error in gitree_search_clues:", error);
    res.status(500).json({
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
}

/**
 * Get provenance data for concepts
 * POST /gitree/provenance
 * Body: { conceptIds: string[] } (array of concept IDs)
 *
 * Returns hierarchical structure: Concepts → Files → Code Entities
 * Code entities are filtered by text matching against concept documentation
 *
 * Example:
 * curl -X POST http://localhost:3355/gitree/provenance \
 *   -H "Content-Type: application/json" \
 *   -d '{"conceptIds": ["auth-system", "api-endpoints"]}'
 */
export async function gitree_provenance(req: Request, res: Response) {
  try {
    const { conceptIds } = req.body;

    // Validate request body
    if (!conceptIds || !Array.isArray(conceptIds)) {
      res.status(400).json({
        error:
          "Missing or invalid conceptIds in request body. Expected array of strings.",
      });
      return;
    }

    if (conceptIds.length === 0) {
      res.json({ concepts: [] });
      return;
    }

    // Initialize storage
    const storage = new GraphStorage();
    await storage.initialize();

    // Get provenance data from storage
    const provenanceData = await storage.getProvenanceForConcepts(conceptIds);

    // Transform to response format
    const concepts = provenanceData.map((concept) => ({
      id: concept.conceptId,
      name: concept.name,
      description: concept.description,
      files: concept.files.map((file) => ({
        refId: file.refId,
        name: file.name,
        path: file.path,
        codeEntities: file.entities.map((entity) => ({
          refId: entity.refId,
          name: entity.name,
          nodeType: entity.nodeType as
            | "Function"
            | "Page"
            | "Endpoint"
            | "Datamodel"
            | "UnitTest"
            | "IntegrationTest"
            | "E2etest",
          file: entity.file,
          start: entity.start,
          end: entity.end,
        })),
      })),
    }));

    res.json({ concepts });
  } catch (error: any) {
    console.error("Error getting provenance:", error);
    res.status(500).json({
      error: error.message || "Failed to get provenance data",
    });
  }
}
