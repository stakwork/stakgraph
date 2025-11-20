import { Request, Response } from "express";
import * as asyncReqs from "../graph/reqs.js";
import { GraphStorage } from "./store/index.js";
import { LLMClient } from "./llm.js";
import { StreamingFeatureBuilder } from "./builder.js";
import { Summarizer } from "./summarizer.js";
import { FileLinker } from "./fileLinker.js";
import { Octokit } from "@octokit/rest";
import { getApiKeyForProvider } from "../aieo/src/provider.js";
import {
  toReturnNode,
  parseNodeTypes,
  parseLimit,
  buildGraphMeta,
  isTrue,
  IS_TEST,
} from "../graph/utils.js";
import { NodeType, Neo4jNode } from "../graph/types.js";

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
    // Remove trailing .git if present
    let cleanUrl = url.replace(/\.git$/, "");

    // Handle SSH format (git@host:owner/repo)
    const sshMatch = cleanUrl.match(/git@[^:]+:(.+)/);
    if (sshMatch) {
      cleanUrl = sshMatch[1];
    }

    // Remove protocol and domain (https://, http://, etc.)
    cleanUrl = cleanUrl.replace(/^https?:\/\//, "");
    cleanUrl = cleanUrl.replace(/^[^\/]+\//, ""); // Remove domain/host

    // Extract the last two path segments (owner/repo)
    const pathParts = cleanUrl.split("/").filter((p) => p.length > 0);

    if (pathParts.length >= 2) {
      // Take the last two segments as owner and repo
      const owner = pathParts[pathParts.length - 2];
      const repo = pathParts[pathParts.length - 1];
      return { owner, repo };
    }

    // Try simple owner/repo format
    const simpleMatch = cleanUrl.match(/^([^\/]+)\/([^\/]+)$/);
    if (simpleMatch) {
      return { owner: simpleMatch[1], repo: simpleMatch[2] };
    }

    return null;
  } catch (error) {
    return null;
  }
}

/**
 * Process a Git repository to extract features (PRs and commits)
 * POST /gitree/process?owner=stakwork&repo=sphinx-tribes&token=...
 * POST /gitree/process?repo_url=https://github.com/stakwork/sphinx-tribes&token=...
 * POST /gitree/process?repo_url=https://gitlab.com/owner/repo&token=...
 * POST /gitree/process?repo_url=git@github.com:owner/repo.git&token=...
 * POST /gitree/process?repo_url=...&token=...&summarize=true&link=true
 * GET /progress?request_id=xxx
 */

// curl -X POST "http://localhost:3355/gitree/process?owner=stakwork&repo=hive&summarize=true&link=true"
export async function gitree_process(req: Request, res: Response) {
  console.log("===> gitree_process", req.url, req.method);
  const request_id = asyncReqs.startReq();
  try {
    let owner = req.query.owner as string;
    let repo = req.query.repo as string;
    const repoUrl = req.query.repo_url as string;
    const githubTokenQuery = req.query.token as string;
    const shouldSummarize = req.query.summarize === "true";
    const shouldLink = req.query.link === "true";
    const githubToken = githubTokenQuery || process.env.GITHUB_TOKEN;

    // Parse repo_url if provided
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

    // Process repository in background
    (async () => {
      try {
        const anthropicKey = getApiKeyForProvider("anthropic");
        const storage = new GraphStorage();
        await storage.initialize();

        const octokit = new Octokit({ auth: githubToken });
        const llm = new LLMClient("anthropic", anthropicKey);
        const builder = new StreamingFeatureBuilder(storage, llm, octokit);

        const processUsage = await builder.processRepo(owner, repo);

        let summarizeUsage = null;
        let linkResult = null;

        // If summarize flag is set, run summarization after processing
        if (shouldSummarize) {
          console.log("===> Starting feature summarization...");
          const summarizer = new Summarizer(storage, "anthropic", anthropicKey);
          summarizeUsage = await summarizer.summarizeAllFeatures();
        }

        // If link flag is set, link files to features
        if (shouldLink) {
          console.log("===> Starting feature-file linking...");
          const linker = new FileLinker(storage);
          linkResult = await linker.linkAllFeatures();
        }

        // Build response message and usage
        const messageParts = [`Processed ${owner}/${repo}`];
        if (shouldSummarize) messageParts.push("summarized");
        if (shouldLink) messageParts.push("linked files");

        const totalUsage = {
          inputTokens:
            processUsage.inputTokens + (summarizeUsage?.inputTokens || 0),
          outputTokens:
            processUsage.outputTokens + (summarizeUsage?.outputTokens || 0),
          totalTokens:
            processUsage.totalTokens + (summarizeUsage?.totalTokens || 0),
        };

        const result: any = {
          status: "success",
          message: messageParts.join(", "),
          usage: totalUsage,
        };

        if (linkResult) {
          result.linkResult = linkResult;
        }

        asyncReqs.finishReq(request_id, result);
      } catch (error) {
        asyncReqs.failReq(request_id, error);
      }
    })();

    res.json({ request_id, status: "pending" });
  } catch (error) {
    console.log("===> error", error);
    asyncReqs.failReq(request_id, error);
    res.status(500).json({ error: "Failed to process repository" });
  }
}

/**
 * List all features
 * GET /gitree/features
 */
export async function gitree_list_features(_req: Request, res: Response) {
  try {
    const storage = new GraphStorage();
    await storage.initialize();

    const features = await storage.getAllFeatures();
    res.json({
      features: features.map((f) => ({
        id: f.id,
        name: f.name,
        description: f.description,
        prCount: f.prNumbers.length,
        commitCount: (f.commitShas || []).length,
        lastUpdated: f.lastUpdated.toISOString(),
        hasDocumentation: !!f.documentation,
      })),
      total: features.length,
    });
  } catch (error: any) {
    console.error("Error listing features:", error);
    res.status(500).json({ error: error.message || "Failed to list features" });
  }
}

/**
 * Get a specific feature
 * GET /gitree/features/:id?include=files
 */
export async function gitree_get_feature(req: Request, res: Response) {
  try {
    const featureId = req.params.id;
    const include = req.query.include as string | undefined;
    const storage = new GraphStorage();
    await storage.initialize();

    const feature = await storage.getFeature(featureId);

    if (!feature) {
      res.status(404).json({ error: "Feature not found" });
      return;
    }

    const prs = await storage.getPRsForFeature(featureId);
    const commits = await storage.getCommitsForFeature(featureId);

    const response: any = {
      feature: {
        id: feature.id,
        name: feature.name,
        description: feature.description,
        prNumbers: feature.prNumbers,
        commitShas: feature.commitShas || [],
        createdAt: feature.createdAt.toISOString(),
        lastUpdated: feature.lastUpdated.toISOString(),
        documentation: feature.documentation,
      },
      prs: prs.map((pr) => ({
        number: pr.number,
        title: pr.title,
        summary: pr.summary,
        mergedAt: pr.mergedAt.toISOString(),
        url: pr.url,
      })),
      commits: commits.map((commit) => ({
        sha: commit.sha,
        message: commit.message.split('\n')[0],
        summary: commit.summary,
        author: commit.author,
        committedAt: commit.committedAt.toISOString(),
        url: commit.url,
      })),
    };

    // Include files if requested
    if (include === "files") {
      const files = await storage.getFilesForFeature(featureId);
      response.files = files;
    }

    res.json(response);
  } catch (error: any) {
    console.error("Error getting feature:", error);
    res.status(500).json({ error: error.message || "Failed to get feature" });
  }
}

/**
 * Get a specific PR
 * GET /gitree/prs/:number
 */
export async function gitree_get_pr(req: Request, res: Response) {
  try {
    const prNumber = parseInt(req.params.number);
    const storage = new GraphStorage();
    await storage.initialize();

    const pr = await storage.getPR(prNumber);

    if (!pr) {
      res.status(404).json({ error: "PR not found" });
      return;
    }

    const features = await storage.getFeaturesForPR(prNumber);

    res.json({
      pr: {
        number: pr.number,
        title: pr.title,
        summary: pr.summary,
        mergedAt: pr.mergedAt.toISOString(),
        url: pr.url,
        files: pr.files,
        newDeclarations: pr.newDeclarations,
      },
      features: features.map((f) => ({
        id: f.id,
        name: f.name,
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
 * GET /gitree/commits/:sha
 */
export async function gitree_get_commit(req: Request, res: Response) {
  try {
    const sha = req.params.sha;
    const storage = new GraphStorage();
    await storage.initialize();

    const commit = await storage.getCommit(sha);

    if (!commit) {
      res.status(404).json({ error: "Commit not found" });
      return;
    }

    const features = await storage.getFeaturesForCommit(sha);

    res.json({
      commit: {
        sha: commit.sha,
        message: commit.message,
        summary: commit.summary,
        author: commit.author,
        committedAt: commit.committedAt.toISOString(),
        url: commit.url,
        files: commit.files,
        newDeclarations: commit.newDeclarations,
      },
      features: features.map((f) => ({
        id: f.id,
        name: f.name,
        description: f.description,
      })),
    });
  } catch (error: any) {
    console.error("Error getting commit:", error);
    res.status(500).json({ error: error.message || "Failed to get commit" });
  }
}

/**
 * Get files for a specific feature
 * GET /gitree/features/:id/files?expand=contains,calls&output=text
 */
export async function gitree_get_feature_files(req: Request, res: Response) {
  try {
    const featureId = req.params.id;
    const expandParam = req.query.expand as string | undefined;
    const outputFormat = req.query.output as string | undefined;
    const storage = new GraphStorage();
    await storage.initialize();

    // Parse expand parameter (comma-separated for future expansion)
    const expand = expandParam ? expandParam.split(",") : [];

    const files = await storage.getFilesForFeature(featureId, expand);

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
    console.error("Error getting feature files:", error);
    res
      .status(500)
      .json({ error: error.message || "Failed to get feature files" });
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
 * GET /gitree/stats
 */
export async function gitree_stats(_req: Request, res: Response) {
  try {
    const storage = new GraphStorage();
    await storage.initialize();

    const features = await storage.getAllFeatures();
    const prs = await storage.getAllPRs();
    const lastProcessed = await storage.getLastProcessedPR();

    const avgPRsPerFeature =
      features.length > 0
        ? features.reduce((sum, f) => sum + f.prNumbers.length, 0) /
          features.length
        : 0;

    const mostActive =
      features.length > 0
        ? features.reduce((max, f) =>
            f.prNumbers.length > max.prNumbers.length ? f : max
          )
        : null;

    res.json({
      totalFeatures: features.length,
      totalPRs: prs.length,
      lastProcessedPR: lastProcessed,
      avgPRsPerFeature: parseFloat(avgPRsPerFeature.toFixed(1)),
      mostActiveFeature: mostActive
        ? {
            id: mostActive.id,
            name: mostActive.name,
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
 * Summarize a specific feature
 * POST /gitree/summarize/:id
 * GET /progress?request_id=xxx
 */
export async function gitree_summarize_feature(req: Request, res: Response) {
  console.log("===> gitree_summarize_feature", req.url, req.method);
  const request_id = asyncReqs.startReq();
  try {
    const featureId = req.params.id;

    // Summarize in background
    (async () => {
      try {
        const anthropicKey = getApiKeyForProvider("anthropic");
        const storage = new GraphStorage();
        await storage.initialize();

        const summarizer = new Summarizer(storage, "anthropic", anthropicKey);
        const usage = await summarizer.summarizeFeature(featureId);

        asyncReqs.finishReq(request_id, {
          status: "success",
          message: `Summarized feature ${featureId}`,
          usage,
        });
      } catch (error) {
        asyncReqs.failReq(request_id, error);
      }
    })();

    res.json({ request_id, status: "pending" });
  } catch (error) {
    console.log("===> error", error);
    asyncReqs.failReq(request_id, error);
    res.status(500).json({ error: "Failed to summarize feature" });
  }
}

/**
 * Summarize all features
 * POST /gitree/summarize-all
 * GET /progress?request_id=xxx
 */
export async function gitree_summarize_all(req: Request, res: Response) {
  console.log("===> gitree_summarize_all", req.url, req.method);
  const request_id = asyncReqs.startReq();
  try {
    // Summarize in background
    (async () => {
      try {
        const anthropicKey = getApiKeyForProvider("anthropic");
        const storage = new GraphStorage();
        await storage.initialize();

        const summarizer = new Summarizer(storage, "anthropic", anthropicKey);
        const usage = await summarizer.summarizeAllFeatures();

        asyncReqs.finishReq(request_id, {
          status: "success",
          message: "Summarized all features",
          usage,
        });
      } catch (error) {
        asyncReqs.failReq(request_id, error);
      }
    })();

    res.json({ request_id, status: "pending" });
  } catch (error) {
    console.log("===> error", error);
    asyncReqs.failReq(request_id, error);
    res.status(500).json({ error: "Failed to summarize all features" });
  }
}

/**
 * Link features to file nodes in the graph
 * POST /gitree/link-files?feature_id=xxx (optional feature_id)
 * GET /progress?request_id=xxx
 */
export async function gitree_link_files(req: Request, res: Response) {
  console.log("===> gitree_link_files", req.url, req.method);
  const request_id = asyncReqs.startReq();
  try {
    const featureId = req.query.feature_id as string | undefined;

    // Link in background
    (async () => {
      try {
        const storage = new GraphStorage();
        await storage.initialize();

        const linker = new FileLinker(storage);

        let result;
        if (featureId) {
          result = await linker.linkFeature(featureId);
        } else {
          result = await linker.linkAllFeatures();
        }

        asyncReqs.finishReq(request_id, {
          status: "success",
          message: featureId
            ? `Linked files for feature ${featureId}`
            : "Linked files for all features",
          result,
        });
      } catch (error) {
        asyncReqs.failReq(request_id, error);
      }
    })();

    res.json({ request_id, status: "pending" });
  } catch (error) {
    console.log("===> error", error);
    asyncReqs.failReq(request_id, error);
    res.status(500).json({ error: "Failed to link files" });
  }
}

/**
 * Convert node to concise format with only name, file, ref_id, and node_type
 */
function toConciseNode(node: Neo4jNode): { name: string; file: string; ref_id: string; node_type: string } {
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
 * Get all features with files and contained nodes as a flat graph structure
 * GET /gitree/all-features-graph?limit=100&node_types=Function,Class&concise=true&depth=2&per_type_limits=Function:50,Class:25
 */
export async function gitree_all_features_graph(req: Request, res: Response) {
  try {
    const storage = new GraphStorage();
    await storage.initialize();

    // Get all data from GraphStorage
    const data = await storage.getAllFeaturesWithFilesAndContains();

    // Parse query parameters
    const limit = parseLimit(req.query);
    const requestedNodeTypes = parseNodeTypes(req.query);
    const concise = isTrue(req.query.concise as string);
    const depth = parseInt(req.query.depth as string) || undefined;
    const perTypeLimitsParam = req.query.per_type_limits as string;

    // Parse per-type limits (e.g., "Function:50,Class:25")
    const perTypeLimits: { [nodeType: string]: number } = {};
    if (perTypeLimitsParam) {
      const pairs = perTypeLimitsParam.split(',');
      for (const pair of pairs) {
        const [nodeType, limitStr] = pair.trim().split(':');
        if (nodeType && limitStr) {
          const typeLimit = parseInt(limitStr);
          if (!isNaN(typeLimit)) {
            perTypeLimits[nodeType] = typeLimit;
          }
        }
      }
    }

    // Combine all nodes
    let allNodes = [
      ...data.features,
      ...data.files,
      ...data.containedNodes,
    ];

    // Filter by node types if specified
    if (requestedNodeTypes.length > 0) {
      allNodes = allNodes.filter((node) => {
        const nodeType = node.labels.find((l: string) => l !== "Data_Bank");
        return nodeType === "Feature" || nodeType === "File" || requestedNodeTypes.includes(nodeType);
      });
    }

    // Apply per-type limits if specified
    if (Object.keys(perTypeLimits).length > 0) {
      const nodesByType: { [nodeType: string]: any[] } = {};

      // Group nodes by type
      allNodes.forEach((node) => {
        const nodeType = node.labels.find((l: string) => l !== "Data_Bank") || "Unknown";
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

    // TODO: Implement depth filtering
    // The depth parameter would require traversing the graph relationships
    // from Features -> Files -> Contained nodes, limiting the traversal depth
    // This would need to be implemented in the GraphStorage layer

    // Apply global limit if specified (after per-type limits)
    if (limit) {
      allNodes = allNodes.slice(0, limit);
    }

    // Transform nodes to return format
    const returnNodes = concise
      ? allNodes.map((node) => toConciseNode(node))
      : allNodes.map((node) => toReturnNode(node));

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
        allNodes.map((n) => {
          const label = n.labels.find((l: string) => l !== "Data_Bank");
          return label;
        }).filter(Boolean)
      )
    ) as NodeType[];

    const meta = buildGraphMeta(
      nodeTypes,
      allNodes,
      limit,
      "total",
      undefined
    );

    res.json({
      nodes: returnNodes,
      edges: allEdges,
      status: "Success",
      meta,
    });
  } catch (error: any) {
    console.error("Error getting all features graph:", error);
    res.status(500).json({
      error: error.message || "Failed to get all features graph",
    });
  }
}
