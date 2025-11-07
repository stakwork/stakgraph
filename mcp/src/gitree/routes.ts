import { Request, Response } from "express";
import * as asyncReqs from "../graph/reqs.js";

/**
 * Process a GitHub repository to extract features
 * POST /gitree/process?owner=stakwork&repo=sphinx-tribes&token=...
 * GET /progress?request_id=xxx
 */
export async function gitree_process(req: Request, res: Response) {
  console.log("===> gitree_process", req.url, req.method);
  const request_id = asyncReqs.startReq();
  try {
    const owner = req.query.owner as string;
    const repo = req.query.repo as string;
    const githubToken = req.query.token as string;

    if (!owner || !repo) {
      asyncReqs.failReq(request_id, new Error("Missing owner or repo"));
      res.status(400).json({ error: "Missing owner or repo" });
      return;
    }

    if (!githubToken) {
      asyncReqs.failReq(request_id, new Error("Missing GitHub token"));
      res.status(400).json({ error: "Missing GitHub token" });
      return;
    }

    // Import gitree components
    import("./index.js")
      .then(async (gitree) => {
        const { Octokit } = await import("@octokit/rest");
        const { GraphStorage } = gitree;
        const { LLMClient } = gitree;
        const { StreamingFeatureBuilder } = gitree;
        const { getApiKeyForProvider } = await import("../aieo/src/provider.js");

        const anthropicKey = getApiKeyForProvider("anthropic");
        const storage = new GraphStorage();
        await storage.initialize();

        const octokit = new Octokit({ auth: githubToken });
        const llm = new LLMClient("anthropic", anthropicKey);
        const builder = new StreamingFeatureBuilder(storage, llm, octokit);

        await builder.processRepo(owner, repo);

        asyncReqs.finishReq(request_id, {
          status: "success",
          message: `Processed ${owner}/${repo}`,
        });
      })
      .catch((error) => {
        asyncReqs.failReq(request_id, error);
      });

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
export async function gitree_list_features(req: Request, res: Response) {
  try {
    const { GraphStorage } = await import("./index.js");
    const storage = new GraphStorage();
    await storage.initialize();

    const features = await storage.getAllFeatures();
    res.json({
      features: features.map((f) => ({
        id: f.id,
        name: f.name,
        description: f.description,
        prCount: f.prNumbers.length,
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
 * GET /gitree/features/:id
 */
export async function gitree_get_feature(req: Request, res: Response) {
  try {
    const featureId = req.params.id;
    const { GraphStorage } = await import("./index.js");
    const storage = new GraphStorage();
    await storage.initialize();

    const feature = await storage.getFeature(featureId);

    if (!feature) {
      res.status(404).json({ error: "Feature not found" });
      return;
    }

    const prs = await storage.getPRsForFeature(featureId);

    res.json({
      feature: {
        id: feature.id,
        name: feature.name,
        description: feature.description,
        prNumbers: feature.prNumbers,
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
    });
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
    const { GraphStorage } = await import("./index.js");
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
 * Get knowledge base statistics
 * GET /gitree/stats
 */
export async function gitree_stats(req: Request, res: Response) {
  try {
    const { GraphStorage } = await import("./index.js");
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

    import("./index.js")
      .then(async (gitree) => {
        const { GraphStorage, Summarizer } = gitree;
        const { getApiKeyForProvider } = await import("../aieo/src/provider.js");

        const anthropicKey = getApiKeyForProvider("anthropic");
        const storage = new GraphStorage();
        await storage.initialize();

        const summarizer = new Summarizer(storage, "anthropic", anthropicKey);
        await summarizer.summarizeFeature(featureId);

        asyncReqs.finishReq(request_id, {
          status: "success",
          message: `Summarized feature ${featureId}`,
        });
      })
      .catch((error) => {
        asyncReqs.failReq(request_id, error);
      });

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
    import("./index.js")
      .then(async (gitree) => {
        const { GraphStorage, Summarizer } = gitree;
        const { getApiKeyForProvider } = await import("../aieo/src/provider.js");

        const anthropicKey = getApiKeyForProvider("anthropic");
        const storage = new GraphStorage();
        await storage.initialize();

        const summarizer = new Summarizer(storage, "anthropic", anthropicKey);
        await summarizer.summarizeAllFeatures();

        asyncReqs.finishReq(request_id, {
          status: "success",
          message: "Summarized all features",
        });
      })
      .catch((error) => {
        asyncReqs.failReq(request_id, error);
      });

    res.json({ request_id, status: "pending" });
  } catch (error) {
    console.log("===> error", error);
    asyncReqs.failReq(request_id, error);
    res.status(500).json({ error: "Failed to summarize all features" });
  }
}
