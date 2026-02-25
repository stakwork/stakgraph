import { Request, Response } from "express";
import { startTracking, endTracking } from "../../busy.js";
import {
  clone_and_explore_parse_files,
  clone_and_explore,
} from "../../gitsee/agent/index.js";
import { GitSeeHandler } from "gitsee/server";
import * as asyncReqs from "../reqs.js";
import {
  prepareGitHubRepoNode,
  prepareContributorNode,
  prepareStarsNode,
  prepareCommitsNode,
  prepareAgeNode,
  prepareIssuesNode,
} from "../gitsee-nodes.js";
import { db } from "../neo4j.js";
import { parseBody, parseParams, parseQuery } from "./validation.js";
import {
  gitseeBodySchema,
  gitseeEventsParamsSchema,
  gitseeServicesQuerySchema,
  gitseeAgentQuerySchema,
  requestIdQuerySchema,
} from "./schemas/gitsee.js";

export const gitSeeHandler = new GitSeeHandler({
  token: process.env.GITHUB_TOKEN,
});

/*
curl -X POST -H "Content-Type: application/json" -d '{"owner": "stakwork", "repo": "hive", "data": ["contributors", "icon", "repo_info", "stats"]}' "http://localhost:3355/gitsee"

interface GitSeeRequest {
  owner: string;
  repo: string;
  data: ("contributors" | "icon" | "repo_info" | "commits" | "branches" | "files" | "stats" | "file_content" | "exploration")[];
  filePath?: string;
  explorationMode?: "features" | "first_pass";
  explorationPrompt?: string;
  cloneOptions?: CloneOptions;
  useCache?: boolean;
}
*/
export async function gitsee(req: Request, res: Response) {
  console.log("===> gitsee API request", req.path, req.method);
  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }
  try {
    const parsed = parseBody(req, res, gitseeBodySchema);
    if (!parsed) return;

    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");

    const opId = startTracking("gitsee");

    const originalEnd = res.end.bind(res);
    let responseData: any;

    (res.end as any) = function (this: Response, chunk?: any): Response {
      if (chunk) {
        try {
          responseData = typeof chunk === "string" ? JSON.parse(chunk) : chunk;
          console.log(
            "===> responseData",
            JSON.stringify(responseData, null, 2)
          );
          if (
            responseData.repo ||
            responseData.contributors ||
            responseData.stats
          ) {
            ingestGitSeeData(responseData)
              .catch((err) => console.error("Background ingestion error:", err))
              .finally(() => {
                endTracking(opId);
              });
          } else {
            endTracking(opId);
          }
        } catch (e) {
          console.error("Error parsing response for ingestion:", e);
          endTracking(opId);
        }
      } else {
        endTracking(opId);
      }
      return originalEnd(chunk);
    };

    const payload = { ...parsed, useCache: false };
    await gitSeeHandler.handleJson(payload, res);
  } catch (error) {
    console.error("gitsee API error:", error);
    res.status(500).json({
      error:
        error instanceof Error
          ? error.message
          : "Failed to handle gitsee request",
    });
  }
}

// curl -X POST -H "Content-Type: application/json" -d '' "http://localhost:3355/gitsee"
async function ingestGitSeeData(data: any): Promise<void> {
  // console.log("===> ingestGitSeeData", JSON.stringify(data, null, 2));
  try {
    let repoRefId: string | undefined;

    console.log("===> data.repo", JSON.stringify(data.repo, null, 2));
    if (data.repo) {
      const fullName = data.repo.full_name;
      const minimalRepo = {
        id: fullName,
        name: fullName,
        html_url: data.repo.html_url,
        stargazers_count: data.stats?.stars || 0,
        forks_count: 0,
        icon: data.icon,
      };
      const repoNode = prepareGitHubRepoNode(minimalRepo);
      repoRefId = await db.add_node(repoNode.node_type, repoNode.node_data);
      console.log(`✓ Added GitHubRepo: ${fullName}`);
    }

    if (data.contributors && repoRefId) {
      for (const contributor of data.contributors) {
        const contribNode = prepareContributorNode(contributor);
        const contribRefId = await db.add_node(
          contribNode.node_type,
          contribNode.node_data
        );

        await db.add_edge("HAS_CONTRIBUTOR", repoRefId, contribRefId);
        console.log(`✓ Added Contributor: ${contributor.login}`);
      }
    }

    if (data.stats && repoRefId) {
      // Get repo full name from the actual repo data
      const repoFullName = data.repo.full_name;

      // Create Stars node
      const starsNode = prepareStarsNode(data.stats.stars, repoFullName);
      const starsRefId = await db.add_node(
        starsNode.node_type,
        starsNode.node_data
      );
      await db.add_edge("HAS_STARS", repoRefId, starsRefId);
      console.log(`✓ Added Stars for ${repoFullName}`);

      // Create Commits node
      const commitsNode = prepareCommitsNode(
        data.stats.totalCommits,
        repoFullName
      );
      const commitsRefId = await db.add_node(
        commitsNode.node_type,
        commitsNode.node_data
      );
      await db.add_edge("HAS_COMMITS", repoRefId, commitsRefId);
      console.log(`✓ Added Commits for ${repoFullName}`);

      // Create Age node
      const ageNode = prepareAgeNode(data.stats.ageInYears, repoFullName);
      const ageRefId = await db.add_node(ageNode.node_type, ageNode.node_data);
      await db.add_edge("HAS_AGE", repoRefId, ageRefId);
      console.log(`✓ Added Age for ${repoFullName}`);

      // Create Issues node
      const issuesNode = prepareIssuesNode(
        data.stats.totalIssues,
        repoFullName
      );
      const issuesRefId = await db.add_node(
        issuesNode.node_type,
        issuesNode.node_data
      );
      await db.add_edge("HAS_ISSUES", repoRefId, issuesRefId);
      console.log(`✓ Added Issues for ${repoFullName}`);
    }
  } catch (error) {
    console.error("Error ingesting GitSee data:", error);
  }
}

export async function gitseeEvents(req: Request, res: Response) {
  console.log("===> gitsee SSE request", req.path, req.method);
  if (req.method !== "GET") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }
  // Extract owner/repo from URL params
  const params = parseParams(req, res, gitseeEventsParamsSchema);
  if (!params) return;
  const { owner, repo } = params;
  console.log(`📡 SSE connection for ${owner}/${repo}`);
  try {
    return await gitSeeHandler.handleEvents(
      req as any,
      res as any,
      owner as string,
      repo as string,
    );
  } catch (error) {
    console.error("gitsee SSE error:", error);
    res.status(500).json({ error: "Failed to handle SSE connection" });
  }
}

export async function gitsee_services(req: Request, res: Response) {
  // curl "http://localhost:3355/services_agent?owner=stakwork&repo=hive"
  // curl "http://localhost:3355/progress?request_id=123"
  console.log("===> gitsee_services", req.path, req.method);
  const request_id = asyncReqs.startReq();
  try {
    const parsed = parseQuery(req, res, gitseeServicesQuerySchema);
    if (!parsed) return;

    const owner = parsed.owner;
    const repo = parsed.repo;
    const username = parsed.username;
    const pat = parsed.pat;
    clone_and_explore_parse_files(
      owner,
      repo,
      "How do I set up this repo?",
      "services",
      {
        username,
        token: pat,
      }
    )
      .then((ctx) => {
        asyncReqs.finishReq(request_id, ctx);
      })
      .catch((error) => {
        asyncReqs.failReq(request_id, error);
      });
    res.json({ request_id, status: "pending" });
  } catch (error) {
    console.log("===> error");
    asyncReqs.failReq(request_id, error);
    console.error("Error getting services config:", error);
    res
      .status(500)
      .json({ error: "Failed to generate services configuration" });
  }
}

export async function gitsee_agent(req: Request, res: Response) {
  // curl "http://localhost:3355/agent?owner=stakwork&repo=hive&prompt=How%20do%20I%20set%20up%20this%20repo"
  // curl "http://localhost:3355/progress?request_id=51f5cce2-d5e8-4619-add3-c2f4cb37e1ba"
  console.log("===> gitsee agent", req.path, req.method, {
    hasPrompt: Boolean(req.query.prompt),
  });
  const request_id = asyncReqs.startReq();
  try {
    const parsed = parseQuery(req, res, gitseeAgentQuerySchema);
    if (!parsed) return;

    const owner = parsed.owner;
    const repo = parsed.repo;
    const prompt = parsed.prompt;
    const system = parsed.system;
    const final_answer = parsed.final_answer;
    const username = parsed.username;
    const pat = parsed.pat;
    clone_and_explore(
      owner,
      repo,
      prompt,
      "generic",
      {
        username,
        token: pat,
      },
      {
        system_prompt: system,
        final_answer_description: final_answer,
      }
    )
      .then((ctx) => {
        asyncReqs.finishReq(request_id, ctx);
      })
      .catch((error) => {
        asyncReqs.failReq(request_id, error);
      });
    res.json({ request_id, status: "pending" });
  } catch (error) {
    console.log("===> error");
    asyncReqs.failReq(request_id, error);
    console.error("Error getting services config:", error);
    res
      .status(500)
      .json({ error: "Failed to generate services configuration" });
  }
}

export async function get_script_progress(req: Request, res: Response) {
  console.log(`===> GET /script_progress`);
  try {
    const parsed = parseQuery(req, res, requestIdQuerySchema);
    if (!parsed) return;
    const request_id = parsed.request_id;
    const progress = asyncReqs.checkReq(request_id);
    if (!progress) {
      res.status(404).json({ error: "Request not found" });
      return;
    }
    res.json(progress);
  } catch (error) {
    console.error("Error checking script progress:", error);
    res.status(500).send("Internal Server Error");
  }
}
