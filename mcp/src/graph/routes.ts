import { Request, Response, NextFunction } from "express";
import { startTracking, endTracking } from "../busy.js";
import {
  ContainerConfig,
  Neo4jNode,
  node_type_descriptions,
  NodeType,
  EdgeType,
  relevant_node_types,
} from "./types.js";
import {
  nameFileOnly,
  toReturnNode,
  isTrue,
  detectLanguagesAndPkgFiles,
  cloneRepoToTmp,
  extractEnvVarsFromRepo,
  findDockerComposeFiles,
  parseNodeTypes,
  parseRefIds,
  parseSince,
  parseLimit,
  parseLimitMode,
  buildGraphMeta,
  normalizeRepoParam,
} from "./utils.js";
import fs from "fs/promises";
import * as G from "./graph.js";
import { db } from "./neo4j.js";
import { parseServiceFile, extractContainersFromCompose } from "./service.js";
import * as path from "path";
import {
  get_context_explore,
  GeneralContextResult,
} from "../tools/explore/tool.js";
import { create_hint_edges_llm } from "../tools/intelligence/seed.js";
import {
  ask_question,
  QUESTIONS,
  LEARN_HTML,
  ask_prompt,
  learnings,
} from "../tools/intelligence/index.js";
import {
  createBudgetTracker,
  addUsage,
  isBudgetExceeded,
  getBudgetInfo,
} from "../tools/budget.js";
import { generate_persona_variants } from "../tools/intelligence/persona.js";
import {
  clone_and_explore_parse_files,
  clone_and_explore,
} from "../gitsee/agent/index.js";
import { GitSeeHandler } from "gitsee/server";
import * as asyncReqs from "./reqs.js";
import {
  prepareGitHubRepoNode,
  prepareContributorNode,
  prepareStarsNode,
  prepareCommitsNode,
  prepareAgeNode,
  prepareIssuesNode,
} from "./gitsee-nodes.js";

export function schema(_req: Request, res: Response) {
  const schema = node_type_descriptions();
  const schemaArray = Object.entries(schema).map(
    ([node_type, description]) => ({
      node_type,
      description,
    })
  );
  res.json(schemaArray);
}

export function logEndpoint(req: Request, res: Response, next: NextFunction) {
  if (req.headers["x-api-token"]) {
    console.log(`=> ${req.method} ${req.path} [auth]`);
  } else {
    console.log(`=> ${req.method} ${req.path}`);
  }
  next();
}

export function authMiddleware(
  req: Request,
  res: Response,
  next: NextFunction
) {
  const apiToken = process.env.API_TOKEN;
  if (!apiToken) {
    return next();
  }

  // Check for x-api-token header
  const requestToken = req.header("x-api-token");
  if (requestToken && requestToken === apiToken) {
    return next();
  }

  // Check for Basic Auth header
  const authHeader = req.header("Authorization") || req.header("authorization");
  if (authHeader && authHeader.startsWith("Basic ")) {
    try {
      const base64Credentials = authHeader.substring(6);
      const credentials = Buffer.from(base64Credentials, "base64").toString(
        "ascii"
      );
      const [username, token] = credentials.split(":");
      if (token && token === apiToken) {
        return next();
      }
    } catch (error) {
      // Invalid base64 encoding
    }
  }

  res.status(401).json({ error: "Unauthorized: Invalid API token" });
  return;
}

export async function explore(req: Request, res: Response) {
  const prompt = req.query.prompt as string;
  if (!prompt) {
    res.status(400).json({ error: "Missing prompt" });
    return;
  }
  try {
    const result = await get_context_explore(prompt);
    res.json({ result: result.final, usage: result.usage });
  } catch (error) {
    console.error("Explore Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function understand(req: Request, res: Response) {
  try {
    const question = req.query.question as string;
    const similarityThreshold =
      parseFloat(req.query.threshold as string) || 0.88;
    if (!question) {
      res.status(400).json({ error: "Missing question" });
      return;
    }
    const provider = req.query.provider as string | undefined;
    const answer = await ask_question(question, similarityThreshold, provider);
    res.json(answer);
  } catch (e: any) {
    console.error(e);
    res.status(500).json({ error: "Failed" });
  }
}

export async function seed_understanding(req: Request, res: Response) {
  try {
    const budgetDollars = req.query.budget
      ? parseFloat(req.query.budget as string)
      : undefined;
    const provider = (req.query.provider as string) || "anthropic";

    const answers = [];
    let budgetTracker = createBudgetTracker(
      budgetDollars || Number.MAX_SAFE_INTEGER,
      provider as any
    );
    let budgetExceeded = false;

    if (budgetDollars) {
      console.log(`Budget limit enabled: $${budgetDollars}`);
    }

    // Sequential processing - one at a time
    for (const question of QUESTIONS) {
      if (budgetDollars && isBudgetExceeded(budgetTracker)) {
        console.log("Budget exceeded, stopping processing");
        budgetExceeded = true;
        break;
      }

      const answer = await ask_question(question, 0.85, provider);
      if (!answer.reused) {
        console.log("ANSWERED question:", question);
      }
      answers.push(answer);

      budgetTracker = addUsage(
        budgetTracker,
        answer.usage.inputTokens,
        answer.usage.outputTokens,
        provider as any
      );
      const info = getBudgetInfo(budgetTracker);
      if (budgetDollars) {
        console.log(
          `Budget: $${info.totalCost.toFixed(4)} / $${budgetDollars} (${
            answers.length
          } questions)`
        );
      } else {
        console.log(
          `Cost: $${info.totalCost.toFixed(4)} (${answers.length} questions)`
        );
      }
    }

    const info = getBudgetInfo(budgetTracker);
    const response: any = {
      answers,
      budget: {
        totalCost: info.totalCost,
        budgetExceeded,
        remainingBudget: budgetDollars ? info.remainingBudget : undefined,
        questionsProcessed: answers.length,
        questionsSkipped: QUESTIONS.length - answers.length,
        inputTokens: info.inputTokens,
        outputTokens: info.outputTokens,
        totalTokens: info.totalTokens,
      },
    };

    res.json(response);
  } catch (e: any) {
    console.error(e);
    res.status(500).json({ error: "Failed" });
  }
}

export async function ask(req: Request, res: Response) {
  const question = req.query.question as string;
  if (!question) {
    res.status(400).json({ error: "Missing question" });
    return;
  }
  const similarityThreshold =
    parseFloat(req.query.threshold as string) || undefined;
  const provider = req.query.provider as string | undefined;

  // Parse cache control options
  const cacheControl: any = {};
  if (req.query.maxAgeHours) {
    cacheControl.maxAgeHours = parseFloat(req.query.maxAgeHours as string);
  }
  if (req.query.forceRefresh) {
    cacheControl.forceRefresh =
      req.query.forceRefresh === "true" || req.query.forceRefresh === "1";
  }
  if (req.query.forceCache) {
    cacheControl.forceCache =
      req.query.forceCache === "true" || req.query.forceCache === "1";
  }

  try {
    const answer = await ask_prompt(
      question,
      provider,
      similarityThreshold,
      cacheControl
    );
    res.json(answer);
  } catch (error) {
    console.error("Ask Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function get_learnings(req: Request, res: Response) {
  // curl "http://localhost:3355/learnings?question=how%20does%20auth%20work%20in%20the%20repo"
  const question =
    (req.query.question as string) ||
    "What are the core user stories in this project?";

  try {
    // Fetch top 25 Prompt nodes using vector search
    const { prompts, hints } = await learnings(question);

    res.json({
      prompts: Array.isArray(prompts) ? prompts : [],
      hints: Array.isArray(hints) ? hints : [],
    });
  } catch (error) {
    console.error("Learnings Error:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
}

export async function fetch_node_with_related(req: Request, res: Response) {
  // curl "http://localhost:3355/node?ref_id=bcc79e17-fae9-41d6-8932-40ea60e34b54"
  try {
    const ref_id = req.query.ref_id as string;
    if (!ref_id) {
      res.status(400).json({ error: "Missing ref_id parameter" });
      return;
    }

    const result = await db.get_node_with_related(ref_id);

    res.json(result);
  } catch (error) {
    console.error("Fetch Node with Related Error:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
}

export async function fetch_workflow_published_version(
  req: Request,
  res: Response
) {
  // curl "http://localhost:3355/workflow?ref_id=<workflow-ref-id>&concise=true"
  try {
    const ref_id = req.query.ref_id as string;
    if (!ref_id) {
      res.status(400).json({ error: "Missing ref_id parameter" });
      return;
    }

    const concise = isTrue(req.query.concise as string);
    const result = await db.get_workflow_published_version_subgraph(
      ref_id,
      concise
    );

    res.json(result);
  } catch (error) {
    console.error("Fetch Workflow Published Version Error:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
}

export async function generate_siblings(req: Request, res: Response) {
  try {
    const orphanHints = await db.hints_without_siblings();
    let processed = 0;

    for (const hint of orphanHints) {
      const origRef = hint.ref_id || hint.properties.ref_id;
      const question = hint.properties.question || hint.properties.name || "";
      const answer = hint.properties.body || "";
      if (!origRef || !question || !answer) continue;
      processed++;
    }

    await generate_persona_variants();
    res.json({ processed });
  } catch (error) {
    console.error("Generate Siblings Error:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
}

export function learn(req: Request, res: Response) {
  const apiToken = process.env.API_TOKEN;
  if (!apiToken) {
    res.setHeader("Content-Type", "text/html");
    res.send(LEARN_HTML);
    return;
  }

  // Check if user is already authenticated
  const authHeader = req.header("Authorization") || req.header("authorization");
  if (authHeader && authHeader.startsWith("Basic ")) {
    try {
      const base64Credentials = authHeader.substring(6);
      const credentials = Buffer.from(base64Credentials, "base64").toString(
        "ascii"
      );
      const [username, token] = credentials.split(":");

      if (token && token === apiToken) {
        res.setHeader("Content-Type", "text/html");
        res.send(LEARN_HTML);
        return;
      }
    } catch (error) {
      // Invalid base64 encoding, fall through to challenge
    }
  }

  // Send Basic Auth challenge
  res.setHeader("WWW-Authenticate", 'Basic realm="API Access"');
  res.status(401).send("Authentication required");
}

export async function create_pull_request(req: Request, res: Response) {
  const { name, docs, number } = req.body;

  if (!name || !docs || !number) {
    res.status(400).json({
      error: "Missing required fields: name, docs, and number are required",
    });
    return;
  }

  try {
    // Vectorize the docs
    const { vectorizeQuery } = await import("../vector/index.js");
    const embeddings = await vectorizeQuery(docs);

    // Create the PullRequest node
    const result = await db.create_pull_request(name, docs, embeddings, number);

    res.json({
      success: true,
      ref_id: result.ref_id,
      number: result.number,
      node_key: result.node_key,
    });
  } catch (error) {
    console.error("Create PullRequest Error:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
}

export async function create_learning(req: Request, res: Response) {
  const { question, answer, context, featureIds, conceptIds } = req.body;

  // Accept either featureIds or conceptIds (used interchangeably on frontend)
  const ids = conceptIds || featureIds;

  // Validate required fields
  if (!question || !answer) {
    res.status(400).json({
      error: "Missing required fields: question and answer are required",
    });
    return;
  }

  // Validate either/or requirement: featureIds/conceptIds OR context
  if (ids === undefined && !context) {
    res.status(400).json({
      error: "Either featureIds/conceptIds or context must be provided",
    });
    return;
  }

  try {
    // Embed the question only
    const { vectorizeQuery } = await import("../vector/index.js");
    const embeddings = await vectorizeQuery(question);

    // Create the Learning node
    const result = await db.create_learning(
      question,
      answer,
      embeddings,
      context
    );

    // Create ABOUT edges to Feature nodes if ids provided
    let linkedFeatures: string[] = [];
    if (ids && Array.isArray(ids) && ids.length > 0) {
      const edgeResult = await db.create_learning_about_edges(
        result.ref_id,
        ids
      );
      linkedFeatures = edgeResult.linked_features;
    }

    res.json({
      success: true,
      ref_id: result.ref_id,
      node_key: result.node_key,
      linked_features: linkedFeatures,
    });
  } catch (error) {
    console.error("Create Learning Error:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
}

export async function seed_stories(req: Request, res: Response) {
  const default_prompt =
    "How does this repository work? Please provide a summary of the codebase, a few key files, and 50 core user stories.";
  const prompt = (req.query.prompt as string | undefined) || default_prompt;
  const budgetDollars = req.query.budget
    ? parseFloat(req.query.budget as string)
    : undefined;
  const provider = (req.query.provider as string) || "anthropic";

  try {
    let budgetTracker = createBudgetTracker(
      budgetDollars || Number.MAX_SAFE_INTEGER,
      provider as any
    );
    let budgetExceeded = false;

    if (budgetDollars) {
      console.log(`Budget limit enabled: $${budgetDollars}`);
    }

    const gres = await get_context_explore(prompt, false, true);

    budgetTracker = addUsage(
      budgetTracker,
      gres.usage.inputTokens,
      gres.usage.outputTokens,
      provider as any
    );
    const contextInfo = getBudgetInfo(budgetTracker);
    if (budgetDollars) {
      console.log(
        `Initial context: $${contextInfo.totalCost.toFixed(
          4
        )} / $${budgetDollars}`
      );
    } else {
      console.log(`Initial context: $${contextInfo.totalCost.toFixed(4)}`);
    }

    const stories = JSON.parse(gres.final) as GeneralContextResult;
    let answers = [];

    for (const feature of stories.features) {
      if (budgetDollars && isBudgetExceeded(budgetTracker)) {
        console.log("Budget exceeded, stopping processing");
        budgetExceeded = true;
        break;
      }

      console.log("+++++++++ feature:", feature);
      const answer = await ask_prompt(feature, provider);
      answers.push(answer);

      budgetTracker = addUsage(
        budgetTracker,
        answer.usage.inputTokens,
        answer.usage.outputTokens,
        provider as any
      );
      const info = getBudgetInfo(budgetTracker);
      if (budgetDollars) {
        console.log(
          `Budget: $${info.totalCost.toFixed(4)} / $${budgetDollars} (${
            answers.length
          } features)`
        );
      } else {
        console.log(
          `Cost: $${info.totalCost.toFixed(4)} (${answers.length} features)`
        );
      }
    }

    const info = getBudgetInfo(budgetTracker);
    const response: any = {
      answers,
      budget: {
        totalCost: info.totalCost,
        budgetExceeded,
        remainingBudget: budgetDollars ? info.remainingBudget : undefined,
        featuresProcessed: answers.length,
        featuresSkipped: stories.features.length - answers.length,
        inputTokens: info.inputTokens,
        outputTokens: info.outputTokens,
        totalTokens: info.totalTokens,
      },
    };

    res.json(response);
  } catch (error) {
    console.error("Seed Stories Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function get_nodes(req: Request, res: Response) {
  try {
    console.log("=> get_nodes", req.method, req.path);
    const node_type = req.query.node_type as NodeType;
    const concise = isTrue(req.query.concise as string);
    let ref_ids: string[] = [];
    if (req.query.ref_ids) {
      ref_ids = (req.query.ref_ids as string).split(",");
    }
    const output = req.query.output as G.OutputFormat;
    const language = req.query.language as string;

    const result = await G.get_nodes(
      node_type,
      concise,
      ref_ids,
      output,
      language
    );
    if (output === "snippet") {
      res.send(result);
    } else {
      res.json(result);
    }
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function post_nodes(req: Request, res: Response) {
  try {
    console.log("=> post_nodes", req.method, req.path);
    const node_type = req.body.node_type as NodeType;
    const concise = req.body.concise === true || req.body.concise === "true";
    let ref_ids: string[] = [];
    if (req.body.ref_ids) {
      if (Array.isArray(req.body.ref_ids)) {
        ref_ids = req.body.ref_ids;
      } else {
        res.status(400).json({ error: "ref_ids must be an array" });
        return;
      }
    }
    const output = req.body.output as G.OutputFormat;
    const language = req.body.language as string;

    const result = await G.get_nodes(
      node_type,
      concise,
      ref_ids,
      output,
      language
    );
    if (output === "snippet") {
      res.send(result);
    } else {
      res.json(result);
    }
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function get_edges(req: Request, res: Response) {
  try {
    const edge_type = req.query.edge_type as EdgeType;
    const concise = isTrue(req.query.concise as string);
    let ref_ids: string[] = [];
    if (req.query.ref_ids) {
      ref_ids = (req.query.ref_ids as string).split(",");
    }
    const output = req.query.output as G.OutputFormat;
    const language = req.query.language as string;

    const result = await G.get_edges(
      edge_type,
      concise,
      ref_ids,
      output,
      language
    );
    if (output === "snippet") {
      res.send(result);
    } else {
      res.json(result);
    }
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function search(req: Request, res: Response) {
  try {
    const query = req.query.query as string;
    const limit = parseInt(req.query.limit as string) || 25;
    const concise = isTrue(req.query.concise as string);
    let node_types: NodeType[] = [];
    if (req.query.node_types) {
      node_types = (req.query.node_types as string).split(",") as NodeType[];
    } else if (req.query.node_type) {
      node_types = [req.query.node_type as NodeType];
    }
    const method = req.query.method as G.SearchMethod;
    const output = req.query.output as G.OutputFormat;
    let tests = isTrue(req.query.tests as string);
    const maxTokens = parseInt(req.query.max_tokens as string);
    const language = req.query.language as string;

    if (maxTokens) {
      console.log("search with max tokens", maxTokens);
    }
    const result = await G.search(
      query,
      limit,
      node_types,
      concise,
      maxTokens || 100000,
      method,
      output || "snippet",
      tests,
      language
    );
    if (output === "snippet") {
      res.send(result);
    } else {
      res.json(result);
    }
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}
export async function get_rules_files(req: Request, res: Response) {
  try {
    const snippets = await G.get_rules_files();
    res.json(snippets);
  } catch (error) {
    console.error("Error fetching rules files:", error);
    res.status(500).json({ error: "Failed to fetch rules files" });
  }
}

const gitSeeHandler = new GitSeeHandler({
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

    req.body.useCache = false;
    await gitSeeHandler.handleJson(req.body, res);
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
  const { owner, repo } = req.params;
  if (!owner || !repo) {
    res.status(400).json({ error: "Owner and repo are required" });
    return;
  }
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
    const owner = req.query.owner as string;
    const repo = req.query.repo as string | undefined;
    if (!repo || !owner) {
      res.status(400).json({ error: "Missing repo" });
      return;
    }
    const username = req.query.username as string | undefined;
    const pat = req.query.pat as string | undefined;
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
    const owner = req.query.owner as string;
    const repo = req.query.repo as string | undefined;
    const prompt = req.query.prompt as string | undefined;
    const system = req.query.system as string | undefined;
    const final_answer = req.query.final_answer as string | undefined;
    if (!repo || !owner) {
      res.status(400).json({ error: "Missing repo" });
      return;
    }
    if (!prompt) {
      res.status(400).json({ error: "Missing prompt" });
      return;
    }
    const username = req.query.username as string | undefined;
    const pat = req.query.pat as string | undefined;
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

export async function get_services(req: Request, res: Response) {
  try {
    if (req.query.clone === "true" && req.query.repo_url) {
      const repoUrl = req.query.repo_url as string;
      const username = req.query.username as string | undefined;
      const pat = req.query.pat as string | undefined;
      const commit = req.query.commit as string | undefined;

      const repoDir = await cloneRepoToTmp(repoUrl, username, pat, commit);
      const detected = await detectLanguagesAndPkgFiles(repoDir);

      const envVarsByFile = await extractEnvVarsFromRepo(repoDir);

      const services = [];
      for (const { language, pkgFile } of detected) {
        const body = await fs.readFile(pkgFile, "utf8");
        const service = parseServiceFile(pkgFile, body, language);

        const serviceDir = path.dirname(pkgFile);
        const envVars = new Set<string>();
        for (const [file, vars] of Object.entries(envVarsByFile)) {
          if (file.startsWith(serviceDir)) {
            vars.forEach((v) => envVars.add(v));
          }
        }

        service.env = {};
        envVars.forEach((v) => (service.env[v] = process.env[v] || ""));

        const { pkgFile: _, ...cleanService } = service;
        services.push(cleanService);
      }
      const composeFiles = await findDockerComposeFiles(repoDir);
      let containers: ContainerConfig[] = [];
      for (const composeFile of composeFiles) {
        const found = await extractContainersFromCompose(composeFile);
        containers = containers.concat(found);
      }
      res.json({ services, containers });
      return;
    } else {
      const { services, containers } = await G.get_services();
      res.json({ services, containers });
    }
  } catch (error) {
    console.error("Error getting services config:", error);
    res
      .status(500)
      .json({ error: "Failed to generate services configuration" });
  }
}

export async function mocks_inventory(req: Request, res: Response) {
  try {
    const search = req.query.search as string | undefined;
    const repo = normalizeRepoParam(req.query.repo as string | undefined);
    const limit = parseInt(req.query.limit as string) || 50;
    const offset = parseInt(req.query.offset as string) || 0;

    const result = await G.get_mocks_inventory(search, limit, offset, repo);
    res.json(result);
  } catch (error) {
    console.error("Error getting mocks inventory:", error);
    res.status(500).json({ error: "Failed to get mocks inventory" });
  }
}

export function toNode(node: Neo4jNode, concise: boolean): any {
  return concise ? nameFileOnly(node) : toReturnNode(node);
}

const DEFAULT_DEPTH = 7;

interface MapParams {
  node_type: string;
  name: string;
  file: string;
  ref_id: string;
  tests: boolean;
  depth: number;
  direction: G.Direction;
  trim: string[];
}

function mapParams(req: Request): MapParams {
  const node_type = req.query.node_type as string;
  const name = req.query.name as string;
  const file = req.query.file as string;
  const ref_id = req.query.ref_id as string;
  const name_and_type = node_type && name;
  const file_and_type = node_type && file;
  if (!name_and_type && !file_and_type && !ref_id) {
    throw new Error(
      "either node_type+name, node_type+file, or ref_id required"
    );
  }
  const direction = req.query.direction as G.Direction;
  const tests = !(req.query.tests === "false" || req.query.tests === "0");
  const depth = parseInt(req.query.depth as string) || DEFAULT_DEPTH;
  const default_direction = "both" as G.Direction;
  return {
    node_type: node_type || "",
    name: name || "",
    file: file || "",
    ref_id: ref_id || "",
    tests,
    depth,
    direction: direction || default_direction,
    trim: ((req.query.trim as string) || "").split(","),
  };
}

export async function get_map(req: Request, res: Response) {
  try {
    const html = await G.get_map(mapParams(req));
    res.send(`<pre>\n${html}\n</pre>`);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function get_repo_map(req: Request, res: Response) {
  try {
    const name = req.query.name as string;
    const ref_id = req.query.ref_id as string;
    const node_type = req.query.node_type as NodeType;
    const normalizedName =
      (node_type || "Repository") === "Repository"
        ? normalizeRepoParam(name) || name || ""
        : name || "";
    const include_functions_and_classes =
      req.query.include_functions_and_classes === "true" ||
      req.query.include_functions_and_classes === "1";
    const html = await G.get_repo_map(
      normalizedName,
      ref_id || "",
      node_type || "Repository",
      include_functions_and_classes
    );
    res.send(`<pre>\n${html}\n</pre>`);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function get_code(req: Request, res: Response) {
  try {
    const text = await G.get_code(mapParams(req));
    res.send(text);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function get_shortest_path(req: Request, res: Response) {
  try {
    const result = await G.get_shortest_path(
      req.query.start_node_key as string,
      req.query.end_node_key as string,
      req.query.start_ref_id as string,
      req.query.end_ref_id as string
    );
    res.send(result);
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function get_graph(req: Request, res: Response) {
  try {
    const edge_type =
      (req.query.edge_type as EdgeType) || ("CALLS" as EdgeType);
    const concise = isTrue(req.query.concise as string);
    const include_edges = isTrue(req.query.edges as string);
    const language = req.query.language as string | undefined;
    const since = parseSince(req.query);
    const limit_param = parseLimit(req.query);
    const limit_mode = parseLimitMode(req.query);
    let labels = parseNodeTypes(req.query);
    if (labels.length === 0) labels = relevant_node_types();

    const perTypeDefault = 100;
    let nodes: any[] = [];
    const ref_ids = parseRefIds(req.query);
    if (ref_ids.length > 0) {
      nodes = await db.nodes_by_ref_ids(ref_ids, language);
    } else {
      if (limit_mode === "total") {
        nodes = await db.nodes_by_types_total(
          labels,
          limit_param || perTypeDefault,
          since,
          language
        );
      } else {
        nodes = await db.nodes_by_types_per_type(
          labels,
          limit_param || perTypeDefault,
          since,
          language
        );
      }
    }

    let edges: any[] = [];
    if (include_edges) {
      const keys = nodes.map((n) => n.properties.node_key).filter(Boolean);
      edges = await db.edges_between_node_keys(keys);
    }

    res.json({
      nodes: concise
        ? nodes.map((n) => nameFileOnly(n))
        : nodes.map((n) => toReturnNode(n)),
      edges: include_edges
        ? concise
          ? edges.map((e) => ({
              edge_type: e.edge_type,
              source: e.source,
              target: e.target,
            }))
          : edges
        : [],
      status: "Success",
      meta: buildGraphMeta(labels, nodes, limit_param, limit_mode, since),
    });
  } catch (error) {
    console.error("Error:", error);
    res.status(500).send("Internal Server Error");
  }
}

export async function get_script_progress(req: Request, res: Response) {
  console.log(`===> GET /script_progress`);
  try {
    const request_id = req.query.request_id as string;
    if (!request_id) {
      res.status(400).json({ error: "request_id is required" });
      return;
    }
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

export async function reconnect_orphaned_hints(req: Request, res: Response) {
  try {
    const provider = (req.query.provider as string) || "anthropic";
    const orphanedHints = await db.get_orphaned_hints();

    const results = {
      processed: orphanedHints.length,
      reconnected: 0,
      failed: [] as { ref_id: string; error: string }[],
    };

    for (const hint of orphanedHints) {
      const ref_id = hint.ref_id || hint.properties.ref_id;
      const answer = hint.properties.body || hint.properties.answer;

      if (!ref_id || !answer) {
        results.failed.push({
          ref_id: ref_id || "unknown",
          error: "Missing ref_id or answer",
        });
        continue;
      }

      try {
        const result = await create_hint_edges_llm(ref_id, answer, provider);
        if (result.edges_added > 0) {
          results.reconnected++;
        }
      } catch (error: any) {
        results.failed.push({
          ref_id,
          error: error.message || "Unknown error",
        });
      }
    }

    res.json(results);
  } catch (error: any) {
    res
      .status(500)
      .json({ error: error.message || "Failed to reconnect orphaned hints" });
  }
}
