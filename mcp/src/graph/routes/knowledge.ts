import { Request, Response } from "express";
import {
  get_context_explore,
  GeneralContextResult,
} from "../../tools/explore/tool.js";
import { create_hint_edges_llm } from "../../tools/intelligence/seed.js";
import {
  ask_question,
  QUESTIONS,
  LEARN_HTML,
  ask_prompt,
  learnings,
} from "../../tools/intelligence/index.js";
import {
  createBudgetTracker,
  addUsage,
  isBudgetExceeded,
  getBudgetInfo,
} from "../../tools/budget.js";
import { generate_persona_variants } from "../../tools/intelligence/persona.js";
import { db } from "../neo4j.js";
import {
  parseBody,
  parseQuery,
} from "./validation.js";
import {
  exploreQuerySchema,
  understandQuerySchema,
  seedUnderstandingQuerySchema,
  askQuerySchema,
  getLearningsQuerySchema,
  createPullRequestBodySchema,
  createLearningBodySchema,
  seedStoriesQuerySchema,
  reconnectQuerySchema,
} from "./schemas/knowledge.js";

export async function explore(req: Request, res: Response) {
  const parsed = parseQuery(req, res, exploreQuerySchema);
  if (!parsed) return;
  const prompt = parsed.prompt;
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
    const parsed = parseQuery(req, res, understandQuerySchema);
    if (!parsed) return;

    const question = parsed.question;
    const similarityThreshold = parsed.threshold || 0.88;
    const provider = parsed.provider;
    const answer = await ask_question(question, similarityThreshold, provider);
    res.json(answer);
  } catch (e: any) {
    console.error(e);
    res.status(500).json({ error: "Failed" });
  }
}

export async function seed_understanding(req: Request, res: Response) {
  try {
    const parsed = parseQuery(req, res, seedUnderstandingQuerySchema);
    if (!parsed) return;

    const budgetDollars = parsed.budget;
    const provider = parsed.provider || "anthropic";

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
  const parsed = parseQuery(req, res, askQuerySchema);
  if (!parsed) return;

  const question = parsed.question;
  const similarityThreshold = parsed.threshold || undefined;
  const provider = parsed.provider;

  // Parse cache control options
  const cacheControl: any = {};
  if (parsed.maxAgeHours !== undefined) {
    cacheControl.maxAgeHours = parsed.maxAgeHours;
  }
  if (parsed.forceRefresh !== undefined) {
    cacheControl.forceRefresh = parsed.forceRefresh;
  }
  if (parsed.forceCache !== undefined) {
    cacheControl.forceCache = parsed.forceCache;
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
  const parsed = parseQuery(req, res, getLearningsQuerySchema);
  if (!parsed) return;

  const question =
    parsed.question ||
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

export async function create_pull_request(req: Request, res: Response) {
  const parsed = parseBody(req, res, createPullRequestBodySchema);
  if (!parsed) return;

  const { name, docs, number } = parsed;

  try {
    // Vectorize the docs
    const { vectorizeQuery } = await import("../../vector/index.js");
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
  const parsed = parseBody(req, res, createLearningBodySchema);
  if (!parsed) return;

  const { question, answer, context, featureIds, conceptIds } = parsed;

  // Accept either featureIds or conceptIds (used interchangeably on frontend)
  const ids = conceptIds || featureIds;

  try {
    // Embed the question only
    const { vectorizeQuery } = await import("../../vector/index.js");
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
  const parsed = parseQuery(req, res, seedStoriesQuerySchema);
  if (!parsed) return;

  const prompt = parsed.prompt || default_prompt;
  const budgetDollars = parsed.budget;
  const provider = parsed.provider || "anthropic";

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

export async function reconnect_orphaned_hints(req: Request, res: Response) {
  try {
    const parsed = parseQuery(req, res, reconnectQuerySchema);
    if (!parsed) return;
    const provider = parsed.provider || "anthropic";
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
