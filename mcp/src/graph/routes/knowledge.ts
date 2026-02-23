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
  const { name, docs, number } = req.body;

  if (!name || !docs || !number) {
    res.status(400).json({
      error: "Missing required fields: name, docs, and number are required",
    });
    return;
  }

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
