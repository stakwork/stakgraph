import { Request, Response } from "express";
import { db } from "./neo4j.js";
import { vectorizeQuery } from "../vector/index.js";
import { generateObject, jsonSchema } from "ai";
import {
  getApiKeyForProvider,
  getModel,
  Provider,
} from "../aieo/src/provider.js";

// === Learning + Scope routes ===

interface LearningInput {
  id: string;
  rule: string;
  reason?: string;
  scopes: string | string[];
}

export async function post_learnings(req: Request, res: Response) {
  const body = req.body;
  if (!Array.isArray(body)) {
    res.status(400).json({ error: "Request body must be an array of learnings" });
    return;
  }

  try {
    const results: { id: string; ref_id: string; scopes: string[] }[] = [];

    for (const item of body as LearningInput[]) {
      if (!item.id || !item.rule) {
        res.status(400).json({ error: "Each learning must have 'id' and 'rule'" });
        return;
      }

      // Normalize scopes to string[]
      const scopes = Array.isArray(item.scopes)
        ? item.scopes
        : item.scopes
          ? [item.scopes]
          : [];

      // Embed the rule text
      const ruleEmbeddings = await vectorizeQuery(item.rule);

      // Upsert the Learning node (handles duplicates by id)
      const learning = await db.upsert_learning(
        item.id,
        item.rule,
        ruleEmbeddings,
        item.reason,
      );

      // Upsert each Scope node and create HAS_SCOPE edges
      for (const scopeName of scopes) {
        const scopeEmbeddings = await vectorizeQuery(scopeName);
        await db.upsert_scope(scopeName, scopeEmbeddings);
        await db.create_has_scope_edge(item.id, scopeName);
      }

      results.push({
        id: item.id,
        ref_id: learning.ref_id,
        scopes,
      });
    }

    res.json({ success: true, learnings: results });
  } catch (error) {
    console.error("POST learnings error:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
}

export async function get_all_learnings(_req: Request, res: Response) {
  try {
    const learnings = await db.get_all_learnings_with_scopes();
    res.json(learnings);
  } catch (error) {
    console.error("GET learnings/all error:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
}

export async function post_relevant_learnings(req: Request, res: Response) {
  const { prompt } = req.body;
  const limit = parseInt(req.query.limit as string) || 25;

  if (!prompt) {
    res.status(400).json({ error: "Missing prompt in request body" });
    return;
  }

  try {
    const provider = (process.env.LLM_PROVIDER || "anthropic") as Provider;
    const apiKey = getApiKeyForProvider(provider);
    const model = getModel(provider, { apiKey, modelName: "haiku" });

    // 1. List all scopes
    const allScopes = await db.get_all_scopes();
    if (allScopes.length === 0) {
      res.json({ learnings: [], prompt, scopes: [] });
      return;
    }

    // 2. Ask haiku to pick the relevant scopes
    const scopePrompt = `<prompt>${prompt}</prompt>
<scopes>${JSON.stringify(allScopes)}</scopes>

Given the user's prompt, pick the most relevant scopes from the list. Return up to ${limit} scope names that are relevant to the prompt.`;

    const scopeSchema = {
      type: "object" as const,
      properties: {
        relevantScopes: {
          type: "array" as const,
          items: { type: "string" as const },
          description: "Array of scope names relevant to the prompt",
        },
      },
      required: ["relevantScopes"],
      additionalProperties: false,
    };

    const scopeResult = await generateObject({
      model,
      prompt: scopePrompt,
      schema: jsonSchema(scopeSchema),
    });

    const relevantScopes =
      ((scopeResult.object as any).relevantScopes as string[])?.slice(0, limit) || [];

    if (relevantScopes.length === 0) {
      res.json({ learnings: [], prompt, scopes: [] });
      return;
    }

    // 3. List all learnings with those scopes
    const candidateLearnings = await db.get_learnings_by_scopes(relevantScopes);

    if (candidateLearnings.length === 0) {
      res.json({ learnings: [], prompt, scopes: relevantScopes });
      return;
    }

    // 4. Ask haiku to pick the top learnings relevant to the prompt
    const learningsForAI = candidateLearnings.map((l) => ({
      id: l.id,
      rule: l.rule,
      reason: l.reason,
      scopes: l.scopes,
    }));

    const learningPrompt = `<prompt>${prompt}</prompt>
<learnings>${JSON.stringify(learningsForAI, null, 2)}</learnings>

Given the user's prompt, pick the most relevant learnings from the list. Return up to ${limit} learning IDs that are relevant to the prompt.`;

    const learningSchema = {
      type: "object" as const,
      properties: {
        relevantLearningIds: {
          type: "array" as const,
          items: { type: "string" as const },
          description: "Array of learning IDs relevant to the prompt",
        },
      },
      required: ["relevantLearningIds"],
      additionalProperties: false,
    };

    const learningResult = await generateObject({
      model,
      prompt: learningPrompt,
      schema: jsonSchema(learningSchema),
    });

    const relevantIds =
      ((learningResult.object as any).relevantLearningIds as string[])?.slice(0, limit) || [];

    // Filter to only the selected learnings, preserving full data
    const relevantLearnings = candidateLearnings.filter((l) =>
      relevantIds.includes(l.id),
    );

    res.json({
      learnings: relevantLearnings,
      prompt,
      scopes: relevantScopes,
      usage: {
        inputTokens: (scopeResult.usage?.inputTokens || 0) + (learningResult.usage?.inputTokens || 0),
        outputTokens: (scopeResult.usage?.outputTokens || 0) + (learningResult.usage?.outputTokens || 0),
        totalTokens: (scopeResult.usage?.totalTokens || 0) + (learningResult.usage?.totalTokens || 0),
      },
    });
  } catch (error: any) {
    console.error("POST relevant-learnings error:", error);
    res.status(500).json({ error: error.message || "Internal Server Error" });
  }
}