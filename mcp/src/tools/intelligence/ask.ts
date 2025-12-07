import { getApiKeyForProvider, Provider } from "../../aieo/src/provider.js";
import { callGenerateObject, ModelMessage } from "../../aieo/src/index.js";
import { z } from "zod";
import { db } from "../../graph/neo4j.js";
import { get_context_explore } from "../explore/tool.js";
import { vectorizeQuery } from "../../vector/index.js";
import { create_hint_edges_llm } from "./seed.js";
import * as G from "../../graph/graph.js";
import { filterAnswers } from "./filter.js";
import { Persona } from "./persona.js";
import { Neo4jNode } from "../../graph/types.js";

/*

curl "http://localhost:3355/ask?question=how%20does%20auth%20work%20in%20the%20repo"

curl "http://localhost:3355/explore?prompt=how%20does%20auth%20work%20in%20the%20repo"

*/

/*
DECOMPOSE QUESTION into business context and specific implementation questions
*/

function DECOMPOSE_PROMPT(user_query: string) {
  return `
You are bridging business requirements to technical implementation. Given a user request:

**User Request:** ${user_query}

## Step 1: Business Context Analysis
Identify the core business functionality and user workflows involved.

## Step 2: Implementation Questions
Generate 1-5 specific questions that developers would need to answer. Make each question:
- **Short and concise**
- **Specific enough** to match existing cached answers
- **Workflow-oriented** (following user journeys)
- **Entity-specific** (mention likely code components)

YOU DO NOT HAVE TO WRITE 5 QUESTIONS. IF ITS A VERY SIMPLE USER REQUEST, A SINGLE QUESTION IS ENOUGH! Just try to boil it down to fundamental user flow(s).

**Question Types to Consider:**
- **User Workflows**: "How does a user [specific action] in this system?"
- **Data Handling**: "How is [business entity] data stored/validated/retrieved?"
- **Business Logic**: "What happens when [business event] occurs?"
- **Integration Points**: "How does [business process] connect with [other system/feature]?"
- **Permissions**: "What user roles can [perform action] and how is this enforced?"
- **UI Patterns**: "What UI components handle [user interaction type]?"

**Format each question for optimal embedding:**
- Use business terminology first, technical second. Focus on front-end terminology, since that is what the user is seeing and asking about.

**IMPORTANT:**
MAKE YOUR QUESTIONS SHORT AND CONCISE. DO NOT ASSUME THINGS ABOUT THE CODEBASE THAT YOU DON'T KNOW.

**EXPECTED OUTPUT FORMAT:**
{
  "business_context": "Brief description of core functionality",
  "questions": [
    "How does user registration workflow handle email verification and account activation?",
    "What user permission system controls access to workspace creation features?",
    ...
  ]
}
`;
}

export interface Answer {
  question: string;
  answer: string;
  hint_ref_id: string;
  reused: boolean;
  reused_question?: string;
  edges_added: number;
  linked_ref_ids: string[];
  usage: { inputTokens: number; outputTokens: number; totalTokens: number };
}

function cached_answer(question: string, e: Neo4jNode): Answer {
  return {
    question,
    answer: e.properties.body,
    hint_ref_id: e.ref_id as string,
    reused: true,
    reused_question: e.properties.question,
    edges_added: 0,
    linked_ref_ids: [],
    usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
  };
}

export const QUESTION_HIGHLY_RELEVANT_THRESHOLD = 0.94;

interface FilterByRelevanceFromCacheResult {
  cachedAnswer?: Answer;
  reexplore: boolean;
  filterUsage?: {
    inputTokens: number;
    outputTokens: number;
    totalTokens: number;
  };
}

async function filter_by_relevance_from_cache(
  question: string,
  similarityThreshold: number,
  provider?: string,
  originalPrompt?: string,
  persona?: Persona
): Promise<FilterByRelevanceFromCacheResult | undefined> {
  const existingAll = await G.search(
    question,
    5,
    ["Hint"],
    false,
    100000,
    "vector",
    "json"
  );
  if (!Array.isArray(existingAll)) {
    return;
  }
  const existing = existingAll.filter(
    (e: any) => e.properties.score && e.properties.score >= similarityThreshold
  );
  const candidates = persona
    ? existing.filter((e: any) => (e.properties.persona || "PM") === persona)
    : existing;
  if (Array.isArray(candidates) && candidates.length > 0) {
    if (originalPrompt) {
      let qas = "";
      candidates.forEach((e: any) => {
        qas += `**Question:** ${e.properties.question}\n**Answer:** ${e.properties.body}\n\n`;
      });
      const filterResult = await filterAnswers(qas, originalPrompt, provider);
      if (filterResult.answer === "NO_MATCH") {
        return;
      }
      const top: any = candidates.find(
        (e: any) => e.properties.question === filterResult.answer
      );
      if (!top) {
        return;
      }
      if (top.properties.score > QUESTION_HIGHLY_RELEVANT_THRESHOLD) {
        console.log(
          ">> REUSED RELEVANT question!!!:",
          question,
          ">>",
          top.properties.question
        );
        return {
          cachedAnswer: cached_answer(question, top),
          reexplore: false,
          filterUsage: filterResult.usage,
        };
      } else {
        console.log(
          ">> REUSED RELEVANT question but not highly relevant ---- RE-EXPLORING!!!:",
          question,
          ">>",
          top.properties.question
        );
        return {
          cachedAnswer: cached_answer(question, top),
          reexplore: true,
          filterUsage: filterResult.usage,
        };
      }
    }
    const top: any = candidates[0];
    console.log(">> REUSED question:", question, ">>", top.properties.question);
    return { cachedAnswer: cached_answer(question, top), reexplore: false };
  }
}

export async function ask_question(
  question: string,
  similarityThreshold: number,
  provider?: string,
  originalPrompt?: string,
  persona?: Persona
): Promise<Answer> {
  const filtered = await filter_by_relevance_from_cache(
    question,
    similarityThreshold,
    provider,
    originalPrompt,
    persona
  );
  if (filtered && filtered.cachedAnswer && !filtered.reexplore) {
    if (filtered.filterUsage) {
      filtered.cachedAnswer.usage = filtered.filterUsage;
    }
    return filtered.cachedAnswer;
  }
  let q: string | ModelMessage[] = question;
  let reexplore = false;
  if (filtered && filtered.cachedAnswer && filtered.reexplore) {
    const msgs: ModelMessage[] = [
      { role: "user", content: filtered.cachedAnswer.question },
      { role: "assistant", content: filtered.cachedAnswer.answer },
      { role: "user", content: question },
    ];
    q = msgs;
    reexplore = true;
  }
  console.log(">> NEW question:", question);
  const ctx = await get_context_explore(q, reexplore, false, provider);
  const answer = ctx.final;
  const embeddings = await vectorizeQuery(question);
  const created = await db.create_hint(question, answer, embeddings, "PM");
  let edges_added = 0;
  let linked_ref_ids: string[] = [];
  let totalUsage = {
    inputTokens: ctx.usage.inputTokens,
    outputTokens: ctx.usage.outputTokens,
    totalTokens: ctx.usage.totalTokens,
  };
  try {
    const r = await create_hint_edges_llm(created.ref_id, answer, provider);
    edges_added = r.edges_added;
    linked_ref_ids = r.linked_ref_ids;
    totalUsage.inputTokens += r.usage.inputTokens;
    totalUsage.outputTokens += r.usage.outputTokens;
    totalUsage.totalTokens += r.usage.totalTokens;
  } catch (e) {
    console.error("Failed to create edges from hint", e);
  }
  return {
    question,
    answer,
    hint_ref_id: created.ref_id,
    reused: false,
    edges_added,
    linked_ref_ids,
    usage: totalUsage,
  };
}

interface DecomposedQuestion {
  business_context: string;
  questions: string[];
}
export async function decomposeQuestion(
  question: string,
  llm_provider?: string
): Promise<DecomposedQuestion> {
  const provider = llm_provider ? llm_provider : "anthropic";
  const apiKey = getApiKeyForProvider(provider);
  const schema = z.object({
    business_context: z.string(),
    questions: z.array(z.string()),
  });
  const result = await callGenerateObject({
    provider: provider as Provider,
    apiKey,
    prompt: DECOMPOSE_PROMPT(question),
    schema,
  });
  return result.object;
}
