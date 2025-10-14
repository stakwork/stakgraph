import { ask_question } from "./ask.js";
import { decomposeAndAsk, QUESTIONS } from "./questions.js";
import { recomposeAnswer, RecomposedAnswer } from "./answer.js";
import { LEARN_HTML } from "./learn.js";
import * as G from "../../graph/graph.js";
import { db } from "../../graph/neo4j.js";
import { vectorizeQuery } from "../../vector/index.js";

/**
 * Utility function to map connected hints to the expected format
 */
function mapConnectedHints(connected_hints: any[]) {
  return connected_hints.map((hint: any) => ({
    question: hint.properties.question || hint.properties.name,
    answer: hint.properties.body || "",
    hint_ref_id: hint.ref_id || hint.properties.ref_id,
    reused: true,
    reused_question: hint.properties.question || hint.properties.name,
    edges_added: 0,
    linked_ref_ids: [],
    usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
  }));
}

/**
 * Cache Control Examples:
 *
 * // Use cached result if available (default behavior)
 * await ask_prompt("How does auth work?");
 *
 * // Force replacement of existing result regardless of cache
 * await ask_prompt("How does auth work?", undefined, undefined, { forceRefresh: true });
 *
 * // Replace existing result only if older than 24 hours
 * await ask_prompt("How does auth work?", undefined, undefined, { maxAgeHours: 24 });
 *
 * // Replace existing result only if older than 1 hour
 * await ask_prompt("How does auth work?", undefined, undefined, { maxAgeHours: 1 });
 *
 * // HTTP API usage:
 * // GET /ask?question=How%20does%20auth%20work?&maxAgeHours=24
 * // GET /ask?question=How%20does%20auth%20work?&forceRefresh=true
 */

export {
  QUESTIONS,
  ask_question,
  decomposeAndAsk,
  recomposeAnswer,
  LEARN_HTML,
};

export const PROMPT_SIMILARITY_THRESHOLD = 0.9;
export const QUESTION_SIMILARITY_THRESHOLD = 0.78;

export interface CacheControlOptions {
  maxAgeHours?: number; // Maximum age in hours for cached results
  forceRefresh?: boolean; // Force generation of new results regardless of cache
  forceCache?: boolean; // Force use of cached results, or return nothing
}

/**
 * Ask a prompt and get a recomposed answer with hints.
 *
 * @param prompt - The question or prompt to ask
 * @param provider - Optional LLM provider to use
 * @param similarityThreshold - Threshold for similarity matching (default: 0.78)
 * @param cacheControl - Optional cache control settings:
 *   - maxAgeHours: Maximum age in hours for cached results (will replace existing if older)
 *   - forceRefresh: Force replacement of existing results regardless of cache age
 * @returns Promise<RecomposedAnswer> - The answer with connected hints
 */
export async function ask_prompt(
  prompt: string,
  provider?: string,
  similarityThreshold: number = QUESTION_SIMILARITY_THRESHOLD,
  cacheControl?: CacheControlOptions
): Promise<RecomposedAnswer> {
  // first get a 0.95 match
  const existing = await G.search(
    prompt,
    5,
    ["Prompt", "Hint"],
    false,
    100000,
    "vector",
    "json"
  );
  if (Array.isArray(existing)) {
    console.log(">> existing prompts and hints::");
    existing.forEach((e: any) =>
      console.log(e.properties.question, e.properties.score, e.node_type)
    );
  }
  let existingRefIdToReplace: string | null = null;

  if (Array.isArray(existing) && existing.length > 0) {
    const top: any = existing[0];
    // THIS threshold is hardcoded because we want to reuse the answer if it's very similar to the prompt
    if (
      top.properties.score &&
      top.properties.score >= PROMPT_SIMILARITY_THRESHOLD
    ) {
      // Check cache control options
      if (cacheControl?.forceRefresh) {
        console.log(
          ">> Cache control: forceRefresh=true, replacing existing answer"
        );
        existingRefIdToReplace = top.ref_id;
      } else if (cacheControl?.maxAgeHours) {
        const nodeAge = top.properties.date_added_to_graph;
        if (nodeAge) {
          const currentTime = Date.now() / 1000; // Convert to seconds
          const ageInHours = (currentTime - nodeAge) / 3600; // Convert to hours

          if (ageInHours > cacheControl.maxAgeHours) {
            console.log(
              `>> Cache control: node age ${ageInHours.toFixed(
                2
              )}h exceeds maxAge ${
                cacheControl.maxAgeHours
              }h, replacing existing answer`
            );
            existingRefIdToReplace = top.ref_id;
          } else {
            console.log(
              `>> Cache control: node age ${ageInHours.toFixed(
                2
              )}h within maxAge ${
                cacheControl.maxAgeHours
              }h, using cached answer`
            );
            // Fetch connected hints (sub_answers) for this existing prompt
            const connected_hints = await db.get_connected_hints(top.ref_id);
            const hints = mapConnectedHints(connected_hints);
            const totalUsage = hints.reduce(
              (acc, h) => ({
                inputTokens: acc.inputTokens + h.usage.inputTokens,
                outputTokens: acc.outputTokens + h.usage.outputTokens,
                totalTokens: acc.totalTokens + h.usage.totalTokens,
              }),
              { inputTokens: 0, outputTokens: 0, totalTokens: 0 }
            );

            return {
              answer: top.properties.body,
              hints,
              ref_id: top.ref_id,
              usage: totalUsage,
            };
          }
        } else {
          console.log(
            ">> Cache control: no date_added_to_graph property found, using cached answer"
          );
          // Fetch connected hints (sub_answers) for this existing prompt
          const connected_hints = await db.get_connected_hints(top.ref_id);
          const hints = mapConnectedHints(connected_hints);
          const totalUsage = hints.reduce(
            (acc, h) => ({
              inputTokens: acc.inputTokens + h.usage.inputTokens,
              outputTokens: acc.outputTokens + h.usage.outputTokens,
              totalTokens: acc.totalTokens + h.usage.totalTokens,
            }),
            { inputTokens: 0, outputTokens: 0, totalTokens: 0 }
          );

          return {
            answer: top.properties.body,
            hints,
            ref_id: top.ref_id,
            usage: totalUsage,
          };
        }
      } else {
        // No cache control specified, use cached answer
        console.log(">> No cache control specified, using cached answer");
        // Fetch connected hints (sub_answers) for this existing prompt
        const connected_hints = await db.get_connected_hints(top.ref_id);
        const hints = mapConnectedHints(connected_hints);
        const totalUsage = hints.reduce(
          (acc, h) => ({
            inputTokens: acc.inputTokens + h.usage.inputTokens,
            outputTokens: acc.outputTokens + h.usage.outputTokens,
            totalTokens: acc.totalTokens + h.usage.totalTokens,
          }),
          { inputTokens: 0, outputTokens: 0, totalTokens: 0 }
        );

        return {
          answer: top.properties.body,
          hints,
          ref_id: top.ref_id,
          usage: totalUsage,
        };
      }
    }
  }

  // If forceCache is true and no cached result was found, return empty result
  if (cacheControl?.forceCache) {
    console.log(">> Cache control: forceCache=true, no cached result found, returning empty result");
    return {
      answer: "",
      hints: [],
      usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
    };
  }

  // then decompose and ask
  try {
    const answers = await decomposeAndAsk(
      prompt,
      similarityThreshold,
      provider
    );
    const answer = await recomposeAnswer(prompt, answers, provider);

    // If we need to replace an existing node, delete it just before creating the new one
    if (existingRefIdToReplace) {
      console.log(
        `>> Deleting existing node with ref_id: ${existingRefIdToReplace}`
      );
      const deletedCount = await db.delete_node_by_ref_id(
        existingRefIdToReplace
      );
      console.log(`>> Deleted ${deletedCount} node(s)`);
    }

    const embeddings = await vectorizeQuery(prompt);
    const created = await db.create_prompt(prompt, answer.answer, embeddings);

    for (const hint of answer.hints) {
      // Create edge from main prompt to sub answer hint
      console.log(
        ">> creating edge from main prompt to hint",
        created.ref_id,
        hint.hint_ref_id
      );
      await db.createEdgesDirectly(created.ref_id, [
        {
          ref_id: hint.hint_ref_id,
          relevancy: 0.8, // Sub answers are highly relevant to the main prompt
        },
      ]);
    }

    return {
      ...answer,
      ref_id: created.ref_id,
    };
  } catch (error) {
    console.error("Ask Prompt Error:", error);
    throw error;
  }
}

export async function learnings(question: string) {
  const prompts = await G.search(
    question,
    25,
    ["Prompt"],
    false,
    100000,
    "vector",
    "json"
  );
  const hints = await G.search(
    question,
    25,
    ["Hint"],
    false,
    100000,
    "vector",
    "json"
  );

  // Extract only the question field (or name if no question) from node properties
  const extractQuestion = (node: any) => {
    return node?.properties?.question || node?.properties?.name || "";
  };

  return {
    prompts: Array.isArray(prompts) ? prompts.map(extractQuestion) : [],
    hints: Array.isArray(hints) ? hints.map(extractQuestion) : [],
  };
}

/*

NOTES

- ask: decompose into questions
  - question: search for existing question match
    - if not found: explore, link edges
- answer: recompose

*/
