import {
  callModel,
  getApiKeyForProvider,
  Provider,
} from "../../aieo/src/index.js";

function FILTER_PROMPT(originalPrompt: string, candidatePairs: string) {
  return `
You are a relevance filter for a code knowledge system. Your task is to identify which cached question-answer pair (if any) best addresses the user's current request. The answer MUST contain specific references to the codebase that would DEFINITELY help answer the user's request.

**Original User Request:**
${originalPrompt}

**Candidate Question-Answer Pairs:**
${candidatePairs}

**Instructions:**
1. Evaluate each candidate's relevance to the original request
2. A match is relevant if it would help answer or implement the user's request
3. Consider both semantic similarity AND practical utility
4. If multiple candidates are relevant, choose the most specific/actionable one
5. If NO candidates are highly relevant to the user's request, return "NO_MATCH"
6. DO NOT RETURN A CANDIDATE UNLESS IT IS HIGHLY RELEVANT!!!

**Evaluation Criteria:**
- Does this Q&A address the same business functionality?
- Would the answer help implement the user's request?
- Is the context/scope similar enough to be useful?

**Response Format:**
Return ONLY the exact question text of the most relevant match, or "NO_MATCH" if none are suitable.

**Examples:**
- If user asks "build login flow" and candidate is "How does user authentication work in this system?" → return the question text
- If user asks "add payment processing" and candidates are about user management → return "NO_MATCH"

IMPORTANT: Return ONLY the exact question text of the most relevant match, or "NO_MATCH" if none are suitable.
`;
}

export type FilteredAnswer = "NO_MATCH" | string;

export async function filterAnswers(
  qas: string,
  originalPrompt: string,
  llm_provider?: string
): Promise<{
  answer: FilteredAnswer;
  usage: { inputTokens: number; outputTokens: number; totalTokens: number };
}> {
  console.log(">> filterAnswers!!!!");
  const provider = (llm_provider || "anthropic") as Provider;
  const apiKey = getApiKeyForProvider(provider);
  const result = await callModel({
    provider,
    apiKey,
    messages: [{ role: "user", content: FILTER_PROMPT(originalPrompt, qas) }],
  });
  return {
    answer: result.text,
    usage: result.usage,
  };
}
