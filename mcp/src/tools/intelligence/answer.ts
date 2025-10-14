import {
  callModel,
  getApiKeyForProvider,
  ModelMessage,
  Provider,
} from "../../aieo/src/index.js";
import { Answer } from "./ask.js";

function RECOMPOSE_PROMPT(user_query: string, qas: string): string {
  return `
You are a technical documentation synthesizer. Your task is to combine fragmented search results into a comprehensive, actionable response while identifying the most important technical references.

**Original User Request:** ${user_query}

**Search Questions & Results:** 
\`\`\`
${qas}
\`\`\`

Your goal is to:
1. Synthesize the fragmented information into a coherent, step-by-step response
2. Extract and highlight the most important technical references (files, functions, APIs, components, etc.)
3. Provide actionable guidance that directly addresses the user's request

## Instructions:

### A. Technical Reference Extraction
First, identify and categorize ALL technical references found in the search results:
- **Files/Modules** (file paths)
- **Functions/Methods/Components**: (function names) 
- **Data Models**: (schemas, tables, interfaces, etc.)
- **Endpoint/Request**: (API endpoints, requests, etc.)

### B. Response Synthesis
Create a comprehensive response that:
- Directly answers the original user request
- Integrates information from multiple search results
- Provides clear implementation steps
- References specific code entities where available

## Quality Guidelines:

- Prioritize references that appear across multiple search results
- Focus on concrete, actionable technical details
- Eliminate redundant information while preserving important context
- If search results are incomplete, clearly indicate what additional research is needed
- Maintain the technical depth appropriate for the user's request

Please synthesize the information and provide the structured response.

**IMPORTANT:**
IF YOU ARE ASKED TO BUILD A FEATURE IN THE Original User Request, DONT WORRY ABOUT ACTUALLY WRITING CODE.
YOUR JOB IS TO GENERATE CONCISE INSIGHTS, GUIDANCE, AND POINTERS TO RELEVANTS PARTS OF THE CODEBASE.
`;
}

export interface RecomposedAnswer {
  answer: string;
  hints: Answer[];
  ref_id?: string;
  usage: { inputTokens: number; outputTokens: number; totalTokens: number };
}

export async function recomposeAnswer(
  user_query: string,
  answers: Answer[],
  llm_provider?: string
): Promise<RecomposedAnswer> {
  let qas = "";
  for (const answer of answers) {
    qas +=
      "Question: " + answer.question + "\n" + "Answer: " + answer.answer + "\n";
  }
  const content = RECOMPOSE_PROMPT(user_query, qas);
  const provider = llm_provider || "anthropic";
  const apiKey = getApiKeyForProvider(provider as Provider);
  const messages: ModelMessage[] = [{ role: "user", content }];
  const result = await callModel({
    provider: provider as Provider,
    apiKey,
    messages,
  });
  const totalUsage = {
    inputTokens: result.usage.inputTokens,
    outputTokens: result.usage.outputTokens,
    totalTokens: result.usage.totalTokens,
  };
  for (const answer of answers) {
    totalUsage.inputTokens += answer.usage.inputTokens;
    totalUsage.outputTokens += answer.usage.outputTokens;
    totalUsage.totalTokens += answer.usage.totalTokens;
  }
  return {
    answer: result.text,
    hints: answers,
    usage: totalUsage,
  };
}
