import { generateText, ModelMessage, StopCondition, ToolSet } from "ai";
import { getModel, getApiKeyForProvider, Provider } from "../aieo/src/index.js";
import { get_tools, ToolsConfig } from "./tools.js";
import { ContextResult } from "../tools/types.js";
import {
  appendTextToPrompt,
  logStep,
  extractFinalAnswer,
  createHasEndMarkerCondition,
  createHasAskQuestionsCondition,
} from "./utils.js";

const DEFAULT_SYSTEM = `You are a code exploration assistant. Please use the provided tools to answer the user's prompt.

CRITICAL: When you are ready to provide your final answer, output your complete response followed by [END_OF_ANSWER] on a new line.

Example format:
Your complete answer here with all details, explanations, and code examples if needed.

[END_OF_ANSWER]

Write your answer directly as text and end with [END_OF_ANSWER].`;

const ASK_CLARIFYING_QUESTIONS_SYSTEM = `You are a code exploration assistant. Please use the provided tools to answer the user's prompt.

CRITICAL: When you finish exploring, you MUST do ONE of these:

1. Write your complete answer as text, then output [END_OF_ANSWER] on a new line
2. Call ask_clarifying_questions tool with format: { "questions": [...] }

Call ask_clarifying_questions when:
 - The user's query is too general
 - You can provide a better answer by first gathering more information from the user
 - Your technical exploration has revealed multiple possible approaches, and you want the user's input on which to choose

Otherwise, provide your answer directly followed by [END_OF_ANSWER].`;

export async function get_context(
  prompt: string | ModelMessage[],
  repoPath: string,
  pat: string | undefined,
  toolsConfig?: ToolsConfig,
  systemOverride?: string
): Promise<ContextResult> {
  const startTime = Date.now();
  const provider = process.env.LLM_PROVIDER || "anthropic";
  const apiKey = getApiKeyForProvider(provider);
  const model = await getModel(provider as Provider, apiKey as string);
  console.log("===> model", model);

  const tools = get_tools(repoPath, apiKey, pat, toolsConfig);
  let system = systemOverride || DEFAULT_SYSTEM;

  const hasEndMarker = createHasEndMarkerCondition<typeof tools>();
  const hasAskQuestions = createHasAskQuestionsCondition<typeof tools>();

  let stopWhen: StopCondition<ToolSet> | StopCondition<ToolSet>[] =
    hasEndMarker;
  let stopSequences: string[] = ["[END_OF_ANSWER]"];
  let finalPrompt: string | ModelMessage[] = prompt;

  if (toolsConfig?.ask_clarifying_questions) {
    system = ASK_CLARIFYING_QUESTIONS_SYSTEM;
    stopWhen = [hasEndMarker, hasAskQuestions];

    // Add clarifying questions text to prompt
    finalPrompt = appendTextToPrompt(
      prompt,
      " After exploring a bit, ask clarifying questions if needed."
    );
  }

  for (const tool of Object.keys(tools)) {
    console.log("===> tool", tool, "===>", tools[tool].description);
  }

  const { steps, totalUsage } = await generateText({
    model,
    tools,
    prompt: finalPrompt,
    system,
    stopWhen,
    stopSequences,
    onStepFinish: (sf) => logStep(sf.content),
  });

  const final = extractFinalAnswer(steps);

  const endTime = Date.now();
  const duration = endTime - startTime;
  console.log(
    `⏱️ get_context completed in ${duration}ms (${(duration / 1000).toFixed(
      2
    )}s)`
  );

  return {
    final: final.answer,
    tool_use: final.tool_use,
    content: final.answer,
    usage: {
      inputTokens: totalUsage.inputTokens || 0,
      outputTokens: totalUsage.outputTokens || 0,
      totalTokens: totalUsage.totalTokens || 0,
    },
  };
}

/*
curl -X POST -H "Content-Type: application/json" -d '{"repo_url": "https://github.com/stakwork/hive", "prompt": "how does auth work in the repo"}' "http://localhost:3355/repo/agent"

curl -X POST -H "Content-Type: application/json" -d '{
  "repo_url": "https://github.com/stakwork/hive",
  "prompt": "how does auth work in the repo? I don'\''t need a detailed answer, just a high-level overview. ANSWER QUICKLY!",
  "toolsConfig": {
    "final_answer": "Provide the final answer to the user. YOU **MUST** CALL THIS TOOL AT THE END OF YOUR EXPLORATION. Please explore quickly, only read a couple files summaries. YOU MUST START THE FINAL ANSWER WITH 3 ROCKET EMOJIS!",
    "repo_overview": "",
    "file_summary": ""
  }
}' "http://localhost:3355/repo/agent"

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/stakwork/hive",
    "prompt": "please call the bash tool to make sure it works. List my docker containers currently running. Then call final_answer to say the answer.",
    "toolsConfig": {
      "bash": ""
    }
  }' \
  "http://localhost:3355/repo/agent"

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/stakwork/hive",
    "prompt": "please call the web_search tool to make sure it works. Search for Evan Feenstra online. Then call final_answer to say the answer.",
    "toolsConfig": {
      "web_search": ""
    }
  }' \
  "http://localhost:3355/repo/agent"

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/stakwork/hive",
    "prompt": "i want to build a user inbox to show recent activity and notifications.",
    "toolsConfig": {
      "ask_clarifying_questions": true
    }
  }' \
  "http://localhost:3355/repo/agent"

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/stakwork/hive",
    "prompt": "i want to build a sub-account feature. please tell me a brief technical architecture for this.",
    "toolsConfig": {
      "ask_clarifying_questions": true
    }
  }' \
  "http://localhost:3355/repo/agent"

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/stakwork/hive",
    "prompt": "i want to build a sub-account feature."
  }' \
  "http://localhost:3355/repo/agent"
*/
