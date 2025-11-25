import {
  generateText,
  hasToolCall,
  ModelMessage,
  StopCondition,
  ToolSet,
} from "ai";
import { getModel, getApiKeyForProvider, Provider } from "../aieo/src/index.js";
import { get_tools, ToolsConfig } from "./tools.js";
import { ContextResult } from "../tools/types.js";
import { appendTextToPrompt, logStep, extractFinalAnswer } from "./utils.js";

// const DEFAULT_SYSTEM = `You are a code exploration assistant. Please use the provided tools to answer the user's prompt. ALWAYS USE THE final_answer TOOL AT THE END OF YOUR EXPLORATION. Do NOT write to a document with your answer, instead ALWAYS finish with the final_answer tool!!!!!!!!!!`;

// const ASK_CLARIFYING_QUESTIONS_SYSTEM = `You are a code exploration assistant. Please use the provided tools to answer the user's prompt. ALWAYS USE EITHER THE final_answer OR ask_clarifying_questions TOOL AT THE END OF YOUR EXPLORATION. Do NOT write to a document with your answer, instead ALWAYS finish with the final_answer or ask_clarifying_questions tool!!!!!!!!!!`;

const DEFAULT_SYSTEM = `You are a code exploration assistant. Please use the provided tools to answer the user's prompt. 

CRITICAL: When you are ready to provide your final answer, you MUST call the final_answer tool with a properly formatted JSON object.

CORRECT FORMAT:
{
  "answer": "Your complete answer here with all details, explanations, and code examples if needed"
}

WRONG FORMAT (DO NOT DO THIS):
{}
{ }
Just calling final_answer without parameters

The 'answer' field is REQUIRED and must contain your ENTIRE response as a string.

ALWAYS USE THE final_answer TOOL AT THE END OF YOUR EXPLORATION with the answer parameter filled in!`;

const ASK_CLARIFYING_QUESTIONS_SYSTEM = `You are a code exploration assistant. Please use the provided tools to answer the user's prompt. 

CRITICAL: When you finish exploring, you MUST call EITHER:

1. final_answer tool with format: { "answer": "Your complete response here" }
2. ask_clarifying_questions tool with format: { "questions": [...] }

The parameters are REQUIRED. Do NOT call these tools with empty objects {}!

Reasons to call ask_clarifying_questions:
 - The user's query is too general.
 - You think you can provide a better answer by first gathering more information from the user.
 - MOST IMPORTANT: Your technical exploration has revealed that there are multiple possible approaches to take, and you want to get the user's input on which one to choose.

ALWAYS USE EITHER THE final_answer OR ask_clarifying_questions TOOL AT THE END!`;

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

  // FIXME: flexible stopWhen: final_answer might be called with {} and then text as answer after final_answer is called.
  /*
    const hasAnswer: StopCondition<typeof tools> = ({ steps }) => {
      // Stop when the model generates text containing "ANSWER:"
      return steps.some(step => step.text?.includes('ANSWER:')) ?? false;
    };
  */
  let stopWhen: StopCondition<ToolSet> | StopCondition<ToolSet>[] =
    hasToolCall("final_answer");
  let finalPrompt: string | ModelMessage[] = prompt;

  if (toolsConfig?.ask_clarifying_questions) {
    system = ASK_CLARIFYING_QUESTIONS_SYSTEM;
    stopWhen = [
      hasToolCall("final_answer"),
      hasToolCall("ask_clarifying_questions"),
    ];

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
    final,
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
*/
