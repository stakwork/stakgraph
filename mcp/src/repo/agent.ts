import { generateText, ModelMessage, StopCondition, ToolSet } from "ai";
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

// Custom stop condition: final_answer might be called with {} (fail) but text after it is the real answer
function createHasAnswerCondition<T extends ToolSet>(): StopCondition<T> {
  return ({ steps }) => {
    let foundFinalAnswerCall = false;

    for (const step of steps) {
      for (const item of step.content) {
        // Check if final_answer tool was called
        if (item.type === "tool-call" && item.toolName === "final_answer") {
          foundFinalAnswerCall = true;
        }
        // If we found final_answer call before, and now there's text content, stop
        if (foundFinalAnswerCall && item.type === "text" && item.text?.trim()) {
          return true;
        }
        // Check for tool-result of final_answer (successful call with non-empty output)
        if (item.type === "tool-result" && item.toolName === "final_answer") {
          // Only stop if the output is not empty, otherwise wait for the next text content
          const output = (item as any).output;
          if (output && output.trim()) {
            return true;
          }
        }
      }
    }

    return false;
  };
}

function createHasAskQuestionsCondition<T extends ToolSet>(): StopCondition<T> {
  return ({ steps }) => {
    for (const step of steps) {
      for (const item of step.content) {
        // Check for successful ask_clarifying_questions call
        if (
          item.type === "tool-result" &&
          item.toolName === "ask_clarifying_questions"
        ) {
          return true;
        }
      }
    }
    return false;
  };
}

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

  const hasAnswer = createHasAnswerCondition<typeof tools>();
  const hasAskQuestions = createHasAskQuestionsCondition<typeof tools>();

  let stopWhen: StopCondition<ToolSet> | StopCondition<ToolSet>[] = hasAnswer;
  let finalPrompt: string | ModelMessage[] = prompt;

  if (toolsConfig?.ask_clarifying_questions) {
    system = ASK_CLARIFYING_QUESTIONS_SYSTEM;
    stopWhen = [hasAnswer, hasAskQuestions];

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
