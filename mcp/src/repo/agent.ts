import {
  generateText,
  hasToolCall,
  ModelMessage,
  StopCondition,
  ToolSet,
  StepResult,
} from "ai";
import { getModel, getApiKeyForProvider, Provider } from "../aieo/src/index.js";
import { get_tools, ToolsConfig } from "./tools.js";
import { ContextResult } from "../tools/types.js";

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

ALWAYS USE EITHER THE final_answer OR ask_clarifying_questions TOOL AT THE END!`;

function logStep(contents: any) {
  if (!Array.isArray(contents)) return;
  for (const content of contents) {
    if (content.type === "tool-call" && content.toolName !== "final_answer") {
      console.log("TOOL CALL:", content.toolName, ":", content.input);
    }
    console.log("CONTENT:", JSON.stringify(content, null, 2));
  }
}

function extractFinalAnswer(steps: StepResult<ToolSet>[]): string {
  let lastText = "";

  // Collect last text content as fallback
  for (const step of steps) {
    for (const item of step.content) {
      if (item.type === "text" && item.text && item.text.trim().length > 0) {
        lastText = item.text.trim();
      }
    }
  }

  // Search for ask_clarifying_questions or final_answer tool result (reverse order for efficiency)
  for (let i = steps.length - 1; i >= 0; i--) {
    // Check for ask_clarifying_questions first (higher priority)
    const askQuestionsResult = steps[i].content.find(
      (c) =>
        c.type === "tool-result" && c.toolName === "ask_clarifying_questions"
    );
    if (askQuestionsResult) {
      return (askQuestionsResult as any).output;
    }

    // Then check for final_answer
    const finalAnswerResult = steps[i].content.find(
      (c) => c.type === "tool-result" && c.toolName === "final_answer"
    );
    if (finalAnswerResult) {
      return (finalAnswerResult as any).output;
    }
  }

  // Fallback to last text if neither tool was found
  if (lastText) {
    console.warn(
      "No final_answer or ask_clarifying_questions tool call detected; falling back to last text."
    );
    return `${lastText}\n\n(Note: Model did not invoke final_answer tool; using last text as final answer.)`;
  }

  return "";
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

  let stopWhen: StopCondition<ToolSet> | StopCondition<ToolSet>[] =
    hasToolCall("final_answer");
  if (toolsConfig?.ask_clarifying_questions) {
    system = ASK_CLARIFYING_QUESTIONS_SYSTEM;
    stopWhen = [
      hasToolCall("final_answer"),
      hasToolCall("ask_clarifying_questions"),
    ];
  }

  for (const tool of Object.keys(tools)) {
    console.log("===> tool", tool, "===>", tools[tool].description);
  }

  const { steps, totalUsage } = await generateText({
    model,
    tools,
    prompt,
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
