import { generateText, hasToolCall, ModelMessage } from "ai";
import { getModel, getApiKeyForProvider, Provider } from "../aieo/src/index.js";
import { get_tools, ToolsConfig } from "./tools.js";

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
*/

function logStep(contents: any) {
  if (!Array.isArray(contents)) return;
  for (const content of contents) {
    if (content.type === "tool-call" && content.toolName !== "final_answer") {
      console.log("TOOL CALL:", content.toolName, ":", content.input);
    }
  }
}

export async function get_context(
  prompt: string | ModelMessage[],
  repoPath: string,
  pat: string | undefined,
  toolsConfig?: ToolsConfig,
  systemOverride?: string
): Promise<string> {
  const startTime = Date.now();
  const provider = process.env.LLM_PROVIDER || "anthropic";
  const apiKey = getApiKeyForProvider(provider);
  const model = await getModel(provider as Provider, apiKey as string);
  console.log("===> model", model);

  const tools = get_tools(repoPath, apiKey, pat, toolsConfig);
  const system =
    systemOverride ||
    `You are a code exploration assistant. Please use the provided tools to answer the user's prompt.`;
  const { steps } = await generateText({
    model,
    tools,
    prompt,
    system,
    stopWhen: hasToolCall("final_answer"),
    onStepFinish: (sf) => logStep(sf.content),
  });
  let final = "";
  let lastText = "";
  for (const step of steps) {
    for (const item of step.content) {
      if (item.type === "text" && item.text && item.text.trim().length > 0) {
        lastText = item.text.trim();
      }
    }
  }
  steps.reverse();
  for (const step of steps) {
    // console.log("step", JSON.stringify(step.content, null, 2));
    const final_answer = step.content.find((c) => {
      return c.type === "tool-result" && c.toolName === "final_answer";
    });
    if (final_answer) {
      final = (final_answer as any).output;
    }
  }
  if (!final && lastText) {
    console.warn(
      "No final_answer tool call detected; falling back to last reasoning text."
    );
    final = `${lastText}\n\n(Note: Model did not invoke final_answer tool; using last reasoning text as answer.)`;
  }

  const endTime = Date.now();
  const duration = endTime - startTime;
  console.log(
    `⏱️ get_context completed in ${duration}ms (${(duration / 1000).toFixed(
      2
    )}s)`
  );

  return final;
}
