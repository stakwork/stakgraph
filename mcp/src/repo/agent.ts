import { Request, Response } from "express";
import { cloneOrUpdateRepo } from "./clone.js";
import { generateText, hasToolCall, ModelMessage } from "ai";
import { getModel, getApiKeyForProvider, Provider } from "../aieo/src/index.js";
import { get_tools } from "./tools.js";

/*
curl -X POST -H "Content-Type: application/json" -d '{"repo_url": "https://github.com/stakwork/hive", "prompt": "how does auth work in the repo"}' "http://localhost:3355/repo/agent"
*/

export async function repo_agent(req: Request, res: Response) {
  console.log("===> repo_agent", req.body);
  try {
    const repoUrl = req.body.repo_url as string;
    const username = req.body.username as string | undefined;
    const pat = req.body.pat as string | undefined;
    const commit = req.body.commit as string | undefined;
    const branch = req.body.branch as string | undefined;
    const prompt = req.body.prompt as string | undefined;
    if (!prompt) {
      res.status(400).json({ error: "Missing prompt" });
      return;
    }

    const repoDir = await cloneOrUpdateRepo(repoUrl, username, pat, commit);

    console.log(`===> POST /repo/agent ${repoDir}`);

    const final_answer = await get_context(prompt, repoDir, pat);

    console.log("===> final_answer", final_answer);
    res.json({ success: true, final_answer });
  } catch (e) {
    console.error("Error in repo_agent", e);
    res.status(500).json({ error: "Internal server error" });
  }
}

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
  pat: string | undefined
): Promise<string> {
  const startTime = Date.now();
  const provider = process.env.LLM_PROVIDER || "anthropic";
  const apiKey = getApiKeyForProvider(provider);
  const model = await getModel(provider as Provider, apiKey as string);
  console.log("===> model", model);

  const tools = get_tools(repoPath, apiKey, pat);
  const system = `You are a code exploration assistant. Please use the provided tools to answer the user's prompt.`;
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
