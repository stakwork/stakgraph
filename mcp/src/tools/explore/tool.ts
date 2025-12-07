import { generateText, tool, hasToolCall, ModelMessage } from "ai";
import {
  getModel,
  getApiKeyForProvider,
  Provider,
} from "../../aieo/src/provider.js";
import {
  EXPLORER,
  RE_EXPLORER,
  GENERAL_EXPLORER,
  GENERAL_FINAL_ANSWER_DESCRIPTION,
  FINAL_ANSWER_DESCRIPTION,
} from "./prompts.js";
import { z } from "zod";
import * as G from "../../graph/graph.js";
import { ContextResult } from "../types.js";

/*
curl "http://localhost:3000/explore?prompt=how%20does%20auth%20work%20in%20the%20repo"
*/

function logStep(contents: any) {
  return;
  if (!Array.isArray(contents)) return;
  for (const content of contents) {
    if (content.type === "tool-call") {
      if (content.toolName === "final_answer") {
        console.log("FINAL ANSWER:", content.input.answer);
      } else {
        console.log("TOOL CALL:", content.toolName, ":", content.input);
      }
    }
    if (content.type === "tool-result") {
      if (content.toolName !== "repo_overview") {
        console.log(content.output);
      }
      // console.log("TOOL RESULT", content.toolName, content.output);
    }
  }
}

export interface GeneralContextResult {
  summary: string;
  key_files: string[];
  features: string[];
}

interface ToolConfig {
  name: string;
  description?: string;
  tokenLimit?: number;
}

export async function get_context_explore(
  prompt: string | ModelMessage[],
  re_explore: boolean = false,
  general_explore: boolean = false,
  provider?: string
): Promise<ContextResult> {
  const effectiveProvider = provider || process.env.LLM_PROVIDER || "anthropic";
  const apiKey = getApiKeyForProvider(effectiveProvider);
  const model = await getModel(effectiveProvider as Provider, apiKey as string);
  // console.log("call claude:");
  const tools = {
    repo_overview: tool({
      description:
        "Get a high-level view of the codebase architecture and structure. Use this to understand the project layout and identify where specific functionality might be located. Call this when you need to: 1) Orient yourself in an unfamiliar codebase, 2) Locate which directories/files might contain relevant code for a user's question, 3) Understand the overall project structure before diving deeper. Don't call this if you already know which specific files you need to examine.",
      inputSchema: z.object({}),
      execute: async () => {
        // git ls-tree -r --name-only HEAD | tree -L 3 --fromfile
        try {
          return await G.get_repo_map("", "", "Repository", false);
        } catch (e) {
          return "Could not retrieve repository map";
        }
      },
    }),
    file_summary: tool({
      description:
        "Get a summary of what a specific file contains and its role in the codebase. Use this when you have identified a potentially relevant file and need to understand: 1) What functions/components it exports, 2) What its main responsibility is, 3) Whether it's worth exploring further for the user's question. Functions, imports, and top-level variables will be returned with their name and first 10 lines of code. If a summary can't be generated, the first 40 lines of the file will be returned. Call this with a hypothesis like 'This file probably handles user authentication' or 'This looks like the main dashboard component'. Don't call this to browse random files.",
      inputSchema: z.object({
        file_path: z.string().describe("Path to the file to summarize"),
        hypothesis: z
          .string()
          .describe(
            "What you think this file might contain or handle, based on its name/location"
          ),
      }),
      execute: async ({ file_path }: { file_path: string }) => {
        try {
          const file_map = await G.get_file_map(file_path);
          // limit to 75000 characters (for large files)
          return file_map.substring(0, 75000);
        } catch (e) {
          return "Bad file path";
        }
      },
    }),
    feature_map: tool({
      description:
        "Discover how a function/component connects to related code to form a complete feature. Use this ONLY when you need to understand: 1) How pieces of a feature work together, 2) The data flow through a system, 3) Dependencies that might be affected by changes. Call this when you have a hypothesis like 'I need to see how user data flows through the system' or 'I want to understand all the pieces involved in checkout process'. This ONLY works for Function or Component nodes. This is expensive - don't use it for general exploration.",
      inputSchema: z.object({
        starting_node: z
          .string()
          .describe("Name of the function/component to examine"),
        depth: z
          .number()
          .describe("How many levels of connections to explore. Default 1")
          .default(1),
        hypothesis: z
          .string()
          .describe(
            "What feature/workflow you think this reveals (e.g., 'user authentication flow', 'data validation pipeline')"
          ),
      }),
      execute: async ({
        starting_node,
        depth,
      }: {
        starting_node: string;
        depth: number;
      }) => {
        try {
          return await G.get_map({
            node_type: "Function",
            name: starting_node,
            file: "",
            ref_id: "",
            tests: false,
            depth: depth || 1,
            direction: "down",
            trim: [],
          });
        } catch (e) {
          return "Could not identify starting node";
        }
      },
    }),
    fulltext_search: tool({
      description:
        "Search the entire codebase for a specific term. Use this when you need to find a specific function, component, or file. Call this when the user provided specific text that might be present in the codebase. For example, if the query is 'Add a subtitle to the User Journeys page', you could call this with the query \"User Journeys\". Don't call this if you do not have specific text to search for",
      inputSchema: z.object({
        query: z.string().describe("The term to search for"),
      }),
      execute: async ({ query }: { query: string }) => {
        return await G.search(
          query,
          5,
          [],
          false,
          100000,
          "fulltext",
          "snippet",
          false
        );
      },
    }),
    final_answer: tool({
      // The tool that signals the end of the process
      description: general_explore
        ? GENERAL_FINAL_ANSWER_DESCRIPTION
        : FINAL_ANSWER_DESCRIPTION,
      inputSchema: z.object({ answer: z.string() }),
      execute: async ({ answer }: { answer: string }) => answer,
    }),
  };
  const system = general_explore
    ? GENERAL_EXPLORER
    : re_explore
    ? RE_EXPLORER
    : EXPLORER;
  const { steps, totalUsage } = await generateText({
    model,
    tools,
    prompt,
    system,
    stopWhen: hasToolCall("final_answer"),
    onStepFinish: (sf) => {
      // console.log("step", JSON.stringify(sf.content, null, 2));
      logStep(sf.content);
    },
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
  // console.log("FINAL", final);
  return {
    final,
    content: final,
    usage: {
      inputTokens: totalUsage.inputTokens || 0,
      outputTokens: totalUsage.outputTokens || 0,
      totalTokens: totalUsage.totalTokens || 0,
    },
  };
}

setTimeout(async () => {
  return;
  console.log("calling GENERAL get_context");
  const gres = await get_context_explore(
    "How does this repository work? Please provide a summary of the codebase, a few key files, and 50 core user stories.",
    false,
    true
  );
  console.log("GENERAL get_context result:", gres);
}, 5000);

/*

CALL MODEL [system, user]

-> tool call -> call the endpoint -> tool result

CALL MODEL[system, user, assistant: tool_call, user: tool_result]

-> tool call -> call the endpoint -> tool result

CALL MODEL[system, user, assistant: tool_call, user: tool_result, assistant: tool_call, user: tool_result]

FINALLY tool call "final_answer"

*/
