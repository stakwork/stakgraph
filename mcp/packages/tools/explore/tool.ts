import { generateText, tool, hasToolCall } from "ai";
import { getModel, getApiKeyForProvider, Provider } from "@stakgraph/aieo";
import { EXPLORER } from "./prompts.js";
import { z } from "zod";
import * as G from "../graph/graph.js";

/*
curl "http://localhost:3000/explore?prompt=how%20does%20auth%20work%20in%20the%20repo"
*/

export async function get_context(prompt: string): Promise<string> {
  const provider = process.env.LLM_PROVIDER || "anthropic";
  const apiKey = getApiKeyForProvider(provider);
  const model = await getModel(provider as Provider, apiKey as string);
  // console.log("call claude:");
  const tools = {
    repo_overview: tool({
      description:
        "Get a high-level view of the codebase architecture and structure. Use this to understand the project layout and identify where specific functionality might be located. Call this when you need to: 1) Orient yourself in an unfamiliar codebase, 2) Locate which directories/files might contain relevant code for a user's question, 3) Understand the overall project structure before diving deeper. Don't call this if you already know which specific files you need to examine.",
      inputSchema: z.object({}),
      execute: async () => {
        try {
          return await G.get_repo_map("", "", "Repository", false);
        } catch (e) {
          return "Could not retrieve repository map";
        }
      },
    }),
    file_summary: tool({
      description:
        "Get a summary of what a specific file contains and its role in the codebase. Use this when you have identified a potentially relevant file and need to understand: 1) What functions/components it exports, 2) What its main responsibility is, 3) Whether it's worth exploring further for the user's question. Functions and top-level variables will be returned with their name and first 10 lines of code. Call this with a hypothesis like 'This file probably handles user authentication' or 'This looks like the main dashboard component'. Don't call this to browse random files.",
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
          return await G.get_file_map(file_path);
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
    // function_interface: {
    //   desciption:
    //     "Examine the interface and immediate dependencies of a specific function/component. Use this when you've identified a key function and need to understand: 1) Its parameters and return type, 2) What it directly calls, 3) How it fits into the larger system. Call this when you have a specific question like 'How does this authentication function work?' or 'What data does this component need?'. Don't call this for functions that seem peripheral to the user's question.",
    //   inputSchema: z.object({
    //     function_name: z
    //       .string()
    //       .describe("Name of the function/component to examine"),
    //     file_path: z.string().describe("File containing the function"),
    //     question: z
    //       .string()
    //       .describe(
    //         "Specific question you're trying to answer about this function"
    //       ),
    //   }),
    //   execute: async () => {
    //     return "function interface:";
    //   },
    // },
    finalAnswer: tool({
      // Define a tool that signals the end of the process
      description:
        "Provide the final answer to the user. ALWAYS include relevant files or function names in the answer. These hints will be used by the next model to actually build the feature.",
      inputSchema: z.object({ answer: z.string() }),
      execute: async ({ answer }: { answer: string }) => answer,
    }),
  };
  const { steps } = await generateText({
    model,
    tools,
    prompt,
    system: EXPLORER,
    stopWhen: hasToolCall("finalAnswer"),
    onStepFinish: (sf) => {
      console.log("step", JSON.stringify(sf.content, null, 2));
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
    const finalAnswer = step.content.find((c) => {
      return c.type === "tool-result" && c.toolName === "finalAnswer";
    });
    if (finalAnswer) {
      final = (finalAnswer as any).output;
    }
  }
  if (!final && lastText) {
    console.warn(
      "No finalAnswer tool call detected; falling back to last reasoning text."
    );
    final = `${lastText}\n\n(Note: Model did not invoke finalAnswer tool; using last reasoning text as answer.)`;
  }
  // console.log("FINAL", final);
  return final;
}
