import { generateText, tool, hasToolCall, ModelMessage } from "ai";
import {
  getModel,
  getApiKeyForProvider,
  Provider,
} from "../../aieo/src/index.js";
import * as prompts from "./prompts/index.js";
import { z } from "zod";
import { getRepoMap, getFileSummary, fulltextSearch } from "./tools.js";

function logStep(contents: any) {
  if (!Array.isArray(contents)) return;
  for (const content of contents) {
    if (content.type === "tool-call" && content.toolName !== "final_answer") {
      console.log("TOOL CALL:", content.toolName, ":", content.input);
    }
  }
}

export type RepoContextMode =
  | "generic"
  | "first_pass"
  | "features"
  | "services";

interface ContextConfig {
  file_lines: number;
  final_answer_description: string;
  system: string;
}

function getConfig(mode: RepoContextMode): ContextConfig {
  const m = prompts[mode];
  return {
    file_lines: m.FILE_LINES,
    system: m.EXPLORER,
    final_answer_description: m.FINAL_ANSWER,
  };
}

export interface FeaturesContextResult {
  summary: string;
  key_files: string[];
  features: string[];
}

export interface FirstPassContextResult {
  summary: string;
  key_files: string[];
  infrastructure: string[];
  dependencies: string[];
  user_stories: string[];
  pages: string[];
}

function makeFad(
  conf: ContextConfig,
  final_answer_description: string | undefined
) {
  let fad = conf.final_answer_description;
  if (final_answer_description) {
    const generic = prompts.generic.FINAL_ANSWER;
    fad = generic + `\n\n` + final_answer_description;
  }
  return fad;
}

export interface Overrides {
  system_prompt?: string;
  final_answer_description?: string;
}

export async function gitsee_context(
  prompt: string | ModelMessage[],
  repoPath: string,
  mode: RepoContextMode = "features",
  overrides?: Overrides
): Promise<string> {
  const startTime = Date.now();
  const CONF = getConfig(mode);
  const provider = process.env.LLM_PROVIDER || "anthropic";
  const apiKey = getApiKeyForProvider(provider);
  const model = await getModel(provider as Provider, apiKey as string);
  let fad = makeFad(CONF, overrides?.final_answer_description);
  if (mode == "services") {
    // replace all instance of MY_REPO_NAME with the actual repo name
    fad = fad.replaceAll(
      "MY_REPO_NAME",
      repoPath.split("/").pop() || "my-repo"
    );
  }
  const tools = {
    repo_overview: tool({
      description:
        "Get a high-level view of the codebase architecture and structure. Use this to understand the project layout and identify where specific functionality might be located. Call this when you need to: 1) Orient yourself in an unfamiliar codebase, 2) Locate which directories/files might contain relevant code for a user's question, 3) Understand the overall project structure before diving deeper. Don't call this if you already know which specific files you need to examine.",
      inputSchema: z.object({}),
      execute: async () => {
        try {
          return await getRepoMap(repoPath);
        } catch (e) {
          return "Could not retrieve repository map";
        }
      },
    }),
    file_summary: tool({
      description:
        "Get a summary of what a specific file contains and its role in the codebase. Use this when you have identified a potentially relevant file and need to understand: 1) What functions/components it exports, 2) What its main responsibility is, 3) Whether it's worth exploring further for the user's question. Only the first 40-100 lines of the file will be returned. Call this with a hypothesis like 'This file probably handles user authentication' or 'This looks like the main dashboard component'. Don't call this to browse random files.",
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
          return getFileSummary(file_path, repoPath, CONF.file_lines);
        } catch (e) {
          return "Bad file path";
        }
      },
    }),
    fulltext_search: tool({
      description:
        "Search the entire codebase for a specific term. Use this when you need to find a specific function, component, or pattern. Call this when the user provided specific text that might be present in the codebase. For example, if the query is 'Add a subtitle to the User Journeys page', you could call this with the query \"User Journeys\". Don't call this if you do not have specific text to search for. It will return a list of files in which the term is found, a count of occurrences, and specific line numbers where the term is found.",
      inputSchema: z.object({
        query: z.string().describe("The term to search for"),
      }),
      execute: async ({ query }: { query: string }) => {
        try {
          return await fulltextSearch(query, repoPath);
        } catch (e) {
          return `Search failed: ${e}`;
        }
      },
    }),
    final_answer: tool({
      // The tool that signals the end of the process
      description: fad,
      inputSchema: z.object({ answer: z.string() }),
      execute: async ({ answer }: { answer: string }) => answer,
    }),
  };
  if (mode === "first_pass") {
    delete (tools as Record<string, any>).fulltext_search;
  }
  const { steps } = await generateText({
    model,
    tools,
    prompt,
    system: overrides?.system_prompt || CONF.system,
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

// infra, dependencies/integratins, user stories, pages
