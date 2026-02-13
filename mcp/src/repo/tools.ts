import { tool, Tool } from "ai";
import { z } from "zod";
import {
  getRepoMap,
  getFileSummary,
  fulltextSearch,
  executeBashCommand,
} from "./bash.js";
import { getProviderTool, Provider } from "../aieo/src/index.js";
import { RepoAnalyzer } from "gitsee/server";
import { listFeatures, getFeatureDocumentation } from "../gitree/service.js";

type ToolName =
  | "repo_overview"
  | "file_summary"
  | "recent_commits"
  | "recent_contributions"
  | "fulltext_search"
  | "web_search"
  | "bash"
  | "final_answer"
  | "ask_clarifying_questions"
  | "list_concepts"
  | "learn_concept";

export type ToolsConfig = Partial<Record<ToolName, string | boolean | null>>;

const DEFAULT_DESCRIPTIONS: Record<ToolName, string> = {
  repo_overview:
    "Get a high-level view of the codebase architecture and structure. Use this to understand the project layout and identify where specific functionality might be located. Call this when you need to: 1) Orient yourself in an unfamiliar codebase, 2) Locate which directories/files might contain relevant code for a user's question, 3) Understand the overall project structure before diving deeper. Don't call this if you already know which specific files you need to examine.",
  file_summary:
    "Get a summary of what a specific file contains and its role in the codebase. Use this when you have identified a potentially relevant file and need to understand: 1) What functions/components it exports, 2) What its main responsibility is, 3) Whether it's worth exploring further for the user's question. Only the first 40-100 lines of the file will be returned. Call this with a hypothesis like 'This file probably handles user authentication' or 'This looks like the main dashboard component'. Don't call this to browse random files.",
  recent_commits:
    "Query a repo for recent commits. The output is a list of recent commits.",
  recent_contributions:
    "Query a repo for recent PRs by a specific contributor. Input is the contributor's GitHub login. The output is a list of their most recent contributions, including PR titles, issue titles, commit messages, and code review comments.",
  fulltext_search:
    "Search the entire codebase for a specific term, using ripgrep (rg). Use this when you need to find a specific function, component, or file. Call this when the user provided specific text that might be present in the codebase. For example, if the query is 'Add a subtitle to the User Journeys page', you could call this with the query \"User Journeys\". Don't call this if you do not have specific text to search for",
  web_search: "Search the web for information",
  bash: "Execute bash commands",
  final_answer: `Provide the final answer to the user. YOU CAN CALL THIS TOOL AT THE END OF YOUR EXPLORATION.
CRITICAL: Put your ENTIRE response inside the 'answer' parameter as a well-formatted string. Do NOT call this tool with an empty object or without the answer field.

Example usage:
final_answer({ 
  "answer": "Based on my exploration of the codebase, here's how authentication works:\n\n1. Users authenticate via...\n2. The auth flow is handled by...\n3. Key files include..."
})`,
  ask_clarifying_questions: `Ask clarifying questions to the user when you need clarification about design or implementation choices. Call this at the end of your exploration.

QUESTION TYPES:
- single_choice: User picks one option
- multiple_choice: User picks one or more options

BASIC QUESTION EXAMPLE: (simple string options):
[{
  "question": "What type of app?",
  "type": "single_choice",
  "options": ["Web", "Mobile", "Desktop"]
}]

COLOR PICKER EXAMPLE: (use for brand colors, themes, UI colors):
[{
  "question": "Which primary color for your brand?",
  "type": "single_choice",
  "options": ["Sky Blue", "Purple", "Emerald"],
  "questionArtifact": {
    "type": "color_swatch",
    "data": [
      {"label": "Sky Blue", "value": "#0EA5E9"},
      {"label": "Purple", "value": "#8B5CF6"},
      {"label": "Emerald", "value": "#10B981"}
    ]
  }
}]

DIAGRAM QUESTION EXAMPLE: (use to confirm flows, architecture, data models):
[{
  "question": "Does this authentication flow look correct?",
  "type": "single_choice",
  "options": ["Yes, proceed", "No, needs changes"],
  "questionArtifact": {"type": "mermaid", "data": "graph TD\\n  A[Login]-->B{Valid?}\\n  B-->|Yes|C[Dashboard]\\n  B-->|No|D[Error]"}
}]

COMPARISON TABLE EXAMPLE: (use when comparing multiple approaches/technologies):
[{
  "question": "Which real-time approach should we use?",
  "type": "single_choice",
  "options": ["SSE", "WebSockets", "Polling"],
  "questionArtifact": {
    "type": "comparison_table",
    "data": {
      "columns": ["Pros", "Cons"],
      "rows": [
        {"label": "SSE", "description": "Server-Sent Events", "cells": {"Pros": ["Simple", "Auto-reconnect"], "Cons": ["Server→Client only"]}},
        {"label": "WebSockets", "description": "Full duplex", "cells": {"Pros": ["Bi-directional", "Low latency"], "Cons": ["Complex"]}},
        {"label": "Polling", "description": "HTTP requests", "cells": {"Pros": ["Works everywhere"], "Cons": ["High latency"]}}
      ]
    }
  }
}]

Rules:
- Maximum 3 questions
- Use mermaid questionArtifact to visualize and confirm flows before implementing
- Use comparison_table when comparing multiple approaches with pros/cons
- Use color_swatch when asking about colors/themes
- Can combine: use questionArtifact to show a diagram AND rich options for choices`,
  list_concepts:
    "List all high-level concepts (features) that have been learned about this codebase. Use this to discover what areas of functionality have been documented and understand the main components of the system. Returns a list of concepts with their names, descriptions, and metadata about associated PRs and commits.",
  learn_concept:
    "Get detailed information about a specific concept (feature) including its full documentation, associated PRs with summaries, and commits. Use this when you need deep understanding of how a particular feature was implemented and evolved over time.",
};

export function get_tools(
  repoPath: string,
  apiKey: string,
  pat: string | undefined,
  toolsConfig?: ToolsConfig,
  provider?: Provider,
  repos?: string[]
) {
  const repoArr = repoPath.split("/");
  const isMultiRepo = repoPath === "/tmp";
  // For single repo, extract owner/name from path. For multi-repo, these will be empty.
  const repoOwner = isMultiRepo ? "" : repoArr[repoArr.length - 2];
  const repoName = isMultiRepo ? "" : repoArr[repoArr.length - 1];
  const web_search_tool = provider==="anthropic" ? getProviderTool(provider, apiKey, "webSearch") : undefined
  const bash_tool = provider==="anthropic" ? getProviderTool(provider, apiKey, "bash") : undefined

  console.log("===> web_search_tool type:", web_search_tool?.type);
  console.log(
    "===> web_search_tool structure:",
    JSON.stringify(web_search_tool, null, 2)
  );

  const defaultDescriptions: Record<ToolName, string> = {
    ...DEFAULT_DESCRIPTIONS,
    web_search: web_search_tool?.description || DEFAULT_DESCRIPTIONS.web_search,
    bash: bash_tool?.description || DEFAULT_DESCRIPTIONS.bash,
  };

  const allTools: Record<string, Tool<any, any>> = {
    repo_overview: tool({
      description: defaultDescriptions.repo_overview,
      inputSchema: z.object({}),
      execute: async () => {
        try {
          return await getRepoMap(repoPath, repos);
        } catch (e) {
          return "Could not retrieve repository map";
        }
      },
    }),
    file_summary: tool({
      description: defaultDescriptions.file_summary,
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
          return getFileSummary(file_path, repoPath, 75);
        } catch (e) {
          return "Bad file path";
        }
      },
    }),
    recent_commits: tool({
      description: defaultDescriptions.recent_commits,
      inputSchema: z.object({ limit: z.number().optional().default(10) }),
      execute: async ({ limit }: { limit?: number }) => {
        if (isMultiRepo) {
          return "This tool is not available for multi-repository contexts. Use fulltext_search or file_summary instead.";
        }
        try {
          const analyzer = new RepoAnalyzer({
            githubToken: pat,
          });
          const coms = await analyzer.getRecentCommitsWithFiles(
            repoOwner,
            repoName,
            {
              limit: limit || 10,
            }
          );
          return coms;
        } catch (e) {
          console.error("Error retrieving recent commits:", e);
          return "Could not retrieve recent commits";
        }
      },
    }),
    recent_contributions: tool({
      description: defaultDescriptions.recent_contributions,
      inputSchema: z.object({
        user: z.string(),
        limit: z.number().optional().default(5),
      }),
      execute: async ({ user, limit }: { user: string; limit?: number }) => {
        if (isMultiRepo) {
          return "This tool is not available for multi-repository contexts. Use fulltext_search or file_summary instead.";
        }
        try {
          const analyzer = new RepoAnalyzer({
            githubToken: pat,
          });
          const output = await analyzer.getContributorPRs(
            repoOwner,
            repoName,
            user,
            limit || 5
          );
          return output;
        } catch (e) {
          console.error("Error retrieving recent contributions:", e);
          return "Could not retrieve repository map";
        }
      },
    }),
    fulltext_search: tool({
      description: defaultDescriptions.fulltext_search,
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
    list_concepts: tool({
      description: defaultDescriptions.list_concepts,
      inputSchema: z.object({}),
      execute: async () => {
        if (isMultiRepo) {
          return "This tool is not available for multi-repository contexts. Use fulltext_search or file_summary instead.";
        }
        try {
          const repo = `${repoOwner}/${repoName}`;
          const result = await listFeatures(repo);
          return {
            concepts: result.features,
            total: result.total,
            repo,
          };
        } catch (e) {
          console.error("Error listing concepts:", e);
          return "Could not retrieve concepts";
        }
      },
    }),
    learn_concept: tool({
      description: defaultDescriptions.learn_concept,
      inputSchema: z.object({
        concept_id: z
          .string()
          .describe("The ID of the concept/feature to learn about"),
      }),
      execute: async ({ concept_id }: { concept_id: string }) => {
        if (isMultiRepo) {
          return "This tool is not available for multi-repository contexts. Use fulltext_search or file_summary instead.";
        }
        try {
          const repo = `${repoOwner}/${repoName}`;
          const doc = await getFeatureDocumentation(concept_id, repo);
          if (!doc) {
            return { error: "Concept not found" };
          }
          return doc;
        } catch (e) {
          console.error("Error getting concept:", e);
          return "Could not retrieve concept";
        }
      },
    }),
    // final_answer: tool({
    //   description: defaultDescriptions.final_answer,
    //   inputSchema: z.object({
    //     answer: z
    //       .string()
    //       .optional()
    //       .describe("Your complete final answer to the user's question."),
    //   }),
    //   execute: async (body: { answer?: string }) => {
    //     console.log("====> final_answer", JSON.stringify(body, null, 2));
    //     return body.answer || "";
    //   },
    // }),
  };

  // Skip provider-defined tools - not compatible with AI SDK v6 when mixed with regular tools???
  if (web_search_tool) {
    allTools.web_search = web_search_tool as any as Tool<any, any>;
    // Provider tools need to be added differently
  }

  // Implement bash tool using our executeBashCommand
  if (bash_tool) {
    allTools.bash = tool({
      description: bash_tool.description || defaultDescriptions.bash,
      inputSchema: z.object({
        command: z.string().describe("The bash command to execute"),
      }),
      execute: async ({ command }: { command: string }) => {
        try {
          return await executeBashCommand(command, repoPath);
        } catch (e) {
          return `Command execution failed: ${e}`;
        }
      },
    });
  }

  // If no config, return all tools
  if (!toolsConfig) {
    return allTools;
  } else {
    // SPECIAL TOOLS: NOT included by default, but can be enabled with toolsConfig
    if (toolsConfig.ask_clarifying_questions) {
      // Schema for artifact objects (color swatches, diagrams, tables, etc.)
      const artifactSchema = z.object({
        type: z.enum(["mermaid", "comparison_table", "color_swatch"]),
        data: z.record(z.string(), z.any()),
      });

      allTools.ask_clarifying_questions = tool({
        description: defaultDescriptions.ask_clarifying_questions,
        inputSchema: z.object({
          questions: z
            .array(
              z.object({
                question: z.string().describe("The question to ask the user"),
                type: z
                  .enum(["single_choice", "multiple_choice", "color_swatch"])
                  .describe("The type of question"),
                options: z
                  .array(z.string())
                  .optional()
                  .describe(
                    "Options - either simple strings or rich objects with artifacts"
                  ),
                questionArtifact: artifactSchema
                  .optional()
                  .describe(
                    "Artifact to display alongside the question (e.g., mermaid diagram)"
                  ),
              })
            )
            .describe("The questions to ask the user (MAXIMUM 4 QUESTIONS)"),
        }),
        execute: async ({ questions }: { questions: any[] }) => {
          return questions;
        },
      });
    }
  }

  // Start with all tools, then apply config to customize or exclude
  const selectedTools: Record<string, Tool<any, any>> = { ...allTools };

  for (const [toolName, config] of Object.entries(toolsConfig) as [
    ToolName,
    string | boolean | null
  ][]) {
    const originalTool = allTools[toolName];
    if (!originalTool) continue;

    // If config is false, exclude the tool
    if (config === false) {
      delete selectedTools[toolName];
    } else if (typeof config === "string" && config !== "") {
      // If config is a non-empty string, override the description
      selectedTools[toolName] = tool({
        description: config,
        inputSchema: (originalTool as any).inputSchema,
        execute: (originalTool as any).execute,
      }) as Tool<any, any>;
    }
    // If config is null or empty string, keep the default (already in selectedTools)
  }

  console.log("selectedTools", Object.keys(selectedTools));

  return selectedTools;
}

export function getDefaultToolDescriptions(): ToolsConfig {
  return DEFAULT_DESCRIPTIONS;
}
