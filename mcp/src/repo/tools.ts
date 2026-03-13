import { tool, Tool } from "ai";
import { z } from "zod";
import {
  getRepoMap,
  getFileSummary,
  runStakgraph,
  fulltextSearch,
  executeBashCommand,
} from "./bash.js";
import { getProviderTool, Provider } from "../aieo/src/index.js";
import { RepoAnalyzer } from "gitsee/server";
import { listFeatures, getFeatureDocumentation } from "../gitree/service.js";
import { db } from "../graph/neo4j.js";
import { callRemoteAgent, type SubAgent } from "./subagent.js";

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
  | "learn_concept"
  | "learn_concepts"
  | "list_workflows"
  | "learn_workflow"
  | "read_workflow_json"
  | "vector_search";

export type ToolsConfig = Partial<Record<ToolName, string | boolean | null>>;

const TOOL_NAMES: Set<string> = new Set<string>([
  "repo_overview", "file_summary", "recent_commits", "recent_contributions",
  "fulltext_search", "web_search", "bash", "final_answer",
  "ask_clarifying_questions", "list_concepts", "learn_concept",
  "learn_concepts", "list_workflows", "learn_workflow", "read_workflow_json",
  "vector_search",
]);

export type SkillsConfig = Partial<Record<string, boolean>>;

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
  learn_concepts: '', // this is just for naming, to enable the above 2.
  list_workflows: 'List all Workflow nodes in the graph, returning workflow_name and node_key for each.',
  learn_workflow: 'Get the generated documentation for a specific workflow by its node_key.',
  read_workflow_json: 'Read the raw workflow_json property of a Workflow node by its node_key.',
  vector_search: 'Search for code nodes by semantic similarity using vector embeddings. Use this when you want to find code related to a concept or description, even if the exact terms are not present in the code. Returns the node name, type, filename, and line number for each match.',
};

export async function get_tools(
  repoPath: string,
  apiKey: string,
  pat: string | undefined,
  toolsConfig?: ToolsConfig,
  provider?: Provider,
  repos?: string[],
  subAgents?: SubAgent[]
) {
  const repoArr = repoPath.split("/");
  const isMultiRepo = repoPath === "/tmp";
  // For single repo, extract owner/name from path. For multi-repo, these will be empty.
  const repoOwner = isMultiRepo ? "" : repoArr[repoArr.length - 2];
  const repoName = isMultiRepo ? "" : repoArr[repoArr.length - 1];
  // Resolve a repo owner/name pair from an optional "owner/name" string,
  // falling back to the single-repo values or the first entry in repos.
  function resolveRepo(repo?: string): { owner: string; name: string } | null {
    if (repo) {
      const parts = repo.split("/");
      if (parts.length === 2) return { owner: parts[0], name: parts[1] };
    }
    if (!isMultiRepo) return { owner: repoOwner, name: repoName };
    if (repos && repos.length > 0) {
      const parts = repos[0].split("/");
      if (parts.length === 2) return { owner: parts[0], name: parts[1] };
    }
    return null;
  }

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
          return await runStakgraph(file_path, repoPath);
        } catch (e) {
          // stakgraph CLI not available or failed, fall back to reading first lines
          try {
            return getFileSummary(file_path, repoPath, 75);
          } catch (e) {
            return "Bad file path";
          }
        }
      },
    }),
    recent_commits: tool({
      description: defaultDescriptions.recent_commits,
      inputSchema: z.object({
        limit: z.number().optional().default(10),
        repo: z.string().optional().describe("Repository in 'owner/name' format. Required for multi-repo contexts."),
      }),
      execute: async ({ limit, repo }: { limit?: number; repo?: string }) => {
        const resolved = resolveRepo(repo);
        if (!resolved) return "Could not determine repository. Provide a repo in 'owner/name' format.";
        try {
          const analyzer = new RepoAnalyzer({
            githubToken: pat,
          });
          const coms = await analyzer.getRecentCommitsWithFiles(
            resolved.owner,
            resolved.name,
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
        repo: z.string().optional().describe("Repository in 'owner/name' format. Required for multi-repo contexts."),
      }),
      execute: async ({ user, limit, repo }: { user: string; limit?: number; repo?: string }) => {
        const resolved = resolveRepo(repo);
        if (!resolved) return "Could not determine repository. Provide a repo in 'owner/name' format.";
        try {
          const analyzer = new RepoAnalyzer({
            githubToken: pat,
          });
          const output = await analyzer.getContributorPRs(
            resolved.owner,
            resolved.name,
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

  // Always register bash tool — Anthropic uses native provider tool, others use executeBashCommand
  if (bash_tool) {
    // Anthropic: use native provider tool (existing behaviour)
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
  } else {
    // Non-Anthropic: use executeBashCommand directly
    allTools.bash = tool({
      description: defaultDescriptions.bash,
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

  // Conditionally register workflow tools if Workflow nodes exist
  if (db) {
    const workflowCount = await db.count_workflow_nodes();
    if (workflowCount > 0) {
      console.log("===> workflow nodes found, registering workflow tools");
      allTools.list_workflows = tool({
        description: defaultDescriptions.list_workflows,
        inputSchema: z.object({}),
        execute: async () => {
          const workflows = await db.get_all_workflows();
          return workflows.map(w => ({ workflow_name: w.workflow_name, node_key: w.node_key }));
        },
      });
      allTools.learn_workflow = tool({
        description: defaultDescriptions.learn_workflow,
        inputSchema: z.object({ node_key: z.string().describe('node_key of the Workflow') }),
        execute: async ({ node_key }) => {
          const doc = await db.get_workflow_documentation(node_key);
          if (!doc) return { error: 'No documentation found for this workflow' };
          return { body: doc.body };
        },
      });
      allTools.read_workflow_json = tool({
        description: defaultDescriptions.read_workflow_json,
        inputSchema: z.object({ node_key: z.string().describe('node_key of the Workflow') }),
        execute: async ({ node_key }) => {
          const workflow = await db.get_workflow_by_key(node_key);
          if (!workflow) return { error: 'Workflow not found' };
          return { workflow_json: workflow.workflow_json };
        },
      });
    } else {
      console.log("===> no workflow nodes found, skipping workflow tools");
    }
  } else {
    console.log("===> no db found, skipping workflow tools");
  }

  // Conditionally register vector_search tool if embeddings exist
  if (db) {
    const embeddingsCount = await db.count_nodes_with_embeddings();
    if (embeddingsCount > 0) {
      console.log(`===> ${embeddingsCount} nodes with embeddings found, registering vector_search tool`);
      allTools.vector_search = tool({
        description: defaultDescriptions.vector_search,
        inputSchema: z.object({
          query: z.string().describe("A natural language description of the code you are looking for"),
          limit: z.number().optional().default(10).describe("Maximum number of results to return"),
        }),
        execute: async ({ query, limit }: { query: string; limit?: number }) => {
          try {
            const codeNodeTypes = [
              "Function", "Class", "Endpoint", "Datamodel", "Request", "Page",
              "Trait", "Var",
            ];
            const results = await db.vectorSearch(query, limit || 10, codeNodeTypes as any, 0.7);
            return results.map((node) => ({
              name: node.properties.name,
              node_type: node.labels.find(l => l !== "Data_Bank") || node.labels[0],
              file: node.properties.file,
              line: node.properties.start,
              score: node.score,
            }));
          } catch (e) {
            console.error("Error in vector_search:", e);
            return `Vector search failed: ${e}`;
          }
        },
      });
    } else {
      console.log("===> no nodes with embeddings found, skipping vector_search tool");
    }
  } else {
    console.log("===> no db found, skipping vector_search tool");
  }

  // Register sub-agent tools (remote agent delegation)
  if (subAgents && subAgents.length > 0) {
    for (const subAgent of subAgents) {
      // Validate required fields
      if (!subAgent.name || !subAgent.url || !subAgent.apiToken) {
        console.warn(
          `[sub-agent] Skipping invalid sub-agent config: missing name, url, or apiToken/apiKey`,
          { name: subAgent.name, hasUrl: !!subAgent.url, hasToken: !!subAgent.apiToken }
        );
        continue;
      }
      // Validate name is a safe tool identifier (alphanumeric + underscores)
      if (!/^[a-zA-Z][a-zA-Z0-9_]*$/.test(subAgent.name)) {
        console.warn(
          `[sub-agent] Skipping sub-agent with invalid name "${subAgent.name}" — must be alphanumeric/underscores starting with a letter`
        );
        continue;
      }
      // Validate URL is parseable
      try {
        new URL(subAgent.url);
      } catch {
        console.warn(
          `[sub-agent] Skipping sub-agent "${subAgent.name}" with invalid URL: ${subAgent.url}`
        );
        continue;
      }

      const description =
        subAgent.description ||
        `Ask the "${subAgent.name}" sub-agent a question. This delegates to a separate agent instance that may have different codebase context.`;

      // Capture subAgent in closure (name is guaranteed non-empty after validation above)
      const sa = subAgent;
      const saName = sa.name!;
      allTools[saName] = tool({
        description,
        inputSchema: z.object({
          prompt: z
            .string()
            .describe(
              "A focused, specific question or task to delegate to the sub-agent"
            ),
        }),
        execute: async ({ prompt }: { prompt: string }) => {
          console.log(
            `[sub-agent:${sa.name}] Executing with prompt: ${prompt.slice(0, 200)}...`
          );
          return await callRemoteAgent(sa, prompt);
        },
      });
      console.log(`===> registered sub-agent tool: ${sa.name}`);
    }
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
    // concepts
    if (toolsConfig.learn_concept || toolsConfig.list_concepts || toolsConfig.learn_concepts) {
      allTools.list_concepts = tool({
        description: defaultDescriptions.list_concepts,
        inputSchema: z.object({}),
        execute: async () => {
          try {
            const repo = isMultiRepo ? undefined : `${repoOwner}/${repoName}`;
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
      })
      allTools.learn_concept = tool({
        description: defaultDescriptions.learn_concept,
        inputSchema: z.object({
          concept_id: z
            .string()
            .describe("The ID of the concept/feature to learn about"),
        }),
        execute: async ({ concept_id }: { concept_id: string }) => {
          try {
            const repo = isMultiRepo ? undefined : `${repoOwner}/${repoName}`;
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
      })
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

/** Accept ToolsConfig as an object or as a flat string like "key1 true key2 true" */
export function normalizeToolsConfig(
  raw: unknown
): ToolsConfig | undefined {
  if (raw == null) return undefined;
  if (typeof raw === "object" && !Array.isArray(raw)) return raw as ToolsConfig;
  if (typeof raw === "string") {
    const tokens = raw.trim().split(/\s+/);
    const config: ToolsConfig = {};
    let i = 0;
    while (i < tokens.length) {
      const key = tokens[i];
      if (!TOOL_NAMES.has(key)) { i++; continue; }
      const next = tokens[i + 1];
      if (next === "true") {
        (config as any)[key] = true;
        i += 2;
      } else if (next === "false") {
        (config as any)[key] = false;
        i += 2;
      } else if (next === "null") {
        (config as any)[key] = null;
        i += 2;
      } else if (next !== undefined && !TOOL_NAMES.has(next)) {
        // treat as a custom string value
        (config as any)[key] = next;
        i += 2;
      } else {
        // no value follows, default to true
        (config as any)[key] = true;
        i++;
      }
    }
    return Object.keys(config).length > 0 ? config : undefined;
  }
  return undefined;
}