import { tool, Tool } from "ai";
import { z } from "zod";
import { getRepoMap, getFileSummary, fulltextSearch } from "./bash.js";
import { getProviderTool } from "../aieo/src/index.js";
import { RepoAnalyzer } from "gitsee/server";

type ToolName =
  | "repo_overview"
  | "file_summary"
  | "recent_commits"
  | "recent_contributions"
  | "fulltext_search"
  | "web_search"
  | "final_answer";

type ToolOptions = {
  name: ToolName;
  description: string;
};

export function get_tools(
  repoPath: string,
  apiKey: string,
  pat: string | undefined,
  tools_opts?: ToolOptions[]
): Record<ToolName, Tool> {
  const repoArr = repoPath.split("/");
  const repoOwner = repoArr[repoArr.length - 2];
  const repoName = repoArr[repoArr.length - 1];
  const web_search = getProviderTool("anthropic", apiKey, "webSearch");
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
          return getFileSummary(file_path, repoPath, 75);
        } catch (e) {
          return "Bad file path";
        }
      },
    }),
    recent_commits: tool({
      description:
        "Query a repo for recent commits. The output is a list of recent commits.",
      inputSchema: z.object({ limit: z.number().optional().default(10) }),
      execute: async ({ limit }: { limit?: number }) => {
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
      description:
        "Query a repo for recent PRs by a specific contributor. Input is the contributor's GitHub login. The output is a list of their most recent contributions, including PR titles, issue titles, commit messages, and code review comments.",
      inputSchema: z.object({
        user: z.string(),
        limit: z.number().optional().default(5),
      }),
      execute: async ({ user, limit }: { user: string; limit?: number }) => {
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
      description:
        "Search the entire codebase for a specific term. Use this when you need to find a specific function, component, or file. Call this when the user provided specific text that might be present in the codebase. For example, if the query is 'Add a subtitle to the User Journeys page', you could call this with the query \"User Journeys\". Don't call this if you do not have specific text to search for",
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
    web_search,
    final_answer: tool({
      // The tool that signals the end of the process
      description: `Provide the final answer to the user. YOU **MUST** CALL THIS TOOL AT THE END OF YOUR EXPLORATION.`,
      inputSchema: z.object({ answer: z.string() }),
      execute: async ({ answer }: { answer: string }) => answer,
    }),
  };
  return tools;
}
