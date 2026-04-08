import { z } from "zod";
import { Tool } from "../types.js";
import { parseSchema } from "../utils.js";
import { relevant_node_types, NodeType } from "../../graph/types.js";
import * as G from "../../graph/graph.js";

export const SearchSchema = z.object({
  query: z
    .string()
    .min(1, "Query is required.")
    .describe("Search query to match against snippet names and content."),
  method: z
    .enum(["fulltext", "vector", "hybrid"])
    .optional()
    .default("hybrid")
    .describe(
      "fulltext: keyword/exact matches. vector: semantic similarity. hybrid: combines both using rank fusion for best results."
    ),
  concise: z
    .boolean()
    .optional()
    .default(true)
    .describe(
      "Whether to return a concise response (only the name and filename). Set to false to include full code body."
    ),
  node_types: z
    .array(z.enum(relevant_node_types() as [string, ...string[]]))
    .optional()
    .describe("Filter by only these node types."),
  limit: z
    .number()
    .optional()
    .default(10)
    .describe("Limit the number of results."),
  max_tokens: z
    .number()
    .optional()
    .default(15000)
    .describe("Limit the number of tokens."),
  language: z
    .string()
    .optional()
    .describe(
      "Filter nodes by programming language (e.g. 'javascript', 'python', 'typescript')"
    ),
  skip_node_types: z
    .array(z.string())
    .optional()
    .default([])
    .describe("Node types to exclude from search results (e.g. ['UnitTest', 'IntegrationTest', 'E2etest'])."),
});

export const SearchTool: Tool = {
  name: "stakgraph_search",
  description:
    "Search the code graph by keyword (fulltext), semantic meaning (vector), or both combined (hybrid). Use hybrid for best recall.",
  inputSchema: parseSchema(SearchSchema),
};

export async function search(args: z.infer<typeof SearchSchema>) {
  console.log("=> Running stakgraph search tool with args:", args);
  const result = await G.search(
    args.query,
    args.limit ?? 25,
    (args.node_types as NodeType[]) ?? [],
    args.concise ?? false,
    args.max_tokens ?? 100000,
    args.method ?? "hybrid",
    "snippet",
    (args.skip_node_types as NodeType[]) ?? [],
    args.language
  );
  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(result),
      },
    ],
  };
}
