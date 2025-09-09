import { z } from "zod";
import { Tool } from "../types.js";
import { parseSchema } from "../utils.js";
import { relevant_node_types, NodeType } from "../graph/types.js";
import * as G from "../graph/graph.js";

export const SearchSchema = z.object({
  query: z
    .string()
    .min(1, "Query is required.")
    .describe("Search query to match against snippet names and content."),
  method: z
    .enum(["fulltext", "vector"])
    .optional()
    .default("fulltext")
    .describe(
      "Search method. Fulltext search for exact matches, vector for semantic similarity."
    ),
  concise: z
    .boolean()
    .optional()
    .default(false)
    .describe(
      "Whether to return a concise response (only the name and filename)."
    ),
  node_types: z
    .array(z.enum(relevant_node_types() as [string, ...string[]]))
    .optional()
    .describe("Filter by only these node types."),
  limit: z
    .number()
    .optional()
    .default(25)
    .describe("Limit the number of results."),
  max_tokens: z
    .number()
    .optional()
    .default(100000)
    .describe("Limit the number of tokens."),
  language: z
    .string()
    .optional()
    .describe(
      "Filter nodes by programming language (e.g. 'javascript', 'python', 'typescript')"
    ),
});

export const SearchTool: Tool = {
  name: "stakgraph_search",
  description: "Search for exact matches.",
  inputSchema: parseSchema(SearchSchema),
};

export async function search(args: z.infer<typeof SearchSchema>) {
  console.log("=> Running fulltext search tool with args:", args);
  const result = await G.search(
    args.query,
    args.limit ?? 25,
    (args.node_types as NodeType[]) ?? [],
    args.concise ?? false,
    args.max_tokens ?? 100000,
    args.method ?? "fulltext",
    "snippet",
    false,
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
