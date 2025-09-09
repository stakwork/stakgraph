import { z } from "zod";
import { Tool } from "../types.js";
import { parseSchema } from "../utils.js";
import { all_edge_types, EdgeType } from "../graph/types.js";
import * as G from "../graph/graph.js";

export const GetEdgesSchema = z.object({
  edge_type: z
    .enum(all_edge_types() as [string, ...string[]])
    .optional()
    .describe("Type of edge to retrieve (e.g. 'CALLS', 'CONTAINS', etc)."),
  concise: z
    .boolean()
    .optional()
    .describe(
      "Whether to return a concise response (only the edge type and source/target)."
    ),
  ref_ids: z
    .string()
    .optional()
    .describe("Comma-separated list of ref_ids to retrieve edges for."),
  language: z
    .string()
    .optional()
    .describe(
      "Filter edges by programming language (e.g. 'javascript', 'python', 'typescript')"
    ),
});

export const GetEdgesTool: Tool = {
  name: "stakgraph_edges",
  description: "Retrieve all edges of a specific type from the codebase.",
  inputSchema: parseSchema(GetEdgesSchema),
};

export async function getEdges(args: z.infer<typeof GetEdgesSchema>) {
  const result = await G.get_edges(
    args.edge_type as EdgeType,
    args.concise ?? false,
    args.ref_ids?.split(",") ?? [],
    "snippet",
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
