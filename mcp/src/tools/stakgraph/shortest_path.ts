import { z } from "zod";
import { Tool } from "../types.js";
import { parseSchema } from "../utils.js";
import * as G from "../../graph/graph.js";

export const ShortestPathSchema = z.object({
  start_node_key: z.string().optional().describe("node_key of the start node."),
  end_node_key: z.string().optional().describe("node_key of the end node."),
  start_ref_id: z
    .string()
    .optional()
    .describe("Reference ID of the start node (if known)."),
  end_ref_id: z
    .string()
    .optional()
    .describe("Reference ID of the end node (if known)."),
});

export const ShortestPathTool: Tool = {
  name: "stakgraph_shortest_path",
  description:
    "Find the shortest path between two nodes in the code graph and return the code snippets along that path.",
  inputSchema: parseSchema(ShortestPathSchema),
};

export async function shortestPath(args: z.infer<typeof ShortestPathSchema>) {
  console.log("=> Running shortest_path tool with args:", args);
  const result = await G.get_shortest_path(
    args.start_node_key || "",
    args.end_node_key || "",
    args.start_ref_id || "",
    args.end_ref_id || ""
  );
  return {
    content: [
      {
        type: "text",
        text: result,
      },
    ],
  };
}
