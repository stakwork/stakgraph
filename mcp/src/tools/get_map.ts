import { z } from "zod";
import { Tool } from "./index.js";
import { parseSchema } from "./utils.js";
import * as G from "../graph/graph.js";

export function toMapParams(args: z.infer<typeof GetMapSchema>): G.MapParams {
  return {
    node_type: args.node_type || "",
    name: args.name || "",
    ref_id: args.ref_id || "",
    tests: args.tests ?? false,
    depth: args.depth ?? 10,
    direction: (args.direction || "down") as G.Direction,
    trim: args.trim?.split(",") || [],
  };
}

export const GetMapSchema = z.object({
  node_type: z.string().optional().describe("Type of the node."),
  name: z.string().optional().describe("Name of the node to map from."),
  ref_id: z
    .string()
    .optional()
    .describe(
      "Reference ID of the node (either ref_id or node_type+name must be provided)."
    ),
  tests: z
    .boolean()
    .optional()
    .describe("Whether to include tests in the map."),
  depth: z
    .number()
    .optional()
    .describe("Depth of the subtree to retrieve (default: 1)."),
  direction: z
    .enum(["up", "down", "both"] as const)
    .optional()
    .describe("Direction of relationships to traverse."),
  trim: z
    .string()
    .optional()
    .describe("Nodes to trim by name (comma separated list)."),
});

export const GetMapTool: Tool = {
  name: "get_map",
  description:
    "Generate a visual map/tree of code relationships from a specified node.",
  inputSchema: parseSchema(GetMapSchema),
};

export async function getMap(args: z.infer<typeof GetMapSchema>) {
  console.log("=> Running get_map tool with args:", args);
  const result = await G.get_map(toMapParams(args));
  return {
    content: [
      {
        type: "html",
        html: result,
      },
    ],
  };
}
