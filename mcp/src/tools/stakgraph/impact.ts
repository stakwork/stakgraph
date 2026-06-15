import { z } from "zod";
import { Tool } from "../types.js";
import { parseSchema } from "../utils.js";
import * as G from "../../graph/graph.js";

export const ImpactSchema = z.object({
  files: z
    .array(z.string())
    .optional()
    .describe(
      "File paths to analyze impact for. All code nodes in these files become seeds for upstream traversal."
    ),
  name: z
    .string()
    .optional()
    .describe("Name of a specific node to analyze impact for."),
  node_type: z
    .string()
    .optional()
    .describe(
      "Filter seed nodes to this type (e.g. Function, Endpoint, Class)."
    ),
  ref_id: z
    .string()
    .optional()
    .describe("ref_id of a specific node to start from."),
  depth: z
    .number()
    .optional()
    .default(3)
    .describe("How many hops upstream to traverse (default: 3)."),
  tests: z
    .boolean()
    .optional()
    .default(true)
    .describe("Whether to include affected tests (default: true)."),
});

export const ImpactTool: Tool = {
  name: "stakgraph_impact",
  description:
    "Find all code affected by changes to the given files or functions. " +
    "Returns affected endpoints, tests, callers, and other upstream dependents. " +
    "Use this to understand the blast radius of a change.",
  inputSchema: parseSchema(ImpactSchema),
};

export async function impact(args: z.infer<typeof ImpactSchema>) {
  console.log("=> Running impact tool with args:", args);

  const hasFiles = args.files && args.files.length > 0;
  const hasName = args.name && args.name.length > 0;
  const hasRefId = args.ref_id && args.ref_id.length > 0;

  if (!hasFiles && !hasName && !hasRefId) {
    return {
      isError: true,
      content: [
        {
          type: "text",
          text: "Error: provide at least one of files, name, or ref_id",
        },
      ],
    };
  }

  const result = await G.get_impact({
    files: args.files || [],
    name: args.name || "",
    node_type: args.node_type || "",
    ref_id: args.ref_id || "",
    depth: args.depth!,
    tests: args.tests ?? true,
  });

  return {
    content: [
      {
        type: "text",
        text: result,
      },
    ],
  };
}
