import { z } from "zod";
import { Tool } from "../types.js";
import { parseSchema } from "../utils.js";
import * as G from "../../graph/graph.js";
import { NodeType } from "../../graph/types.js";

export const RepoMapSchema = z.object({
  name: z
    .string()
    .optional()
    .describe("Name of the Repository node to map from."),
  ref_id: z
    .string()
    .optional()
    .describe("Reference ID of the Repository node."),
  node_type: z
    .string()
    .optional()
    .default("Repository")
    .describe(
      "Type of the node to map from. Either Repository, Directory, or File."
    ),
});

export const RepoMapTool: Tool = {
  name: "stakgraph_repo_map",
  description:
    "Generate a visual map/tree of the directories and files in the repo. If no name or ref_id is provided, it will return a repo map for all repositories.",
  inputSchema: parseSchema(RepoMapSchema),
};

export async function repoMap(args: z.infer<typeof RepoMapSchema>) {
  console.log("=> Running repo_map tool with args:", args);
  const node_type = args.node_type as NodeType;
  const result = await G.get_repo_map(
    args.name || "",
    args.ref_id || "",
    node_type || "Repository"
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
