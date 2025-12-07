import { z } from "zod";
import { Tool } from "../types.js";
import { parseSchema } from "../utils.js";
import { get_context_explore } from "../explore/tool.js";

export const ExploreSchema = z.object({
  prompt: z
    .string()
    .min(1, "Prompt is required.")
    .describe("User prompt to kick off an exploration of the codebase"),
});

export const ExploreTool: Tool = {
  name: "stakgraph_explore",
  description: "Explore codebase and get a summary back",
  inputSchema: parseSchema(ExploreSchema),
};

export async function explore(args: z.infer<typeof ExploreSchema>) {
  console.log("=> Running explore tool with args:", args);
  const result = await get_context_explore(args.prompt);
  return {
    content: [
      {
        type: "text",
        text: JSON.stringify({ result: result.final, usage: result.usage }),
      },
    ],
  };
}
