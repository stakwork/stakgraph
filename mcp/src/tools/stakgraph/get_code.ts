import { z } from "zod";
import { Tool } from "../types.js";
import { parseSchema } from "../utils.js";
import * as G from "../../graph/graph.js";
import { GetMapSchema, toMapParams } from "./get_map.js";

export const GetCodeSchema = GetMapSchema;

export const GetCodeTool: Tool = {
  name: "stakgraph_code",
  description:
    "Retrieve actual code snippets from a subtree starting at the specified node.",
  inputSchema: parseSchema(GetCodeSchema),
};

export async function getCode(args: z.infer<typeof GetCodeSchema>) {
  console.log("=> Running get_code tool with args:", args);
  const result = await G.get_code(toMapParams(args));
  return {
    content: [
      {
        type: "text",
        text: result,
      },
    ],
  };
}
