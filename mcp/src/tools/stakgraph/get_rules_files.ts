import { z } from "zod";
import { Tool } from "../types.js";
import { parseSchema } from "../utils.js";
import * as G from "../../graph/graph.js";

export const GetRulesFilesSchema = z.object({});

export const RULES_PATTERNS = [
  ".windsurfrules",
  ".cursorrules",
  ".aiderules",
  ".aider.conf.md",
  ".clinerules",
  ".continuerules",
  "CLAUDE.md",
  ".cursor/rules/",
  ".github/copilot-instructions.md",
  "AGENTS.md",
  "AI_INSTRUCTIONS.md",
  "INSTRUCTIONS.md",
  ".goosehints",
  ".ai/",
  "README.md",
];

export const GetRulesFilesTool: Tool = {
  name: "stakgraph_rules_files",
  description:
    "Fetch rules files (e.g. .cursorrules, .windsurfrules, CLAUDE.md, etc.) as code snippets.",
  inputSchema: parseSchema(GetRulesFilesSchema),
};

export async function getRulesFiles() {
  const { files_found, snippets } = await G.get_rules_files();
  return {
    content: [
      {
        type: "text",
        text: JSON.stringify({
          patterns_searched: RULES_PATTERNS,
          files_found,
          snippets,
        }),
      },
    ],
  };
}
