import { z } from "zod";
import { defineStep } from "../../core.js";

const EXAMPLE = `- id: summarize
  type: llm
  config:
    prompt: "Summarize this: {{ fetch.body }}"
    provider: anthropic
    model: claude-sonnet-4-20250514`;

export default defineStep({
  type: "llm",
  description: `Call an LLM. Output: { text } for free-form, or structured object if "schema" is set. Providers: anthropic, openai.\n\n${EXAMPLE}`,
  input: z.object({
    prompt: z.string(),
    schema: z.any().optional(), // Zod schema for structured output
    provider: z.string().optional(), // e.g. "anthropic", "openai"
    model: z.string().optional(), // override model
  }),
  output: z.any(),
  async run(cfg) {
    // Dynamic import to avoid hard dependency if not using LLM steps
    const { generateText, generateObject } = await import("ai");

    const provider = cfg.provider ?? process.env["VEIN_LLM_PROVIDER"] ?? "anthropic";
    const model = cfg.model ?? process.env["VEIN_LLM_MODEL"];

    // Resolve the AI SDK model
    let aiModel: Parameters<typeof generateText>[0]["model"];

    switch (provider) {
      case "anthropic": {
        const { anthropic } = await import("@ai-sdk/anthropic");
        aiModel = anthropic(model ?? "claude-sonnet-4-20250514");
        break;
      }
      case "openai": {
        const { openai } = await import("@ai-sdk/openai");
        aiModel = openai(model ?? "gpt-4o");
        break;
      }
      default:
        throw new Error(
          `Unknown LLM provider: "${provider}". Supported: anthropic, openai`,
        );
    }

    if (cfg.schema) {
      // Structured output
      const result = await generateObject({
        model: aiModel,
        prompt: cfg.prompt,
        schema: cfg.schema,
      });
      return result.object;
    } else {
      // Free-form text
      const result = await generateText({
        model: aiModel,
        prompt: cfg.prompt,
      });
      return { text: result.text };
    }
  },
});
