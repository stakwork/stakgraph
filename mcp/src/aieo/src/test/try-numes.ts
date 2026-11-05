import { generateText, stepCountIs, streamText, tool } from "ai";
import * as dotenv from "dotenv";
import { z } from "zod";
import { computeSessionCost, getProviderOptions } from "../provider.js";
import { resolveModel } from "../resolve.js";
import { normalizeUsage } from "../usage.js";

dotenv.config({ path: "../../.env" });

// Numes against the real API: a tool round trip (the model calls the tool,
// then answers from its result) and a streamed reply, for each model named
// on the command line — by default the free ones' alias, "numes".
// NUMES_API_KEY from env or mcp/.env.
async function tryModel(name: string) {
  const r = await resolveModel({ model: name });
  console.log(`\n== ${r.name}`, { contextLimit: r.contextLimit, maxOutputTokens: r.maxOutputTokens });

  const started = Date.now();
  const res = await generateText({
    model: r.model,
    prompt: "What is the weather in Paris? Use the tool, then answer in one sentence.",
    tools: {
      get_weather: tool({
        description: "The current weather for a city",
        inputSchema: z.object({ city: z.string() }),
        execute: async ({ city }) => ({ city, sky: "clear", celsius: 21 }),
      }),
    },
    stopWhen: stepCountIs(3),
    maxOutputTokens: r.maxOutputTokens,
    providerOptions: getProviderOptions(r.provider, "fast", r.name) as any,
  });
  const calls = res.steps.flatMap((s) => s.toolCalls.map((c) => `${c.toolName}(${JSON.stringify(c.input)})`));
  const usage = normalizeUsage(res.totalUsage as any);
  console.log("tools:", {
    ms: Date.now() - started,
    calls,
    text: res.text.slice(0, 120),
    usage,
    cost: computeSessionCost(r.provider, usage, r.modelId),
  });

  const stream = streamText({ model: r.model, prompt: "Count from 1 to 5, comma separated.", maxOutputTokens: 2000 });
  let chunks = 0;
  for await (const _ of stream.textStream) chunks++;
  console.log("stream:", { chunks, text: (await stream.text).slice(0, 80), usage: await stream.usage });
}

async function main() {
  const names = process.argv.slice(2);
  for (const name of names.length ? names : ["numes"]) await tryModel(name);
}

main().catch((error) => {
  console.error("Error occurred while calling model:", error);
  process.exitCode = 1;
});
