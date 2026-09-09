import { generateText, stepCountIs } from "ai";
import * as dotenv from "dotenv";
import { getModelDetails, getProviderOptions } from "../provider.js";
import { createWebSearch, linkifyCitations, WEB_SEARCH_TOOL_NAME } from "../search.js";

dotenv.config({ path: "../../.env" });

const PROMPT =
  "Search the web for what the Sphinx Chat project is and who builds it. " +
  "Then write exactly 3 short bullet points summarizing what you found. Cite a source in every bullet.";

async function once(modelName: string) {
  const { model, provider, apiKey, modelId } = getModelDetails(modelName);
  const ws = createWebSearch({ provider, apiKey });
  const res = await generateText({
    model,
    tools: { [WEB_SEARCH_TOOL_NAME]: ws.tool },
    system: "You are a concise research assistant." + ws.promptSnippet,
    prompt: PROMPT,
    stopWhen: stepCountIs(6),
    providerOptions: getProviderOptions(provider, "fast", modelId) as any,
    onStepFinish: (s) => ws.capture(s.content),
  });
  const { converted, skipped } = linkifyCitations(res.text, ws.results);
  return { converted, skipped, captured: ws.results.length };
}

for (const [label, m] of [["anthropic/sonnet", "sonnet"], ["xai/grok", "grok"]] as const) {
  const rows: string[] = [];
  for (let i = 0; i < 3; i++) {
    try {
      const r = await once(m);
      rows.push(`converted=${r.converted} skipped=${r.skipped} captured=${r.captured}`);
    } catch (e: any) {
      rows.push(`ERROR ${e?.message?.slice(0, 60)}`);
    }
  }
  console.log(`\n### ${label}\n  ` + rows.join("\n  "));
}
