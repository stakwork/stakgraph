import { generateText, stepCountIs } from "ai";
import * as dotenv from "dotenv";
import { getModelDetails, getProviderOptions } from "../provider.js";
import { createWebSearch, WEB_SEARCH_TOOL_NAME } from "../search.js";

dotenv.config({ path: "../../.env" });

// `npm run try-search -- --citations` to exercise the linkifying mode.
const CITATIONS = process.argv.includes("--citations");

// One prompt, both backends: forces at least one search, then a writeup
// that must carry citations — which is what exercises the index plumbing.
const PROMPT =
  "Search the web for what the Sphinx Chat project is and who builds it. " +
  "Then write exactly 3 short bullet points summarizing what you found. " +
  "Cite a source in every bullet.";

async function run(label: string, modelName: string) {
  console.log(`\n${"=".repeat(70)}\n${label}\n${"=".repeat(70)}`);
  const { model, provider, apiKey, modelId } = getModelDetails(modelName);
  const ws = createWebSearch({ provider, apiKey, citations: CITATIONS });
  console.log(
    `backend=${ws.backend} native=${ws.native} hasTool=${!!ws.tool} snippet=${ws.promptSnippet.length}b`,
  );
  if (!ws.tool) {
    console.error("no tool — missing key for this backend");
    return;
  }

  const started = Date.now();
  const res = await generateText({
    model,
    tools: { [WEB_SEARCH_TOOL_NAME]: ws.tool },
    system: "You are a concise research assistant." + ws.promptSnippet,
    prompt: PROMPT,
    stopWhen: stepCountIs(6),
    providerOptions: getProviderOptions(provider, "fast", modelId) as any,
    onStepFinish: (step) => {
      const calls = step.content
        .filter((c: any) => c.type === "tool-call")
        .map((c: any) => `${c.toolName}(${JSON.stringify(c.input)?.slice(0, 60)})`);
      if (calls.length) console.log(`  step: ${calls.join(", ")}`);
      ws.capture(step.content);
    },
  });
  const elapsed = ((Date.now() - started) / 1000).toFixed(1);

  console.log(`\nsteps=${res.steps.length} elapsed=${elapsed}s captured=${ws.results.length}`);
  for (const r of ws.results.slice(0, 6)) {
    console.log(
      `  [${r.index ?? "-"}] ${r.url}  ${r.text ? `(${r.text.length}b text)` : "(no text)"}`,
    );
  }

  console.log(`\n--- raw model text ---\n${res.text}`);
  const link = ws.formatOutput(res.text);
  console.log(
    `\n--- formatted (citations=${CITATIONS} converted=${link.converted} skipped=${link.skipped}) ---\n${link.content}`,
  );
  const leftover = /<cite/.test(link.content);
  console.log(`\nraw <cite> markup left behind: ${leftover}`);
}

async function main() {
  await run("ANTHROPIC (native web_search)", "sonnet").catch((e) =>
    console.error("anthropic failed:", e?.message || e),
  );
  await run("XAI / GROK (exa shim)", "grok").catch((e) =>
    console.error("grok failed:", e?.message || e),
  );
}

main();
