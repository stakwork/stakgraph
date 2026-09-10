import { generateText, stepCountIs } from "ai";
import * as dotenv from "dotenv";
import { getModelDetails, getProviderOptions } from "../provider.js";
import { createWebFetch, WEB_FETCH_TOOL_NAME } from "../fetch.js";

dotenv.config({ path: "../../.env" });

// `npm run try-fetch -- https://some.url` to point both backends at a page.
const URL_TO_FETCH = process.argv.find((a) => /^https?:\/\//.test(a)) ?? "https://sphinx.chat";

// The URL is in the prompt on purpose: Anthropic's native tool only
// fetches URLs that already appeared in the conversation.
const PROMPT =
  `Fetch ${URL_TO_FETCH} and write exactly 3 short bullet points about what the page says. ` +
  "Quote the page title in the first bullet.";

async function run(label: string, modelName: string) {
  console.log(`\n${"=".repeat(70)}\n${label}\n${"=".repeat(70)}`);
  const { model, provider, apiKey, modelId } = getModelDetails(modelName);
  const wf = createWebFetch({ provider, apiKey, maxCharacters: 12_000 });
  console.log(`backend=${wf.backend} native=${wf.native} hasTool=${!!wf.tool}`);
  if (!wf.tool) {
    console.error("no tool — missing key for this backend");
    return;
  }

  const started = Date.now();
  const res = await generateText({
    model,
    tools: { [WEB_FETCH_TOOL_NAME]: wf.tool },
    system: "You are a concise research assistant.",
    prompt: PROMPT,
    stopWhen: stepCountIs(4),
    providerOptions: getProviderOptions(provider, "fast", modelId) as any,
    onStepFinish: (step) => {
      const calls = step.content
        .filter((c: any) => c.type === "tool-call")
        .map((c: any) => `${c.toolName}(${JSON.stringify(c.input)?.slice(0, 80)})`);
      if (calls.length) console.log(`  step: ${calls.join(", ")}`);
      const errors = step.content
        .filter((c: any) => c.type === "tool-result" && (c.isError || c.output?.error))
        .map((c: any) => JSON.stringify(c.output ?? c.result)?.slice(0, 200));
      if (errors.length) console.log(`  tool errors: ${errors.join(" | ")}`);
      wf.capture(step.content);
    },
  });
  const elapsed = ((Date.now() - started) / 1000).toFixed(1);

  console.log(`\nsteps=${res.steps.length} elapsed=${elapsed}s captured=${wf.results.length}`);
  for (const r of wf.results) {
    console.log(
      `  ${r.url}  title=${JSON.stringify(r.title)} type=${r.mediaType} ` +
        `${r.text ? `(${r.text.length}b text${r.truncated ? ", truncated" : ""})` : "(no text)"}`,
    );
  }
  console.log(`\n--- model text ---\n${res.text}`);
}

async function main() {
  await run("ANTHROPIC (native web_fetch)", "sonnet").catch((e) =>
    console.error("anthropic failed:", e?.message || e),
  );
  await run("XAI / GROK (http shim)", "grok").catch((e) =>
    console.error("grok failed:", e?.message || e),
  );
}

main();
