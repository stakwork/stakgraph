import { generateText } from "ai";
import * as dotenv from "dotenv";
import { getModelDetails, getProviderOptions } from "../provider.js";
import { normalizeUsage, withProviderCacheUsage } from "../usage.js";

dotenv.config({ path: "../../.env" });

// Two identical calls with a >1k-token shared prefix: the first is a plain
// smoke test of the xai provider path, the second should surface Grok's
// automatic prefix cache as cache_read in the normalized usage.
async function doTheThing() {
  if (!process.env.XAI_API_KEY) {
    console.error("Missing XAI_API_KEY");
    return;
  }
  const { model, provider, modelId, contextLimit } = getModelDetails(
    "grok-4-fast-non-reasoning"
  );
  console.log({ provider, modelId, contextLimit });

  const system =
    "You are a terse assistant. Reply with one word only.\n" +
    "Filler context (ignore): " + "lorem ipsum dolor sit amet ".repeat(300);

  for (const label of ["first call", "second call"]) {
    const res = await generateText({
      model,
      system,
      prompt: "Say OK",
      providerOptions: getProviderOptions(provider, "fast", modelId) as any,
      maxOutputTokens: 512,
    });
    const usage = withProviderCacheUsage(
      normalizeUsage(res.usage as any),
      res.providerMetadata as Record<string, any> | undefined
    );
    console.log(label, {
      text: res.text.slice(0, 40),
      rawUsage: res.usage,
      normalized: usage,
      providerMetadata: res.providerMetadata,
    });
  }
}

doTheThing().catch((error) => {
  console.error("Error occurred while calling model:", error);
});
