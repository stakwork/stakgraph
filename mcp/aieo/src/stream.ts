import { CoreMessage, streamText, ToolSet } from "ai";
import { Provider, getModel, getProviderOptions } from "./provider";

export async function callModel(
  provider: Provider,
  apiKey: string,
  messages: CoreMessage[],
  tools?: ToolSet,
  parser?: (fullResponse: string) => void
): Promise<string> {
  const model = await getModel(provider, apiKey);
  const providerOptions = getProviderOptions(provider);
  console.log(`Calling ${provider} with options:`, providerOptions);
  const result = streamText({
    model,
    tools,
    messages,
    temperature: 0,
    providerOptions: providerOptions as any,
  });
  let fullResponse = "";
  for await (const part of result.fullStream) {
    switch (part.type) {
      case "error":
        throw part.error;
      case "text-delta":
        if (parser) {
          parser(fullResponse);
        }
        fullResponse += part.textDelta;
        break;
    }
  }
  return fullResponse;
}
