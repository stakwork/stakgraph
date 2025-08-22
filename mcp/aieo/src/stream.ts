import { ModelMessage, streamText, ToolSet } from "ai";
import {
  Provider,
  getModel,
  getProviderOptions,
  ThinkingSpeed,
} from "./provider";

interface CallModelOptions {
  provider: Provider;
  apiKey: string;
  messages: ModelMessage[];
  tools?: ToolSet;
  parser?: (fullResponse: string) => void;
  thinkingSpeed?: ThinkingSpeed;
  cwd?: string;
}

export async function callModel(opts: CallModelOptions): Promise<string> {
  const { provider, apiKey, messages, tools, parser, thinkingSpeed, cwd } =
    opts;
  const model = await getModel(provider, apiKey, cwd);
  const providerOptions = getProviderOptions(provider, thinkingSpeed);
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
        fullResponse += part.text;
        break;
    }
  }
  return fullResponse;
}
