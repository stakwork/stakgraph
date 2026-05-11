import {
  ModelMessage,
  streamText,
  ToolSet,
  generateObject,
  generateText,
} from "ai";
import {
  Provider,
  getModel,
  getProviderOptions,
  ThinkingSpeed,
  ModelName,
} from "./provider.js";
import { AiUsageWithLegacy, normalizeUsage } from "./usage.js";

interface CallModelOptions {
  provider: Provider;
  apiKey: string;
  messages: ModelMessage[];
  tools?: ToolSet;
  parser?: (fullResponse: string) => void;
  thinkingSpeed?: ThinkingSpeed;
  cwd?: string;
  executablePath?: string;
  modelName?: ModelName;
}

export async function callModel(opts: CallModelOptions): Promise<{
  text: string;
  usage: AiUsageWithLegacy;
}> {
  const {
    provider,
    apiKey,
    messages,
    tools,
    parser,
    thinkingSpeed,
    cwd,
    executablePath,
    modelName,
  } = opts;
  const model = await getModel(provider, {
    apiKey,
    cwd,
    executablePath,
    modelName,
  });
  const providerOptions = getProviderOptions(provider, thinkingSpeed, modelName);
  console.log(`Calling ${provider} with options:`, providerOptions);
  const systemMessages = messages.filter((m) => m.role === "system");
  const nonSystemMessages = messages.filter((m) => m.role !== "system");
  const system = systemMessages.length > 0
    ? systemMessages.map((m) => m.content as string).join("\n")
    : undefined;
  const result = streamText({
    model,
    tools,
    system,
    messages: nonSystemMessages,
    temperature: 0,
    providerOptions: providerOptions as any,
  });
  let fullResponse = "";
  for await (const part of result.fullStream) {
    // console.log(part);
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
  const usage = await result.usage;
  return {
    text: fullResponse,
    usage: normalizeUsage(usage),
  };
}

interface GenerateObjectArgs {
  provider: Provider;
  apiKey: string;
  prompt: string;
  schema: any;
}

export async function callGenerateObject(args: GenerateObjectArgs): Promise<{
  object: any;
  usage: AiUsageWithLegacy;
}> {
  const model = await getModel(args.provider, args.apiKey);
  const providerOptions = getProviderOptions(args.provider, "fast");
  const { object, usage } = await generateObject({
    model,
    schema: args.schema,
    prompt: args.prompt,
    providerOptions: providerOptions as any,
  });
  return {
    object,
    usage: normalizeUsage(usage),
  };
}

interface GenerateTextArgs {
  provider: Provider;
  apiKey: string;
  prompt: string;
  thinkingSpeed?: ThinkingSpeed;
}

export async function callGenerateText(args: GenerateTextArgs): Promise<{
  text: string;
  usage: AiUsageWithLegacy;
}> {
  // Convert prompt to messages format
  const messages: ModelMessage[] = [
    {
      role: "user",
      content: args.prompt,
    },
  ];
  const model = await getModel(args.provider, args.apiKey);
  const providerOptions = getProviderOptions(args.provider, args.thinkingSpeed);
  // Use the existing callModel function
  const { text, usage } = await generateText({
    model,
    messages,
    providerOptions: providerOptions as any,
  });
  return {
    text,
    usage: normalizeUsage(usage),
  };
}
