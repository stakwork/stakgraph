import {
  generateObject,
  generateText,
  ModelMessage,
  streamText,
  SystemModelMessage,
  ToolSet,
} from "ai";
import type { StreamTextResult } from "ai";
import { getModel, ModelName, Provider, ThinkingSpeed } from "./provider.js";
import { getProviderPolicy } from "./policy.js";
import { AiUsage } from "./usage.js";

export interface ModelExecutionOptions {
  provider: Provider;
  apiKey: string;
  messages: ModelMessage[];
  tools?: ToolSet;
  thinkingSpeed?: ThinkingSpeed;
  cwd?: string;
  executablePath?: string;
  modelName?: ModelName;
  temperature?: number;
}

export interface GenerateObjectWithUsageOptions {
  provider: Provider;
  apiKey: string;
  prompt?: string;
  messages?: ModelMessage[];
  system?: string | SystemModelMessage | SystemModelMessage[];
  schema: any;
  temperature?: number;
}

export interface GenerateTextWithUsageOptions {
  provider: Provider;
  apiKey: string;
  prompt?: string;
  messages?: ModelMessage[];
  system?: string | SystemModelMessage | SystemModelMessage[];
  tools?: ToolSet;
  thinkingSpeed?: ThinkingSpeed;
  temperature?: number;
}

export interface PrepareModelCallOptions {
  provider: Provider;
  messages?: ModelMessage[];
  prompt?: string;
  system?: string | SystemModelMessage | SystemModelMessage[];
  tools?: ToolSet;
  thinkingSpeed?: ThinkingSpeed;
}

function normalizeSystemMessages(
  system: string | SystemModelMessage | SystemModelMessage[],
): SystemModelMessage[] {
  if (typeof system === "string") {
    return [{ role: "system", content: system }];
  }
  return Array.isArray(system) ? system : [system];
}

function splitLeadingSystemMessages(messages: ModelMessage[]): {
  system?: SystemModelMessage[];
  messages: ModelMessage[];
} {
  const system: SystemModelMessage[] = [];
  let index = 0;
  while (messages[index]?.role === "system") {
    system.push(messages[index] as SystemModelMessage);
    index += 1;
  }
  return {
    system: system.length > 0 ? system : undefined,
    messages: messages.slice(index),
  };
}

function combineSystemMessages(
  explicitSystem: string | SystemModelMessage | SystemModelMessage[] | undefined,
  leadingSystem: SystemModelMessage[] | undefined,
) {
  if (!explicitSystem) return leadingSystem;
  if (!leadingSystem) return explicitSystem;
  return [...normalizeSystemMessages(explicitSystem), ...leadingSystem];
}

function buildMessagesWithSystem(
  provider: Provider,
  system: string | SystemModelMessage | SystemModelMessage[] | undefined,
  messages: ModelMessage[],
): ModelMessage[] {
  const policy = getProviderPolicy(provider);
  const cachedMessages = policy.applyMessages(messages);
  if (!system) return cachedMessages;
  const cachedSystem = policy.applySystem(system);
  const systemMessages = typeof cachedSystem === "string"
    ? [{ role: "system" as const, content: cachedSystem }]
    : Array.isArray(cachedSystem)
    ? cachedSystem
    : [cachedSystem];
  return [...systemMessages, ...cachedMessages] as ModelMessage[];
}

export function prepareModelCall(args: PrepareModelCallOptions) {
  const policy = getProviderPolicy(args.provider, args.thinkingSpeed);
  const inputMessages: ModelMessage[] = args.messages ?? [
    { role: "user", content: args.prompt ?? "" },
  ];
  const { system: leadingSystem, messages } = splitLeadingSystemMessages(inputMessages);
  const system = combineSystemMessages(args.system, leadingSystem);
  return {
    messages: buildMessagesWithSystem(args.provider, system, messages),
    tools: policy.applyTools(args.tools),
    providerOptions: policy.providerOptions,
  };
}

export async function streamTextWithUsage(
  opts: ModelExecutionOptions,
): Promise<StreamTextResult<ToolSet, any>> {
  const model = getModel(opts.provider, {
    apiKey: opts.apiKey,
    cwd: opts.cwd,
    executablePath: opts.executablePath,
    modelName: opts.modelName,
  });
  const prepared = prepareModelCall({
    provider: opts.provider,
    messages: opts.messages,
    tools: opts.tools,
    thinkingSpeed: opts.thinkingSpeed,
  });
  return streamText({
    model,
    tools: prepared.tools,
    messages: prepared.messages,
    temperature: opts.temperature ?? 0,
    providerOptions: prepared.providerOptions as any,
  });
}

export async function consumeStreamTextWithUsage(
  opts: ModelExecutionOptions & { parser?: (fullResponse: string) => void },
): Promise<{ text: string; usage: AiUsage }> {
  const result = await streamTextWithUsage(opts);
  let fullResponse = "";
  for await (const part of result.fullStream) {
    switch (part.type) {
      case "error":
        throw part.error;
      case "text-delta":
        if (opts.parser) {
          opts.parser(fullResponse);
        }
        fullResponse += part.text;
        break;
    }
  }
  const usage = await result.totalUsage;
  const policy = getProviderPolicy(opts.provider, opts.thinkingSpeed);
  return {
    text: fullResponse,
    usage: policy.normalizeUsage(usage),
  };
}

export async function generateObjectWithUsage(
  args: GenerateObjectWithUsageOptions,
): Promise<{ object: any; usage: AiUsage }> {
  const model = getModel(args.provider, args.apiKey);
  const policy = getProviderPolicy(args.provider);
  const messages = args.messages ?? [
    { role: "user" as const, content: args.prompt ?? "" },
  ];
  const prepared = prepareModelCall({
    provider: args.provider,
    messages,
    system: args.system,
  });
  // generateObject uses tool_choice internally; strip thinking from providerOptions
  // to avoid Anthropic error "Thinking may not be enabled when tool_choice forces tool use"
  const providerOptions = { ...prepared.providerOptions } as any;
  if (providerOptions?.anthropic?.thinking) {
    providerOptions.anthropic = { ...providerOptions.anthropic };
    delete providerOptions.anthropic.thinking;
  }
  const { object, usage } = await generateObject({
    model,
    schema: args.schema,
    messages: prepared.messages,
    temperature: args.temperature,
    providerOptions: providerOptions as any,
  });
  return {
    object,
    usage: policy.normalizeUsage(usage),
  };
}

export async function generateTextWithUsage(
  args: GenerateTextWithUsageOptions,
): Promise<{ text: string; usage: AiUsage }> {
  const messages: ModelMessage[] = args.messages ?? [
    { role: "user", content: args.prompt ?? "" },
  ];
  const model = getModel(args.provider, args.apiKey);
  const thinkingSpeed = args.thinkingSpeed ?? "fast";
  const policy = getProviderPolicy(args.provider, thinkingSpeed);
  const prepared = prepareModelCall({
    provider: args.provider,
    messages,
    system: args.system,
    tools: args.tools,
    thinkingSpeed,
  });
  const { text, usage } = await generateText({
    model,
    tools: prepared.tools,
    messages: prepared.messages,
    temperature: args.temperature,
    providerOptions: prepared.providerOptions as any,
  });
  return {
    text,
    usage: policy.normalizeUsage(usage),
  };
}