import { PM, SENIOR_DEV, JUNIOR_DEV, CEO, AGENT } from "./prompts.js";
import { getApiKeyForProvider, Provider } from "../../../aieo/src/provider.js";
import { callGenerateObject } from "../../../aieo/src/stream.js";
import { z } from "zod";

type Persona = "PM" | "SeniorDev" | "JuniorDev" | "CEO" | "Agent";

export const personas: Record<Persona, (q: string, a: string) => string> = {
  PM: PM,
  SeniorDev: SENIOR_DEV,
  JuniorDev: JUNIOR_DEV,
  CEO: CEO,
  Agent: AGENT,
};

interface QuestionAndAnswer {
  question: string;
  answer: string;
}

export async function rephraseHint(
  persona: Persona,
  question: string,
  answer: string,
  llm_provider?: string
): Promise<QuestionAndAnswer> {
  const provider = llm_provider ? llm_provider : "anthropic";
  const apiKey = getApiKeyForProvider(provider);
  const schema = z.object({
    question: z.string(),
    answer: z.string(),
  });
  const prompt = personas[persona](question, answer);
  return await callGenerateObject({
    provider: provider as Provider,
    apiKey,
    prompt,
    schema,
  });
}
