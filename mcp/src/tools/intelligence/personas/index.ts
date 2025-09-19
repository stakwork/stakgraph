import { PM, SENIOR_DEV, JUNIOR_DEV, CEO, AGENT } from "./prompts.js";

type Persona = "PM" | "SeniorDev" | "JuniorDev" | "CEO" | "Agent";

export const personas: Record<Persona, (q: string, a: string) => string> = {
  PM: PM,
  SeniorDev: SENIOR_DEV,
  JuniorDev: JUNIOR_DEV,
  CEO: CEO,
  Agent: AGENT,
};
