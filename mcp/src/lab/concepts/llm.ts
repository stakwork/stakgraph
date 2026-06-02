import { z } from "zod";
import { callGenerateObject } from "../../aieo/src/stream.js";
import { normalizeUsage } from "../../aieo/src/usage.js";
import { Provider } from "../../aieo/src/provider.js";
import { LLMDecision, Usage } from "./types.js";
import {
  appendMessages,
  appendStepMeta,
  loadSessionMessages,
  loadStepMeta,
} from "../../repo/session.js";

/**
 * Shared documentation guidelines used by bootstrap, summarizer, and exploreNewConcept
 */
export const DOC_GUIDELINES = {
  include: `**What to include**:
- Brief overview (just a few sentences, including any major integrations or dependencies)
- List the 5-15 core files (just paths and 1-line purposes)
- Key concepts/components (high-level only)
- Main API endpoints/functions (names only, no implementations)
- Core data models (names only, brief purpose)
- Essential patterns and gotchas (briefly, if any)`,

  avoid: `**What to AVOID**:
- Code snippets or implementation details
- Long explanations of how things work internally
- Historical information about how it evolved
- Detailed API documentation
- Step-by-step flows unless absolutely essential`,
};

export interface GitreeSessionTracker {
  sessionId: string;
  step: number;
  turn: number;
  cumulativeInput: number;
  cumulativeOutput: number;
}

export function createGitreeSessionTracker(
  sessionId: string,
): GitreeSessionTracker {
  const stepMeta = loadStepMeta(sessionId);
  const lastMeta = stepMeta[stepMeta.length - 1];
  return {
    sessionId,
    step: stepMeta.length,
    turn:
      loadSessionMessages(sessionId).filter((m) => m.role === "user").length +
      1,
    cumulativeInput: lastMeta?.cumulativeInput ?? 0,
    cumulativeOutput: lastMeta?.cumulativeOutput ?? 0,
  };
}

export function appendGitreeLlmExchange(
  tracker: GitreeSessionTracker,
  prompt: string,
  response: string,
  usage: Usage,
  label?: string,
): void {
  const normalizedUsage = normalizeUsage(usage);
  appendMessages(tracker.sessionId, [
    { role: "user", content: prompt },
    { role: "assistant", content: response },
  ]);
  tracker.cumulativeInput += normalizedUsage.inputTokens || 0;
  tracker.cumulativeOutput += normalizedUsage.outputTokens || 0;
  appendStepMeta(tracker.sessionId, [
    {
      step: tracker.step,
      turn: tracker.turn,
      ...(label ? { label } : {}),
      usage: normalizedUsage,
      cumulativeInput: tracker.cumulativeInput,
      cumulativeOutput: tracker.cumulativeOutput,
      toolCalls: [],
      timestamp: new Date().toISOString(),
    },
  ]);
  tracker.step += 1;
  tracker.turn += 1;
}

/**
 * Schema for LLM decision using Zod
 */
const LLMDecisionSchema = z.object({
  actions: z.array(z.enum(["add_to_existing", "create_new", "ignore"])),
  existingConceptIds: z.array(z.string()).optional(),
  newConcepts: z
    .array(
      z.object({
        name: z.string(),
        description: z.string(),
      })
    )
    .optional(),
  updateConcepts: z
    .array(
      z.object({
        conceptId: z.string(),
        newDescription: z.string(),
        reasoning: z.string(),
      })
    )
    .optional(),
  themes: z.array(z.string()).optional(),
  summary: z.string(),
  reasoning: z.string(),
  newDeclarations: z
    .array(
      z.object({
        file: z.string(),
        declarations: z.array(z.string()),
      })
    )
    .optional(),
});

/**
 * LLM client for making decisions about PRs
 */
export class LLMClient {
  constructor(
    private provider: Provider,
    private apiKey: string,
    private sessionTracker?: GitreeSessionTracker,
  ) {}

  /**
   * Ask LLM to decide what to do with a PR
   */
  async decide(
    prompt: string,
    retries = 3,
    sessionId?: string,
    label?: string,
  ): Promise<{ decision: LLMDecision; usage: Usage }> {
    let lastError: Error | undefined;

    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        const result = await callGenerateObject({
          provider: this.provider,
          apiKey: this.apiKey,
          prompt,
          schema: LLMDecisionSchema,
        });

        const response = JSON.stringify(result.object);
        if (this.sessionTracker) {
          appendGitreeLlmExchange(
            this.sessionTracker,
            prompt,
            response,
            result.usage,
            label,
          );
        } else if (sessionId) {
          appendMessages(sessionId, [
            { role: "user", content: prompt },
            { role: "assistant", content: response },
          ]);
        }

        return {
          decision: result.object as LLMDecision,
          usage: result.usage,
        };
      } catch (error) {
        lastError = error as Error;

        if (attempt < retries) {
          const delay = Math.pow(2, attempt) * 1000; // Exponential backoff
          console.log(
            `   ⚠️  Attempt ${attempt} failed, retrying in ${delay / 1000}s...`,
          );
          await new Promise((resolve) => setTimeout(resolve, delay));
        }
      }
    }

    throw new Error(`Failed after ${retries} attempts: ${lastError?.message}`);
  }
}
