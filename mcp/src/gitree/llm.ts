import { z } from "zod";
import { callGenerateObject } from "../aieo/src/stream.js";
import { Provider } from "../aieo/src/provider.js";
import { LLMDecision } from "./types.js";

/**
 * Schema for LLM decision using Zod
 */
const LLMDecisionSchema = z.object({
  actions: z.array(
    z.enum(["add_to_existing", "create_new", "ignore"])
  ),
  existingFeatureIds: z.array(z.string()).optional(),
  newFeatures: z
    .array(
      z.object({
        name: z.string(),
        description: z.string(),
      })
    )
    .optional(),
  summary: z.string(),
  reasoning: z.string(),
});

/**
 * LLM client for making decisions about PRs
 */
export class LLMClient {
  constructor(
    private provider: Provider,
    private apiKey: string
  ) {}

  /**
   * Ask LLM to decide what to do with a PR
   */
  async decide(prompt: string): Promise<LLMDecision> {
    const result = await callGenerateObject({
      provider: this.provider,
      apiKey: this.apiKey,
      prompt,
      schema: LLMDecisionSchema,
    });

    return result.object as LLMDecision;
  }
}

/**
 * System prompt for the LLM
 */
export const SYSTEM_PROMPT = `You are a software historian analyzing a codebase chronologically.

Your job: Read each PR and decide which feature(s) it belongs to.

**Key points:**
- A PR can belong to MULTIPLE features (e.g., a Google OAuth PR touches both "Authentication" and "Google Integration")
- ONLY create new features for significant capabilities - be conservative!
- Most PRs should add to existing features
- When in doubt, add to existing rather than create new

**Your actions (can combine multiple):**
1. Add to one or more existing features
2. Create one or more new features (RARE!)
3. Ignore (if truly trivial - rare since we pre-filter)

Think: "What conceptual feature(s) does this PR contribute to?"`;

/**
 * Decision format instructions for the LLM
 */
export const DECISION_FORMAT = `## Your Decision

Respond with JSON only:

{
  "actions": ["add_to_existing" | "create_new" | "ignore"],  // Array of actions

  // If adding to existing (can be multiple):
  "existingFeatureIds": ["feature-id-1", "feature-id-2"],

  // If creating new (can be multiple):
  "newFeatures": [
    {
      "name": "Feature Name",
      "description": "2-3 sentences explaining what this feature is"
    }
  ],

  // Always include:
  "summary": "One sentence: what does this PR do?",
  "reasoning": "Why did you make this decision?"
}

Examples:

1. Adding to existing:
{
  "actions": ["add_to_existing"],
  "existingFeatureIds": ["payment-processing"],
  "summary": "Adds Stripe webhook handlers for payment events",
  "reasoning": "Extends existing payment processing with webhook support"
}

2. Multiple features:
{
  "actions": ["add_to_existing"],
  "existingFeatureIds": ["authentication", "google-integration"],
  "summary": "Implements Google OAuth login",
  "reasoning": "Touches both auth system and Google integration"
}

3. Create new:
{
  "actions": ["create_new"],
  "newFeatures": [
    {
      "name": "Real-time Notifications",
      "description": "WebSocket-based real-time notification system that pushes updates to users without polling. Includes presence detection and typing indicators."
    }
  ],
  "summary": "Initial WebSocket notification system",
  "reasoning": "This is a major new capability not covered by existing features"
}

4. Both add and create:
{
  "actions": ["add_to_existing", "create_new"],
  "existingFeatureIds": ["data-export"],
  "newFeatures": [
    {
      "name": "PDF Generation",
      "description": "Server-side PDF generation using Puppeteer. Converts HTML reports to downloadable PDFs with custom styling and pagination."
    }
  ],
  "summary": "Adds PDF export to existing data export feature",
  "reasoning": "Extends data export but PDF generation is significant enough to track separately"
}

5. Ignore:
{
  "actions": ["ignore"],
  "summary": "Updates development dependencies",
  "reasoning": "Pure maintenance work with no functional changes"
}`;
