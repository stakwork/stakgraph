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
  async decide(prompt: string, retries = 3): Promise<LLMDecision> {
    let lastError: Error | undefined;

    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        const result = await callGenerateObject({
          provider: this.provider,
          apiKey: this.apiKey,
          prompt,
          schema: LLMDecisionSchema,
        });

        return result.object as LLMDecision;
      } catch (error) {
        lastError = error as Error;

        if (attempt < retries) {
          const delay = Math.pow(2, attempt) * 1000; // Exponential backoff
          console.log(`   ⚠️  Attempt ${attempt} failed, retrying in ${delay/1000}s...`);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }

    throw new Error(`Failed after ${retries} attempts: ${lastError?.message}`);
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
 * Decision guidelines for the LLM
 */
export const DECISION_GUIDELINES = `## Decision Guidelines

You need to decide:
1. **actions**: Array of actions - can include multiple: "add_to_existing", "create_new", "ignore"
2. **existingFeatureIds**: (if adding to existing) Array of feature IDs to add this PR to
3. **newFeatures**: (if creating new) Array of {name, description} for new features to create
4. **summary**: One sentence describing what this PR does
5. **reasoning**: Why you made this decision

Examples:

**Adding to existing feature:**
- actions: ["add_to_existing"]
- existingFeatureIds: ["payment-processing"]
- summary: "Adds Stripe webhook handlers for payment events"
- reasoning: "Extends existing payment processing with webhook support"

**Multiple features:**
- actions: ["add_to_existing"]
- existingFeatureIds: ["authentication", "google-integration"]
- summary: "Implements Google OAuth login"
- reasoning: "Touches both auth system and Google integration"

**Create new feature:**
- actions: ["create_new"]
- newFeatures: [{name: "Real-time Notifications", description: "WebSocket-based real-time notification system..."}]
- summary: "Initial WebSocket notification system"
- reasoning: "This is a major new capability not covered by existing features"

**Both add and create:**
- actions: ["add_to_existing", "create_new"]
- existingFeatureIds: ["data-export"]
- newFeatures: [{name: "PDF Generation", description: "Server-side PDF generation..."}]
- summary: "Adds PDF export to existing data export feature"
- reasoning: "Extends data export but PDF generation is significant enough to track separately"

**Ignore:**
- actions: ["ignore"]
- summary: "Updates development dependencies"
- reasoning: "Pure maintenance work with no functional changes"`;
