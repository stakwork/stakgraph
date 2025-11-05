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
  updateFeatures: z
    .array(
      z.object({
        featureId: z.string(),
        newDescription: z.string(),
        reasoning: z.string(),
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

Your job: Read each PR and decide which **user-facing capability or business feature** it belongs to.

**Critical: Focus on WHAT the software does, not HOW it's built**

Features should represent:
✅ **User-facing capabilities** - What can users accomplish?
   Examples: "Authentication System", "Payment Processing", "Real-time Chat", "Task Management"
✅ **Business logic** - What problems does this solve?
   Examples: "Invoice Generation", "Inventory Tracking", "Email Notifications"
✅ **Major integrations** - What external services are integrated?
   Examples: "Stripe Integration", "Google OAuth", "AWS S3 Storage"

Features should NOT be:
❌ **UI Components** - "Sidebar Navigation", "Modal System", "Button Library"
❌ **Technical Infrastructure** - "Redux Store", "API Client", "Error Handling"
❌ **Code Organization** - "Refactoring", "Type Definitions", "Test Setup"
❌ **Performance** - "Caching Layer", "Database Optimization" (unless it's a major capability like "Performance Monitoring")

**Key points:**
- A PR can belong to MULTIPLE features (e.g., a Google OAuth PR touches both "Authentication" and "Google Integration")
- Create new features when you see a clear user-facing capability emerging - be thoughtful but not overly conservative
- If you notice multiple recent PRs working toward the same capability, create a feature for it
- Most PRs should add to existing features
- If a PR is purely technical (refactoring, infrastructure), consider marking it as "ignore" or adding it to an existing relevant feature

**Updating Feature Descriptions:**
- Features evolve over time - descriptions should reflect current state, not historical implementation
- If a PR fundamentally changes HOW a feature works (Bitcoin auth → GitHub OAuth, REST → GraphQL, etc.), update the feature description
- Keep descriptions focused on WHAT the feature does for users, not implementation details
- Examples of when to update:
  * Authentication changes from one provider to another
  * Major refactor that changes the nature of the feature
  * Feature gains significant new capabilities that weren't in original description

**Your actions (can combine multiple):**
1. Add to one or more existing features
2. Create one or more new features (when you see a clear capability)
3. Update feature descriptions (when implementations fundamentally change)
4. Ignore (for purely technical changes with no user/business impact)

Think: "What user capability or business problem does this PR contribute to?"`;

/**
 * Decision guidelines for the LLM
 */
export const DECISION_GUIDELINES = `## Decision Guidelines

You need to decide:
1. **actions**: Array of actions - can include multiple: "add_to_existing", "create_new", "ignore"
2. **existingFeatureIds**: (if adding to existing) Array of feature IDs to add this PR to
3. **newFeatures**: (if creating new) Array of {name, description} for new features to create
4. **updateFeatures**: (if updating) Array of {featureId, newDescription, reasoning} for features whose descriptions need updating
5. **summary**: One sentence describing what this PR does
6. **reasoning**: Why you made this decision

Examples:

**Adding to existing feature (good):**
- actions: ["add_to_existing"]
- existingFeatureIds: ["payment-processing"]
- summary: "Adds Stripe webhook handlers for payment events"
- reasoning: "Extends the payment processing capability with webhook support"

**Multiple features (good):**
- actions: ["add_to_existing"]
- existingFeatureIds: ["authentication", "google-integration"]
- summary: "Implements Google OAuth login"
- reasoning: "Touches both authentication capability and Google integration"

**Create new feature (good - major capability):**
- actions: ["create_new"]
- newFeatures: [{name: "Task Management", description: "Complete task management system allowing users to create, assign, track, and complete tasks with deadlines and dependencies."}]
- summary: "Initial task management system"
- reasoning: "This is a major new user-facing capability - managing tasks is a core business function"

**Ignore - technical infrastructure (good):**
- actions: ["ignore"]
- summary: "Refactors API client to use axios interceptors"
- reasoning: "Pure technical refactoring with no user-visible changes or new capabilities"

**Ignore - UI components without capability (good):**
- actions: ["ignore"]
- summary: "Adds reusable modal component"
- reasoning: "Generic UI component with no specific feature - will be used across features but isn't itself a feature"

**Update feature description (good):**
- actions: ["add_to_existing"]
- existingFeatureIds: ["authentication"]
- updateFeatures: [{featureId: "authentication", newDescription: "GitHub OAuth-based authentication system with JWT tokens and session management", reasoning: "This PR removes Bitcoin-based auth and replaces it entirely with GitHub OAuth - the feature description needs to reflect current implementation"}]
- summary: "Replaces Bitcoin authentication with GitHub OAuth"
- reasoning: "Fundamental change to how authentication works"

**BAD - Don't create features for UI components:**
❌ actions: ["create_new"]
❌ newFeatures: [{name: "Sidebar Navigation", description: "..."}]
Instead: Add to an existing feature that uses it, or ignore if it's generic

**BAD - Don't create features for infrastructure:**
❌ actions: ["create_new"]
❌ newFeatures: [{name: "Error Handling System", description: "..."}]
Instead: Ignore or add to relevant business feature if it improves error handling there

**BAD - Don't create features for code organization:**
❌ actions: ["create_new"]
❌ newFeatures: [{name: "TypeScript Migration", description: "..."}]
Instead: Ignore - this is pure technical work`;
