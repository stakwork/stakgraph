import { z } from "zod";
import { callGenerateObject } from "../aieo/src/stream.js";
import { Provider } from "../aieo/src/provider.js";
import { LLMDecision } from "./types.js";

/**
 * Schema for LLM decision using Zod
 */
const LLMDecisionSchema = z.object({
  actions: z.array(z.enum(["add_to_existing", "create_new", "ignore"])),
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
  constructor(private provider: Provider, private apiKey: string) {}

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
          console.log(
            `   ⚠️  Attempt ${attempt} failed, retrying in ${delay / 1000}s...`
          );
          await new Promise((resolve) => setTimeout(resolve, delay));
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

**Critical: Focus on WHAT users can DO, not HOW it's built**

**IMPORTANT: Be conservative with feature creation. Use THEMES for low-level work, only create FEATURES when there's a clear user capability.**

Features should represent HIGH-LEVEL capabilities:
✅ **User-facing capabilities** - What can users accomplish?
   - Good: "Real-time Chat" (users can chat)
   - Bad: "Pusher WebSocket Infrastructure" (implementation detail)

✅ **Business functionality** - What business value does this provide?
   - Good: "Invoice Generation" (business can invoice)
   - Bad: "PDF Export Service" (implementation detail)

✅ **System-level test infrastructure** (ONLY high-level)
   - Good: "Unit Tests", "Integration Tests", "E2E Tests"
   - Bad: "Workspace Service Tests", "API Tests" (too specific - use themes)

**Features should NEVER:**
❌ Mention specific technologies in the name ("Pusher", "Redis", "React")
❌ Focus on implementation details ("SSE", "WebSocket", "REST API")
❌ Be too narrow ("Navigation UI", "Button Component", "Error Handler")
❌ Include "-service", "-infrastructure", "-system" unless it's genuinely a major capability

**When to use THEMES instead of FEATURES:**
- Low-level technical work → THEME (e.g., "pusher", "websockets", "redis")
- UI components → THEME (e.g., "navigation-ui", "sidebar")
- Specific test files → THEME (e.g., "workspace-tests", "api-tests")
- Infrastructure pieces → THEME (e.g., "error-handling", "logging")

**When to CREATE a FEATURE:**
- Multiple related THEMES have accumulated (e.g., "pusher", "websockets", "real-time" → "Real-time Messaging")
- Clear user-facing capability emerges
- Major business functionality is complete
- System-wide test infrastructure (not specific test files)

**Default: Use THEMES first. Only create FEATURES when patterns clearly emerge.**

**Key points:**
- A PR can belong to MULTIPLE features (e.g., a Google OAuth PR touches both "Real-time Chat" and "Notifications")
- A PR can have BOTH themes AND features (tag technical details as themes, assign to user-facing features)
- When in doubt: TAG with themes, DON'T create a feature yet
- If a PR is purely technical infrastructure with no user/business value, tag it with themes OR ignore it

**Updating Feature Descriptions:**
- Features evolve over time - descriptions should reflect current state, not historical implementation
- If a PR fundamentally changes HOW a feature works (Bitcoin auth → GitHub OAuth, REST → GraphQL, etc.), update the feature description
- Keep descriptions focused on WHAT the feature does for users, not implementation details
- Examples of when to update:
  * Authentication changes from one provider to another
  * Major refactor that changes the nature of the feature
  * Feature gains significant new capabilities that weren't in original description
  * DO NOT INCLUDE INCLUDE VERY SPECIFIC IMPLEMENTATION DETAILS IN THE DESCRIPTION

**Your actions (can combine multiple):**
1. Add to one or more existing features
2. Create new features liberally - we want a comprehensive feature map
3. Update feature descriptions (when implementations fundamentally change)
4. Ignore (ONLY for pure refactoring/infrastructure with zero functional impact)

Think: "What capability does this add to the application? If there's a clear answer, that's probably a feature."

**Using Theme Tags:**
- You'll see a list of recent technical themes (last 100 low-level tags)
- These are lightweight context hints showing recent technical work
- Examples: "jwt", "oauth", "redis", "webhooks", "graphql"
- Use these as clues to recognize patterns across PRs
- When you see related themes accumulating, it might be time to create a feature
- You can add 1-3 theme tags to each PR (optional) - can be NEW or EXISTING themes
- Keep theme tags short and technical (implementation details, protocols, patterns)`;

/**
 * Decision guidelines for the LLM
 */
export const DECISION_GUIDELINES = `## Decision Guidelines

You need to decide:
1. **actions**: Array of actions - can include multiple: "add_to_existing", "create_new", "ignore"
2. **existingFeatureIds**: (if adding to existing) Array of feature IDs to add this PR to
3. **newFeatures**: (if creating new) Array of {name, description} for new features to create
4. **updateFeatures**: (if updating) Array of {featureId, newDescription, reasoning} for features whose descriptions need updating
5. **summary**: Brief description of what this PR does
   - For simple PRs: One clear sentence
   - For large/complex PRs (many files, multiple concerns): Start with a sentence, then add 2-4 bullet points of key changes
   - Example simple: "Adds user profile editing functionality"
   - Example complex: "Major refactor of authentication system:\n- Migrates from Bitcoin signatures to GitHub OAuth\n- Adds session management with Redis\n- Updates all auth middleware\n- Adds comprehensive auth tests"
6. **reasoning**: Quick blurb of why you made this decision
7. **newDeclarations** (optional): If this PR introduces significant NEW code structures, list them organized by file. Skip for PRs that only modify existing code.
   - Format: Array of {file: string, declarations: string[]}
   - Include: endpoints, classes, functions, types, tables, components - whatever is significant
   - Skip: trivial helpers, internal utilities
   - Example: [{file: "src/api/tasks.ts", declarations: ["GET /api/tasks", "POST /api/tasks", "validateTask"]}, {file: "src/types/task.ts", declarations: ["Task", "TaskStatus"]}]

Examples:

**Adding to existing feature - simple PR (good):**
- actions: ["add_to_existing"]
- existingFeatureIds: ["payment-processing"]
- summary: "Adds Stripe webhook handlers for payment events"
- reasoning: "Extends the payment processing capability with webhook support"
- newDeclarations: [{file: "src/api/webhooks.ts", declarations: ["POST /webhooks/stripe", "handleStripeWebhook"]}]

**Adding to existing feature - complex PR (good):**
- actions: ["add_to_existing"]
- existingFeatureIds: ["authentication", "google-integration"]
- summary: "Implements Google OAuth login with full integration:\n- Adds OAuth flow with GitHub provider\n- Implements token refresh logic\n- Updates user model to store OAuth tokens\n- Adds OAuth callback routes"
- reasoning: "Touches both authentication capability and Google integration, significant changes across multiple areas"

**Create new feature - HIGH-LEVEL only (good):**
- actions: ["create_new"]
- newFeatures: [{name: "Task Management", description: "Users can create, assign, track, and complete tasks with deadlines and dependencies"}]
- themes: ["tasks", "assignments", "deadlines"]
- summary: "Task management system"
- reasoning: "Clear user-facing capability for managing tasks"

**Use themes instead of creating feature (good):**
- actions: ["ignore"]
- themes: ["pusher", "websockets", "real-time-infra"]
- summary: "Adds Pusher WebSocket infrastructure"
- reasoning: "Low-level infrastructure work - tracking with themes, will create feature when user capability emerges"

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
Instead: Ignore - this is pure technical work

**Theme Tags (IMPORTANT - use these frequently!):**
- Add 1-3 theme tags to track low-level technical work
- Use themes for ALL implementation details, infrastructure, UI components
- Can be NEW tags or EXISTING tags from recent themes
- Examples: "jwt", "pusher", "redis", "navigation-ui", "workspace-tests"
- Keep them short and technical
- Check recent themes - if you see related themes accumulating, consider creating a feature

**Example: Low-level work → Theme only**
{
  "actions": ["ignore"],
  "themes": ["pusher", "websockets"],
  "summary": "Adds Pusher WebSocket connection handling",
  "reasoning": "Infrastructure work - tracking with themes"
}

**Example: Theme + Feature**
{
  "actions": ["add_to_existing"],
  "existingFeatureIds": ["real-time-chat"],
  "themes": ["pusher", "message-delivery"],
  "summary": "Improves real-time message delivery using Pusher",
  "reasoning": "Extends real-time chat capability; tracking implementation with themes"
}

**Example: Recognizing pattern from accumulated themes**
{
  "actions": ["create_new"],
  "newFeatures": [{
    "name": "Real-time Messaging",
    "description": "Users can send and receive messages in real-time across devices"
  }],
  "themes": ["real-time-complete"],
  "summary": "Completes real-time messaging system",
  "reasoning": "Recent themes show: pusher, websockets, message-delivery, presence - clear user capability now exists"
}

**BAD - Feature too low-level:**
❌ {
  "actions": ["create_new"],
  "newFeatures": [{name: "Pusher WebSocket Infrastructure", description: "..."}]
}
Instead: Use themes ["pusher", "websockets"], wait for user capability to emerge

**BAD - Feature name has implementation details:**
❌ {
  "actions": ["create_new"],
  "newFeatures": [{name: "Redis Caching Service", description: "..."}]
}
Instead: If creating a feature, name it by capability: "Performance Optimization" + themes ["redis", "caching"]`;
