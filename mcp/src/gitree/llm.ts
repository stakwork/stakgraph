import { z } from "zod";
import { callGenerateObject } from "../aieo/src/stream.js";
import { Provider } from "../aieo/src/provider.js";
import { LLMDecision, Usage } from "./types.js";

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
  async decide(
    prompt: string,
    retries = 3
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

        return {
          decision: result.object as LLMDecision,
          usage: result.usage,
        };
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

Features should represent substantial capabilities (but avoid implementation details in naming):

✅ **Major user-facing capabilities** - What can users accomplish?
   - Good: "Real-time Chat", "Task Management", "File Upload", "Search"
   - Bad: "Pusher WebSocket Infrastructure", "Navigation UI Component"

✅ **Major integrations with external systems**
   - Good: "Stripe Integration", "GitHub OAuth", "AWS S3 Storage", "Slack Notifications"
   - Bad: "Redis Caching", "Database Connection Pool" (internal infrastructure)

✅ **Business functionality**
   - Good: "Invoice Generation", "Payment Processing", "Email Campaigns"
   - Bad: "PDF Export Service" (too narrow - part of Invoice Generation)

✅ **High-level test infrastructure**
   - Good: "Unit Tests", "Integration Tests", "E2E Tests"
   - Bad: "Workspace Service Tests", "API Tests" (too specific - use themes)

**Features should NEVER:**
❌ Mention implementation technologies in the name ("Pusher", "Redis", "WebSocket")
❌ Be too narrow/specific ("Navigation UI", "Button Component", "Error Handler")
❌ Use "-infrastructure" or "-service" in the name unless it's a major external integration

**When to use THEMES instead of FEATURES:**
- Implementation details → THEME (e.g., "pusher", "websockets", "redis")
- UI components without clear capability → THEME (e.g., "navigation-ui", "sidebar")
- Specific test files → THEME (e.g., "workspace-tests", "api-tests")
- Internal infrastructure → THEME (e.g., "error-handling", "logging", "database-migrations")

**When to CREATE a FEATURE:**
- Substantial user-facing work (3+ PRs or major PR completing a capability)
- Major external integration (Stripe, GitHub, AWS, etc.)
- Clear business functionality completed
- System-wide test setup (not just one test file)

**Balance: Create features liberally for substantial work, but use descriptive names focused on capabilities, not implementation.**

**Key points:**
- Create features liberally for substantial work - we want a comprehensive feature map
- A PR can belong to MULTIPLE features (e.g., a Google OAuth PR touches both "Authentication" and "GitHub Integration")
- A PR can have BOTH themes AND features (tag technical details as themes, assign to user-facing features)
- When naming features: Focus on WHAT it does, not HOW (avoid technology names)
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
2. Create new features for:
   - Major user-facing capabilities (Real-time Chat, Task Management, File Upload)
   - Major external integrations (Stripe, GitHub OAuth, AWS S3, Slack)
   - High-level test infrastructure (Unit Tests, Integration Tests, E2E Tests)
3. Update feature descriptions (when implementations fundamentally change)
4. Ignore (ONLY for pure refactoring/infrastructure with zero functional impact)

Think: "What capability does this add? Is it a major integration? If yes to either, create a feature with a good name (no tech details in the name)."

**Using Theme Tags:**
- You'll see a list of recent technical themes (last 100 low-level tags)
- These are lightweight hints showing recent technical work
- Examples: "jwt", "oauth", "redis", "websockets", "navigation-ui", "workspace-tests"
- Use themes to TAG implementation details while creating features for capabilities
- You can add 1-3 theme tags to each PR (optional) - can be NEW or EXISTING themes. 
- Most PRs should only have one theme, only big PRs should have multiple themes.
- Keep theme tags short and technical

**When to use both themes AND features:**
- Create feature for the capability ("Real-time Messaging")
- Add themes for implementation ("pusher", "websockets")
- This gives both high-level (features) and low-level (themes) tracking`;

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

**Create feature for major capability (good):**
- actions: ["create_new"]
- newFeatures: [{name: "Task Management", description: "Users can create, assign, track, and complete tasks with deadlines and dependencies"}]
- themes: ["tasks", "assignments"]
- summary: "Complete task management system"
- reasoning: "Substantial user-facing capability - task management"

**Create feature for major integration (good):**
- actions: ["create_new"]
- newFeatures: [{name: "Stripe Integration", description: "Integrate Stripe for payment processing, subscriptions, and billing"}]
- themes: ["stripe", "payments", "webhooks"]
- summary: "Adds Stripe payment integration"
- reasoning: "Major external integration - Stripe"

**Use themes for implementation details (good):**
- actions: ["ignore"]
- themes: ["pusher", "websockets"]
- summary: "Adds Pusher WebSocket connection handling"
- reasoning: "Infrastructure work - will assign to feature when messaging capability is built"

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

**BAD - Feature name has implementation details:**
❌ {
  "actions": ["create_new"],
  "newFeatures": [{name: "Pusher WebSocket Infrastructure", description: "Pusher-based real-time infrastructure..."}]
}
✅ Instead: {
  "actions": ["create_new"],
  "newFeatures": [{name: "Real-time Messaging", description: "Users can send and receive messages instantly"}],
  "themes": ["pusher", "websockets"]
}

**BAD - Feature too narrow/specific:**
❌ {
  "actions": ["create_new"],
  "newFeatures": [{name: "Navigation UI", description: "Collapsible sidebar navigation..."}]
}
✅ Instead: Add to broader feature or use themes: themes ["navigation-ui", "sidebar"]

**BAD - Test feature too specific:**
❌ {
  "actions": ["create_new"],
  "newFeatures": [{name: "Workspace Service Tests", description: "Tests for workspace service..."}]
}
✅ Instead: {
  "actions": ["add_to_existing"],
  "existingFeatureIds": ["unit-tests"],
  "themes": ["workspace-tests"]
}`;
